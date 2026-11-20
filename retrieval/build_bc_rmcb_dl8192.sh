#!/bin/bash
# Build the Reason-ModernColBERT PLAID index of the BrowseComp-Plus corpus at document length 8192 (needs >30 GB GPU free).
# Runs scripts/retrieval_eval.py in $RETRIEVAL_EVAL_ROOT with $PYLATE_PYTHON (configs/paths.env); see reindex_rmcb_8192.py
# for a resumable alternative. Usage: bash retrieval/build_bc_rmcb_dl8192.sh
set -u
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set -a; . "${_REPO}/configs/paths.env"; set +a
cd "${RETRIEVAL_EVAL_ROOT:?set RETRIEVAL_EVAL_ROOT in configs/paths.env}"
: "${PYLATE_PYTHON:?set PYLATE_PYTHON in configs/paths.env}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/bc_rmcb_dl8192.log; }

need=70   # GB of free disk
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$free" -gt "$need" ] || { say "ABORT: only ${free}G disk free, want >${need}G"; exit 1; }

# Free GPU memory from the CUDA runtime (independent of nvidia-smi).
gpufree=$("${PYLATE_PYTHON}" -c \
  "import torch;f,_=torch.cuda.mem_get_info(0);print(int(f/1048576))" 2>/dev/null | tail -1)
[ -n "$gpufree" ] && [ "$gpufree" -gt 30000 ] || {
  say "ABORT: only ${gpufree:-0}MiB GPU free, need >30000 for PLAID construction."
  exit 1; }
say "disk free ${free}G, GPU free ${gpufree}MiB"
say "building bc_rmcb_dl8192 — Reason-ModernColBERT, doc_len 8192, query_len 256, 100,195 docs"

"${PYLATE_PYTHON}" scripts/retrieval_eval.py \
  --task browsecomp --retriever pylate --model lightonai/Reason-ModernColBERT \
  --k 100 --doc-len 8192 --query-len 256 --batch 4 --qbatch 4 \
  --plaid-batch 4096 --n-full 512 --trust-remote-code \
  --index-name bc_rmcb_dl8192 --out logs/bc_rmcb_dl8192_retrieval.json \
  > logs/bc_rmcb_dl8192_build.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s logs/bc_rmcb_dl8192_retrieval.json ]; then
  say "FAILED (exit $rc) — see logs/bc_rmcb_dl8192_build.log"; tail -5 logs/bc_rmcb_dl8192_build.log | tee -a logs/bc_rmcb_dl8192.log; exit 1
fi
say "index size: $(du -sh .pylate_idx/bc_rmcb_dl8192 2>/dev/null | cut -f1)"
say "RETRIEVAL-ONLY GATE (830 queries, doc_len 8192):"
grep -aE "^  (NDCG|Recall)@" logs/bc_rmcb_dl8192_build.log | tail -6 | tee -a logs/bc_rmcb_dl8192.log
say "done"
