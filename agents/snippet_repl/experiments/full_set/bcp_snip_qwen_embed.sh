#!/bin/bash
# BrowseComp-Plus, snippet REPL agent, GPT-5 on Qwen3-Embedding-8B (:8001), all 830 questions: 100 turns, depth 1,
# no compaction, no startup hook. Resumable: existing result files are skipped.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
mkdir -p "${RUNS_DIR:?}/logs"; cd "$RUNS_DIR"
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/rlmoffqwenfull.log; }
curl -s -m10 http://localhost:8001/health | grep -q ok || { say "ABORT: 8001 not healthy"; exit 1; }
say "8001 (Qwen3-Embedding-8B) verified"
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_MODEL=${AZURE_DEPLOYMENT_GPT5:?}
export PYTHONUNBUFFERED=1 PYTHONPATH=$(snip_pythonpath "")
export EMBEDDING_SERVER_URL=http://localhost:8001
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export ORCHESTRATOR=0
export RLM_TRAJ_DIR=logs/rlm_traj32
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL
W=${W:-8}
for w in $(seq 0 $((W-1))); do
(
  for q in $(seq $w $W 829); do
    [ -f logs/rlmfix32_q$q.json ] && continue
    "$PYTHON" "$SRC/harness.py" --start $q --end $((q+1)) \
      --max-iterations 100 --max-depth 1 --out logs/rlmfix32_q$q > logs/rlmfix32_q$q.log 2>&1
  done
) &
done
say "launched q0-829 (FULL dataset) on Qwen3-Embedding-8B across $W workers"
