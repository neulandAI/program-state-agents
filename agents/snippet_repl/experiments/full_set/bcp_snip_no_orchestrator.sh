#!/bin/bash
# BrowseComp-Plus, snippet REPL agent, GPT-5 on Reason-ModernColBERT (:8024) for the question indices listed in
# $NOORCH_QIDS_FILE: 100 turns, depth 1, no compaction, no startup hook.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
mkdir -p "${RUNS_DIR:?}/logs"; cd "$RUNS_DIR"
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/rlmoffnoorch.log; }
curl -s -m10 http://localhost:8024/health | grep -q 'bc_reason_mcb' || { say "ABORT: 8024 wrong index"; exit 1; }
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_MODEL=${AZURE_DEPLOYMENT_GPT5:?}
export PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8024 PYTHONPATH=$(snip_pythonpath "")
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export ORCHESTRATOR=0
export RLM_TRAJ_DIR=logs/rlm_traj24
unset MAX_TOP_K SEARCH_SNIPPET_LEN
Q=($(cat "${NOORCH_QIDS_FILE:?file with the question indices to run (was /tmp/noorch.txt)}"))
W=${W:-5}
for w in $(seq 0 $((W-1))); do
(
  for idx in $(seq $w $W $((${#Q[@]}-1))); do
    q=${Q[$idx]}
    "$PYTHON" "$SRC/harness.py" --start $q --end $((q+1)) \
      --max-iterations 100 --max-depth 1 --out logs/rlmfix24_q$q > logs/rlmfix24_q$q.log 2>&1
  done
) &
done
say "launched ${#Q[@]} run-19 failures with orchestrator=False"
