#!/usr/bin/env bash
# Run the snippet REPL agent on the 150-question subset against the dense server on :8001.
#   runners/run_snippet_rerun.sh smoke | full [WORKERS]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/agents/snippet_repl/src"
RLMLIB_PATH="${RLMLIB_PATH:-/path/to/rlm-854e688}"   # upstream rlm at 854e688; the harness patches it at import
PYTHON="$ROOT/.venv-rlm/bin/python"
RUNS="${SNIP_RUNS:-$ROOT/data/x3/snippet_runs}"; mkdir -p "$RUNS/logs" "$ROOT/data/x3/logs"
set -a; . "$ROOT/.secrets/x3.env"; set +a
export AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY" AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT%/}/" AZURE_OPENAI_MODEL=gpt-5
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2025-04-01-preview}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$RLMLIB_PATH:$ROOT/agents/shared"
# Ablations (search cap, fixed top-k) are sitecustomize hooks; the harness is unchanged.
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-12}"
if [ -n "${SNIP_SEARCH_CAP:-}" ]; then export PYTHONPATH="$ROOT/agents/hooks/snippet_repl/search_cap_retry429:$PYTHONPATH"
elif [ -n "${SNIP_FIXED_TOPK:-}" ]; then export PYTHONPATH="$ROOT/agents/hooks/snippet_repl/topk25:$PYTHONPATH"
else export PYTHONPATH="$SRC/hooks/retry429:$PYTHONPATH"; fi
export EMBEDDING_SERVER_URL=http://127.0.0.1:8001
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=high
export LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K="${SNIP_FIXED_TOPK:-5}" ORCHESTRATOR=0
export RLM_TRAJ_DIR="$RUNS/logs/traj"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL
curl -s -m 5 http://127.0.0.1:8001/health | grep -q ok || { echo "dense server on :8001 not healthy" >&2; exit 1; }
# Subset query ids -> dataset positions (q_idx) used by the harness.
mapfile -t IDX < <(env -u PYTHONPATH "$ROOT/.venv/bin/python" - <<'PY'
import json
order = json.load(open("data/analysis/bcp_qidx_to_queryid.json")); pos = {q: i for i, q in enumerate(order)}
sub = json.load(open("data/analysis/x3_subset.json"))["query_ids"]
print("\n".join(str(pos[q]) for q in sub))
PY
)
case "${1:-smoke}" in
  smoke) IDX=("${IDX[@]:0:2}"); W=2 ;;
  full)  W="${2:-12}" ;;
  *) echo "usage: $0 smoke|full [workers]" >&2; exit 2 ;;
esac
cd "$RUNS"
echo "$(date -u '+%H:%M') launching ${#IDX[@]} questions on $W workers -> $RUNS/logs/snip_q<idx>.json" | tee -a logs/launch.log
for w in $(seq 0 $((W-1))); do
(
  i=0
  for q in "${IDX[@]}"; do
    if [ $((i % W)) -eq $w ]; then
      [ -f "logs/snip_q$q.json" ] || "$PYTHON" "$SRC/harness.py" --start "$q" --end $((q+1)) --max-iterations 100 --max-depth 1 --out "logs/snip_q$q" > "logs/snip_q$q.log" 2>&1 || true
    fi
    i=$((i+1))
  done
) &
done
wait
echo "$(date -u '+%H:%M') done: $(ls logs/snip_q*.json 2>/dev/null | wc -l) result files" | tee -a logs/launch.log
