#!/usr/bin/env bash
# Run the document REPL agent on BrowseComp-Plus (dense server :8765, or ColBERT :8766 for capcolbert).
#   runners/run_fulldoc_rerun.sh smoke | full | k5 | k5print | k5prompt | nogrep [WORKERS] | cap N [WORKERS] | capcolbert N [WORKERS]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/agents/document_repl/src"
RETRY_HOOK="$ROOT/agents/snippet_repl/src/hooks/retry429"
export FULLDOC_ROOT="$ROOT/data/x3/fulldoc_root"       # symlinks to data/prepared, data/indexes/qwen3-embedding-8b, data/models and rlm_upstream
export RUNS_DIR="$ROOT/data/x3/fulldoc_runs"
export RLMLIB_PATH=/path/to/rlm/RLM/rlm_upstream   # rlm 854e688 plus a 5-line patch
export PYTHON="$ROOT/.venv-rlm/bin/python"
# retry429 hook: openai clients keep retrying Azure 429 responses.
export PYTHONPATH="$RETRY_HOOK:$RLMLIB_PATH:$ROOT/agents/shared${PYTHONPATH:+:$PYTHONPATH}" OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-12}"
export AZURE_DEPLOYMENT_GPT5=gpt-5 AZURE_DEPLOYMENT_JUDGE=gpt-4.1
# Credentials from .secrets/x3.env, exported under the names the harness reads.
set -a; . "$ROOT/.secrets/x3.env"; set +a
export AZURE2_ENDPOINT="${AZURE_OPENAI_ENDPOINT%/}/" AZURE2_API_KEY="$AZURE_OPENAI_API_KEY" AZURE2_API_VERSION="${AZURE_OPENAI_API_VERSION:-2025-04-01-preview}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNS_DIR" "$ROOT/data/x3/logs"
FD_PORT="${FD_SERVER_PORT:-8765}"   # 8765 dense, 8766 Reason-ModernColBERT (runners/colbert_server.sh)
curl -s -m 5 "http://127.0.0.1:$FD_PORT/health" | grep -q . || { echo "Full-Doc retrieval server on :$FD_PORT is not answering" >&2; exit 1; }
NAME="${NAME:-fd_dense_high150}"
CAP="${CAP:-10}"
COMMON=(--design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 --root-deployment "$AZURE_DEPLOYMENT_GPT5" --sub-deployment "$AZURE_DEPLOYMENT_GPT5"
        --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches "$CAP"
        --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator
        --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory --backend azure_openai --stagger 0.4)
cd "$FULLDOC_ROOT"
case "${1:-smoke}" in
  smoke) exec "$PYTHON" "$SRC/run_experiment.py" --name "${NAME}_smoke" --only 3,8 --workers 2 "${COMMON[@]}" ;;
  full)  exec "$PYTHON" "$SRC/run_experiment.py" --name "$NAME" --ids-file "$ROOT/data/x3/subset_150_ids.json" --workers "${2:-12}" "${COMMON[@]}" ;;
  k5)    exec "$PYTHON" "$SRC/run_experiment.py" --name "fd_dense_high150_k5${FD_RUN_SUFFIX:-}" --ids-file "$ROOT/data/x3/subset_150_ids.json" --workers "${2:-24}" \
              --design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 --root-deployment "$AZURE_DEPLOYMENT_GPT5" --sub-deployment "$AZURE_DEPLOYMENT_GPT5" \
              --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches 10 --fixed-k --default-k 5 \
              --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator \
              --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory --backend azure_openai --stagger 0.4 ;;
  cap)   N="${2:?cap value}"
         exec "$PYTHON" "$SRC/run_experiment.py" --name "fd_dense_high50_cap$(printf '%02d' "$N")" --ids-file "${CAP_IDS:-$ROOT/data/x3/fd_cap_ids50.json}" --workers "${3:-10}" \
              --design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 --root-deployment "$AZURE_DEPLOYMENT_GPT5" --sub-deployment "$AZURE_DEPLOYMENT_GPT5" \
              --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches "$N" \
              --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator \
              --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory --backend azure_openai --stagger 0.4 ;;
  capcolbert) N="${2:?cap value}"   # all 830 questions; FD_RUN_SUFFIX / CAP_IDS / FD_SERVER_URL split the load over servers
         exec "$PYTHON" "$SRC/run_experiment.py" --name "fd_colbert8192_high_cap$(printf '%02d' "$N")${FD_RUN_SUFFIX:-}" --ids-file "${CAP_IDS:-$ROOT/data/x3/all_830_ids.json}" --workers "${3:-32}" \
              --design B --server "${FD_SERVER_URL:-http://127.0.0.1:8766}" --root-model gpt-5 --sub-model gpt-5 --root-deployment "$AZURE_DEPLOYMENT_GPT5" --sub-deployment "$AZURE_DEPLOYMENT_GPT5" \
              --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches "$N" \
              --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator \
              --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory --backend azure_openai --stagger 0.4 ;;
  k5print) # k5 with every new document also printed into the prompt
         export PRINT_DOCS_CHAIN_HOOK="$RETRY_HOOK/sitecustomize.py"
         export PYTHONPATH="$ROOT/agents/hooks/document_repl/print_docs:$PYTHONPATH"
         exec "$PYTHON" "$SRC/run_experiment.py" --name "fd_dense_high150_k5print${FD_RUN_SUFFIX:-}" --ids-file "${CAP_IDS:-$ROOT/data/x3/subset_150_ids.json}" --workers "${2:-16}" \
              --design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 --root-deployment "$AZURE_DEPLOYMENT_GPT5" --sub-deployment "$AZURE_DEPLOYMENT_GPT5" \
              --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches 10 --fixed-k --default-k 5 \
              --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator \
              --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory --backend azure_openai --stagger 0.4 ;;
  k5prompt) # k5 with documents printed into the prompt and not kept in the REPL; NOSTATE_ONLY=ids for a smoke test
         export NOSTATE_CHAIN_HOOK="$RETRY_HOOK/sitecustomize.py"
         export PYTHONPATH="$ROOT/agents/hooks/document_repl/print_nostate:$PYTHONPATH"
         SEL=(--ids-file "$ROOT/data/x3/subset_150_ids.json"); [ -n "${NOSTATE_ONLY:-}" ] && SEL=(--only "$NOSTATE_ONLY")
         exec "$PYTHON" "$SRC/run_experiment.py" --name "fd_dense_high150_k5prompt${FD_RUN_SUFFIX:-}" "${SEL[@]}" --workers "${2:-16}" \
              --design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 --root-deployment "$AZURE_DEPLOYMENT_GPT5" --sub-deployment "$AZURE_DEPLOYMENT_GPT5" \
              --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches 10 --fixed-k --default-k 5 \
              --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator \
              --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory --backend azure_openai --stagger 0.4 ;;
  nogrep) # cap 10 without the grep_docs tool; NOGREP_ONLY=ids for a smoke test
         export NO_GREP_CHAIN_HOOK="$RETRY_HOOK/sitecustomize.py"
         export PYTHONPATH="$ROOT/agents/hooks/document_repl/no_grep:$PYTHONPATH"
         SEL=(--ids-file "$ROOT/data/x3/subset_150_ids.json"); [ -n "${NOGREP_ONLY:-}" ] && SEL=(--only "$NOGREP_ONLY")
         exec "$PYTHON" "$SRC/run_experiment.py" --name "fd_dense_high150_nogrep${FD_RUN_SUFFIX:-}" "${SEL[@]}" --workers "${2:-16}" "${COMMON[@]}" ;;
  *) echo "usage: $0 smoke|full [workers]|k5 [workers]|k5print [workers]|nogrep [workers]|cap N [workers]|capcolbert N [workers]" >&2; exit 2 ;;
esac
