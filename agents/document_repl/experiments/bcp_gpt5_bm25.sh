#!/usr/bin/env bash
# BrowseComp-Plus, document REPL agent with GPT-5 and BM25 (k1 0.9, b 0.4) on :8043, through src/run_bm25.py with
# BM25_CORPUS=bcp. Design B, 30 turns, 10 searches, seeded first search, compaction, grounding gate, solo/lean
# prompt, no delegation, no orchestrator. Startup hook: retry429. No --save-trajectory (whole BM25 pages make
# trajectories very large). Credentials from AZ_ENV (a file with AZ_EP / AZ_KEY).
#   W=16 [IDS_FILE=ids.txt] [OUT=<dir>] experiments/bcp_gpt5_bm25.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"
PY=$PYTHON
SUP=$FULLDOC_ROOT
W=${W:-16}
OUT=$(realpath -m "${OUT:-$RUNS_DIR/bcp_gpt5_bm25/runs}"); mkdir -p "$OUT"
SERVER=${SERVER:-http://127.0.0.1:8043}
DEP=${DEP:-${AZURE_DEPLOYMENT_GPT5:?set AZURE_DEPLOYMENT_GPT5}}
say() { printf '%s  [bcp-gpt5-bm25] %s\n' "$(date '+%F %T')" "$*"; }
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
       AZURE_OPENAI_DEPLOYMENT="$DEP"
curl -s -m10 $SERVER/health | grep -q '"ok"' || { say "ABORT: $SERVER down"; exit 1; }
need_hook retry429
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOOKS_DIR/retry429:$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} BM25_CORPUS=bcp
unset QUERIES_FILE SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY
ENTRY=$SRC/run_bm25.py
EFFORT=${EFFORT:-high}
export EFFORT OUT PY SUP ENTRY SERVER DEP
IDS_FILE=${IDS_FILE:-}
if [ -z "$IDS_FILE" ]; then IDS_FILE=$OUT/../ids830.txt
  $PY -c "import json;[print(json.loads(l)['query_id']) for l in open('$SUP/data/prepared/queries.jsonl')]" > "$IDS_FILE"; fi
say "$(wc -l < $IDS_FILE) questions, $W workers, deployment $DEP, effort $EFFORT, server $SERVER -> $OUT"
cat "$IDS_FILE" | xargs -P "$W" -I{} sh -c '
  f=$OUT/{}.json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 3300 $PY $ENTRY --query-id {} --out "$f" \
    --design B --server $SERVER \
    --root-model gpt-5 --sub-model gpt-5 --root-deployment $DEP --sub-deployment $DEP \
    --reasoning-effort $EFFORT --sub-reasoning-effort $EFFORT --default-k 5 \
    --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 2700 \
    > "${f%.json}.log" 2>&1
'
say "rollouts done: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
