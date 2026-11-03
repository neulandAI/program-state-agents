#!/usr/bin/env bash
# FRAMES, document REPL agent with Qwen3.6 on a local vLLM server, over the FRAMES BM25 server :8041.
# Agent flags = BASE_ARGS of src/split_runner.py plus the Qwen3.6 switches (SUP_FENCE_TOLERANT=1, SUP_MAX_BLOCKS=5,
# SUP_PRESENCE_PENALTY=1.5). Resumable. The paper's run uses BM25_PROMPT=1. FRAMES_GRADE_SUP names the grader run
# afterwards.
#   W=8 START=0 END=100 OUT=<dir> [BM25_PROMPT=1] experiments/frames_qwen.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"
PY=$PYTHON
SUP=$FULLDOC_ROOT
W=${W:-8}; START=${START:-0}; END=${END:-100}
MODEL=${MODEL:-qwen36-35b-a3b}
BASE_URL=${BASE_URL:-http://localhost:8030/v1}   # :8031 = second server, GPU 1
OUT=$(realpath -m "${OUT:-$RUNS_DIR/frames/sup_q36}")   # absolute: run_one.py runs with cwd=FULLDOC_ROOT
mkdir -p "$OUT"
say() { printf '%s  [frames-sup] %s\n' "$(date '+%F %T')" "$*"; }

# SERVER: the retrieval endpoint (default the BM25 server; e.g. :8042 = a rerank proxy in front of it)
SERVER=${SERVER:-http://127.0.0.1:8041}
curl -s -m10 $SERVER/health | grep -q '"ok"' || { say "ABORT: $SERVER retrieval server down"; exit 1; }
curl -s -m10 $BASE_URL/models | grep -q "$MODEL" || { say "ABORT: $BASE_URL does not serve $MODEL"; exit 1; }

export PYTHONUNBUFFERED=1 PYTHONPATH=$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ALIBABA_TOKEN=EMPTY QUERIES_FILE=${FRAMES_DATA_DIR:?set FRAMES_DATA_DIR}/frames_queries.jsonl
case "$MODEL" in qwen36*) export SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 ;;
                 *) unset SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY ;; esac
# BM25_PROMPT=1: run through src/run_bm25.py (retriever description for BM25); otherwise run_one.py.
if [ "${BM25_PROMPT:-0}" = 1 ]; then ENTRY=$SRC/run_bm25.py; else ENTRY=$SRC/run_one.py; fi
# MAX_SEARCHES / MAX_ITERS default to 10 / 30.
MAX_SEARCHES=${MAX_SEARCHES:-10}; MAX_ITERS=${MAX_ITERS:-30}
export OUT MODEL PY SUP ENTRY BASE_URL SERVER MAX_SEARCHES MAX_ITERS

say "questions $START..$((END-1)), $W workers, model $MODEL @ $BASE_URL, entry $ENTRY, server $SERVER, searches $MAX_SEARCHES, iters $MAX_ITERS${IDS_FILE:+, ids $IDS_FILE} -> $OUT"
# REVERSE=1 walks the range backwards, so a second runner on the other GPU can share a range from
# the far end; the only overlap is the question in flight where the two meet.
# IDS_FILE: run exactly these question ids (one per line) instead of START..END
if [ -n "${IDS_FILE:-}" ]; then SEQ="cat $IDS_FILE"
elif [ "${REVERSE:-0}" = 1 ]; then SEQ="seq $((END-1)) -1 $START"; else SEQ="seq $START $((END-1))"; fi
$SEQ | xargs -P "$W" -I{} sh -c '
  f=$OUT/$(printf "%03d" {}).json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 6300 $PY $ENTRY --query-id {} --out "$f" \
    --base-url $BASE_URL \
    --design B --server $SERVER --backend openai --api-key-env ALIBABA_TOKEN \
    --root-model $MODEL --sub-model $MODEL --reasoning-effort medium --sub-reasoning-effort medium \
    --max-iterations $MAX_ITERS --max-searches $MAX_SEARCHES --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 5400 --lm-timeout 1800 --save-trajectory \
    > "${f%.json}.log" 2>&1
'
say "rollouts done: $(ls "$OUT"/[0-9]*.json 2>/dev/null | wc -l)"
$PY "${FRAMES_GRADE_SUP:?set FRAMES_GRADE_SUP to frames_grade_sup.py}" "$OUT"
