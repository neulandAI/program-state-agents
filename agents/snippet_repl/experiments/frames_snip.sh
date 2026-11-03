#!/usr/bin/env bash
# FRAMES (824 questions), snippet REPL agent over BM25 Wikipedia on :8040: 25-search cap, 50 turns, compaction.
#   experiments/frames_snip.sh <gpt5|qwen> <out_dir> [workers] [first_q] [last_q_exclusive]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MODEL=$1; OUT=$(realpath -m $2); W=${3:-8}; Q0=${4:-0}; Q1=${5:-824}
PY=${PYTHON:?}
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
: "${FRAMES_DATA_DIR:?set FRAMES_DATA_DIR in configs/paths.env (holds frames_rows.json)}"
mkdir -p "$OUT/traj" "${RUNS_DIR:?}"; cd "$RUNS_DIR"
curl -s -m10 http://localhost:8040/health | grep -q '"status":"ok"' || { echo "ABORT: :8040 BM25 down"; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONPATH=$(snip_pythonpath run5d) HF_HOME=${HF_HOME:?}
export EMBEDDING_SERVER_URL=http://localhost:8040 FRAMES_SEARCH_CAP=25 RETRIEVER_LINE=0 FRAMES_BM25_LINE=1
export ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 RLM_TRAJ_DIR=$OUT/traj RLM_USAGE_LOG=$OUT/usage.jsonl
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LOCAL_JUDGE_BASE_URL=http://localhost:8030/v1 LOCAL_JUDGE_MODEL=qwen36-35b-a3b   # in-run judge; final grading is separate
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL AZURE_OPENAI_DEPLOYMENT
if [ "$MODEL" = gpt5 ]; then
  export AZURE_OPENAI_MODEL=${AZURE_DEPLOYMENT_GPT5:?} AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
  unset LOCAL_LLM_BASE_URL LOCAL_LLM_MODEL FRAMES_REPL_CUE FRAMES_FENCE_TOLERANT
elif [ "$MODEL" = qwen ]; then
  export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LM_TIMEOUT=1800 FRAMES_REPL_CUE=1 FRAMES_FENCE_TOLERANT=1
  export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium
else echo "model must be gpt5 or qwen"; exit 2; fi
export OUT PY MODEL SRC
echo "$(date '+%F %T') FRAMES snippet model=$MODEL q=[$Q0,$Q1) workers=$W -> $OUT"
seq $Q0 $((Q1-1)) | xargs -P "$W" -I{} sh -c '
  q=$(printf "%03d" {}); f=$OUT/q$q
  [ -s "$f.json" ] && exit 0
  if [ "$MODEL" = qwen ]; then export LOCAL_LLM_BASE_URL=http://localhost:$((8030 + {} % 2))/v1; fi
  timeout 7200 $PY $SRC/run_frames.py --start {} --end $(({}+1)) --max-iterations 50 --compaction --out "$f" > "$f.log" 2>&1
'
echo "$(date '+%F %T') done: $(ls $OUT/q*.json 2>/dev/null | wc -l) results"
