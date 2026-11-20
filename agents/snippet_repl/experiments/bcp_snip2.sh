#!/usr/bin/env bash
# BrowseComp-Plus, snippet REPL agent; as bcp_snip.sh, and for qwen also the ```repl cue, prose-final submission,
# 8192 output tokens per call, 50 turns and a 1 h limit per question (MAX_ITERS_QWEN, QTIMEOUT_QWEN).
#   experiments/bcp_snip2.sh <gpt5|qwen> <retriever_url> <out_dir> [workers] [first_q] [last_q_exclusive]
#   retrievers (rlm wire): :8001 Qwen3-Embedding-8B | :8044 BM25 | :8024 Reason-ModernColBERT
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MODEL=$1; RETR=$2; OUT=$(realpath -m $3); W=${4:-8}; Q0=${5:-0}; Q1=${6:-830}
MAX_ITERS=${MAX_ITERS:-100}; QTIMEOUT=${QTIMEOUT:-7200}
PYV=${PYTHON:?}
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
mkdir -p "$OUT/traj"
curl -s -m10 $RETR/health | grep -q ok || { echo "ABORT: retriever $RETR down"; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONPATH=$(snip_pythonpath run5d) HF_HOME=${HF_HOME:?}
export EMBEDDING_SERVER_URL=$RETR ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export RLM_USAGE_LOG=$OUT/usage.jsonl RLM_TRAJ_DIR=$OUT/traj AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL AZURE_OPENAI_DEPLOYMENT
COMPACT=--compaction   # history compaction; fires only near the context limit
if [ "$MODEL" = gpt5 ]; then
  export AZURE_OPENAI_MODEL=${AZURE_DEPLOYMENT_GPT5:?} AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
  unset LOCAL_LLM_BASE_URL LOCAL_LLM_MODEL QWEN_PRESENCE_PENALTY QWEN_FENCE_TOLERANT BCP_SEARCH_CAP
elif [ "$MODEL" = qwen ]; then
  export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LLM_API_KEY=EMPTY LOCAL_LM_TIMEOUT=1800
  export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium QWEN_PRESENCE_PENALTY=1.5 QWEN_FENCE_TOLERANT=1
  export QWEN_REPL_CUE=1 QWEN_PROSE_FINAL=1 QWEN_MAX_TOKENS=${QWEN_MAX_TOKENS:-8192}
  MAX_ITERS=${MAX_ITERS_QWEN:-50}; QTIMEOUT=${QTIMEOUT_QWEN:-3600}
  export BCP_SEARCH_CAP=25
  for p in 8030 8031; do curl -s -m5 localhost:$p/v1/models | grep -q qwen36 || { echo "ABORT: :$p model server down"; exit 1; }; done
else
  echo "model must be gpt5 or qwen"; exit 2
fi
export OUT PYV MODEL COMPACT MAX_ITERS QTIMEOUT SRC
echo "$(date '+%F %T') model=$MODEL retriever=$RETR q=[$Q0,$Q1) workers=$W iters=$MAX_ITERS timeout=${QTIMEOUT}s max_tokens=${QWEN_MAX_TOKENS:-default} out=$OUT"
seq $Q0 $((Q1-1)) | xargs -P "$W" -I{} sh -c '
  f=$OUT/q{}
  [ -s "$f.json" ] && exit 0
  if [ "$MODEL" = qwen ]; then export LOCAL_LLM_BASE_URL=http://localhost:$((8030 + {} % 2))/v1; fi
  timeout $QTIMEOUT $PYV -u $SRC/run_browsecomp.py --start {} --end $(({}+1)) \
    --max-iterations $MAX_ITERS --max-depth 1 $COMPACT --out "$f" > "$f.log" 2>&1
'
echo "$(date '+%F %T') done: $(ls $OUT/q*.json 2>/dev/null | wc -l) result files"
