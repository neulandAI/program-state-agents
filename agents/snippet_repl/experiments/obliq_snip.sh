#!/usr/bin/env bash
# OBLIQ-Bench, snippet REPL agent, GPT-5.2 on Gemini-Embedding-2 (math :8014, twitter :8021, congress :8022):
# top-k 20, 12 turns, relative score advice; task and strategy files from ../configs/obliq/<task>/.
#   experiments/obliq_snip.sh snip <math|twitter|congress> <ids.json> <out_dir> [workers]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARM=$1; TASK=$2; IDS=$(realpath $3); OUT=$(realpath -m $4); W=${5:-8}
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
DEP=${AZURE_DEPLOYMENT_GPT52:?}
P=$ALG/configs/obliq
case $TASK in
  math)     OT=analogues;     OS=math;     IDX=obliq-ana-math;     SNIP_URL=http://localhost:8014
            SNIP_FILES="OBLIQ_TASK_FILE=$P/math/task_snippet.txt OBLIQ_STRATEGY_FILE=$P/math/strategy_snippet.txt" ;;
  twitter)  OT=descriptive;   OS=twitter;  IDX=obliq-des-twitter;  SNIP_URL=http://localhost:8021
            SNIP_FILES="" ;;
  congress) OT=tip-of-tongue; OS=congress; IDX=obliq-tot-congress; SNIP_URL=http://localhost:8022
            SNIP_FILES="OBLIQ_TASK_FILE=$P/congress/task_snippet.txt OBLIQ_STRATEGY_FILE=$P/congress/strategy_snippet.txt" ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
mkdir -p "$OUT"
echo "$(date '+%F %T') arm=$ARM task=$TASK ids=$(basename $IDS) workers=$W out=$OUT"
if [ "$ARM" = snip ]; then
  env -u AZURE_OPENAI_DEPLOYMENT -u OBLIQ_FENCE_TOLERANT -u OBLIQ_PRESENCE_PENALTY $SNIP_FILES \
    AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview \
    AZURE_OPENAI_MODEL=$DEP LOCAL_LLM_BASE_URL="" EXTRA_PYTHONPATH=$SRC/hooks/run5d \
    ROOT_EFFORT=high SUB_EFFORT=medium OBLIQ_RELATIVE_SCORES=1 RLM_USAGE_LOG=$OUT/usage.jsonl \
    OBLIQ_TASK=$OT OBLIQ_SUBSET=$OS OBLIQ_INDEX=$IDX EMBEDDING_SERVER_URL=$SNIP_URL \
    OBLIQ_REPL_CUE=1 OBLIQ_TAG_SEARCH=1 RLM_TRAJ_DIR=$OUT/traj \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$ALG/experiments/obliq_shard.sh" "$IDS" "$OUT/all" 20 12 $W
else
  echo "unknown arm $ARM (the fd arm is in agents/document_repl)"; exit 2
fi
echo "$(date '+%F %T') done"
