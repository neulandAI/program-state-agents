#!/usr/bin/env bash
# Fetch-then-Explore (liuqi6777/search_agent, commit 51ceadc, unmodified) on BrowseComp-Plus with GPT-5 at high effort.
#   agents/fetch_then_explore/run_fte_bcp.sh <run_id> [questions: bcp150|bcp830|<n> for the first n of bcp150] [workers] [max_turns]
# Setup (all ours, FtE's code untouched): search via agents/fetch_then_explore/fte_search_adapter.py on :8082 over our exact dense
# retriever (runners/dense_server.py on :8765; Qwen3-Embedding-8B, top 5, 512-token snippets); startup hook
# agents/fetch_then_explore/hooks: egress allowlist (Azure endpoint host + loopback only), GPT-5 parameter shim, deployment pinned,
# served model logged per call. A preflight call must be served by GPT-5 or the run stops. Judging: agents/fetch_then_explore/fte_to_x3.py
# then evaluation/x3_judge.py (gpt-4.1, benchmark template, 3 votes), exactly as every run in the paper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
RUN="${1:?run id}"; Q="${2:-bcp150}"; W="${3:-8}"; TURNS="${4:-128}"
set -a; . "$ROOT/.secrets/x3.env"; set +a
: "${FTE_PYTHON:?set FTE_PYTHON to the python of a venv with Fetch-then-Explore installed (third_party/fte_search_agent)}"
: "${AZURE_OPENAI_ENDPOINT:?set it with evaluation/x3_secret.sh}" "${AZURE_OPENAI_API_KEY:?}" "${GPT5_DEPLOYMENT:?}"
HOST="$(python3 -c "import urllib.parse,sys;print(urllib.parse.urlparse(sys.argv[1]).hostname)" "$AZURE_OPENAI_ENDPOINT")"
OUT="$ROOT/data/x3/fte/runs"; mkdir -p "$OUT" "$ROOT/data/x3/fte/logs"
# questions
DATA="$ROOT/data/x3/fte/bcp150.jsonl"
case "$Q" in bcp150) ;; bcp830) DATA="$ROOT/data/x3/fte/bcp830.jsonl" ;; *) head -n "$Q" "$ROOT/data/x3/fte/bcp150.jsonl" > "$ROOT/data/x3/fte/first$Q.jsonl"; DATA="$ROOT/data/x3/fte/first$Q.jsonl" ;; esac
# retrieval: exact dense server + FtE adapter
curl -s -m5 http://127.0.0.1:8765/health | grep -q ok || { echo "start the dense server first: python runners/dense_server.py --port 8765" >&2; exit 1; }
curl -s -m5 http://127.0.0.1:8082/healthz | grep -q ok || { nohup "${PYTHON:-python}" agents/fetch_then_explore/fte_search_adapter.py --port 8082 > data/x3/fte/logs/adapter.log 2>&1 & for i in $(seq 1 60); do curl -s -m3 http://127.0.0.1:8082/healthz | grep -q ok && break; sleep 2; done; }
export PYTHONPATH="$ROOT/agents/fetch_then_explore/hooks" FTE_ALLOWED_HOSTS="$HOST" FTE_GPT5_SHIM=1 FTE_DEPLOYMENT="$GPT5_DEPLOYMENT" \
       FTE_SERVED_LOG="$ROOT/data/x3/fte/logs/${RUN}_served.jsonl" FTE_EGRESS_LOG="$ROOT/data/x3/fte/logs/${RUN}_egress.jsonl" \
       TIKTOKEN_CACHE_DIR="$ROOT/data/x3/fte/tiktoken_cache" HF_HUB_OFFLINE=1 \
       AGENT_MODEL_NAME="$GPT5_DEPLOYMENT" AGENT_API_BASE="${AZURE_OPENAI_ENDPOINT%/}/openai/v1" AGENT_API_KEY="$AZURE_OPENAI_API_KEY" \
       AGENT_EXTRA_BODY='{"reasoning_effort":"high"}' AGENT_MAX_TOKENS="${FTE_MAX_TOKENS:-32768}" \
       SEARCH_BACKEND=local LOCAL_SEARCH_API_BASE=http://127.0.0.1:8082 LOCAL_SEARCH_TOP_K=5 TOOL_PROFILE=fetch_explore SERPER_API_KEY= \
       LLM_MAX_RETRIES="${FTE_MAX_RETRIES:-20}"
# preflight: one tiny call through the same hook; it must be served by GPT-5
"$FTE_PYTHON" - <<'PY'
import os, sys
from openai import OpenAI
r = OpenAI(base_url=os.environ["AGENT_API_BASE"], api_key=os.environ["AGENT_API_KEY"]).chat.completions.create(
    model="x", messages=[{"role": "user", "content": "Reply with the single word ok"}], max_tokens=512, extra_body={"reasoning_effort": "high"})
m = (r.model or "").lower(); print(f"preflight: deployment {os.environ['FTE_DEPLOYMENT']} served {r.model}")
sys.exit(0 if m == "gpt-5" or m.startswith("gpt-5-20") else f"ABORT: deployment serves {r.model}, not GPT-5")
PY
echo "$(date '+%F %T') FtE run $RUN: $DATA, $W workers, $TURNS turns, egress allowlist: $HOST + loopback"
cd third_party/fte_search_agent
for pass in 1 2; do   # pass 2 re-runs only questions that ended in an error (FtE resumes and skips the rest)
  "$FTE_PYTHON" -m src.run --dataset_path "$DATA" --dataset_name browsecomp_plus --output_dir "$OUT" --run_id "$RUN" \
    --tool_profile fetch_explore --backend local --max_workers "$W" --max_turns "$TURNS" --timeout_minutes 150
done
cd "$ROOT"
# post-processing runs without the startup hook (it only belongs to the agent process)
unset PYTHONPATH FTE_GPT5_SHIM FTE_ALLOWED_HOSTS FTE_DEPLOYMENT
echo "egress blocks: $(cat data/x3/fte/logs/${RUN}_egress.jsonl 2>/dev/null | wc -l)   served models: $(python3 -c "import json,collections,sys;print(dict(collections.Counter(json.loads(l)['model'] for l in open(sys.argv[1]))))" data/x3/fte/logs/${RUN}_served.jsonl)"
"${PYTHON:-python}" agents/fetch_then_explore/fte_to_x3.py "$OUT/browsecomp_plus/$RUN/predictions.jsonl" "fte_$RUN"
"${PYTHON:-python}" evaluation/x3_judge.py --run "fte_$RUN" --no-pairs
