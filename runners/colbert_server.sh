#!/usr/bin/env bash
# Start, query or stop the Reason-ModernColBERT (document length 8192) retrieval server for the document REPL agent.
#   runners/colbert_server.sh start [GPU] | status | stop
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/path/to/rlm/venvs/colbert/bin/python
export FULLDOC_ROOT=/path/to/rlm
export PYTHONPATH="$ROOT/agents/shared${PYTHONPATH:+:$PYTHONPATH}"
export COLBERT_INDEX_NAME="${COLBERT_INDEX_NAME:-Reason-ModernColBERT-8192}" COLBERT_DOC_LEN="${COLBERT_DOC_LEN:-8192}"
# Search batch, n_full_scores and query length bound GPU memory on this index.
export COLBERT_SEARCH_BATCH="${COLBERT_SEARCH_BATCH:-2048}"
export COLBERT_QUERY_LEN="${COLBERT_QUERY_LEN:-256}"
PORT="${COLBERT_PORT:-8766}"
LOG="$ROOT/data/x3/logs/colbert_server_${PORT}.log"
export COLBERT_N_FULL_SCORES="${COLBERT_N_FULL_SCORES:-2048}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
case "${1:-status}" in
  start)
    if curl -s -m 3 "http://127.0.0.1:$PORT/stats" | grep -q retriever; then echo "ColBERT server on :$PORT already answering"; exit 0; fi
    CUDA_VISIBLE_DEVICES="${2:-0}" setsid nohup "$PY" "$ROOT/runners/colbert_server_wrapped.py" --port "$PORT" --device cuda --query-length "$COLBERT_QUERY_LEN" > "$LOG" 2>&1 < /dev/null &
    echo "starting ColBERT server :$PORT (index $COLBERT_INDEX_NAME, doc len $COLBERT_DOC_LEN, n_full_scores $COLBERT_N_FULL_SCORES, n_ivf_probe ${COLBERT_N_IVF_PROBE:-8}, search batch $COLBERT_SEARCH_BATCH, query len $COLBERT_QUERY_LEN, GPU ${2:-0}); log $LOG"
    for i in $(seq 1 120); do sleep 5; grep -q "READY" "$LOG" && break; grep -q "Traceback" "$LOG" && { tail -20 "$LOG"; exit 1; }; done
    tail -n 2 "$LOG"; curl -s -m 5 "http://127.0.0.1:$PORT/stats" | cut -c1-120; echo ;;
  status) curl -s -m 3 "http://127.0.0.1:$PORT/stats" | cut -c1-200 || echo "not answering"; echo; pgrep -fa "colbert_server_wrappe[d].py --port $PORT|retrieval_server_colber[t].py --port $PORT" | cut -c1-120 ;;
  stop) for p in $(pgrep -f "colbert_server_wrappe[d].py --port $PORT|retrieval_server_colber[t].py --port $PORT"); do kill "$p"; echo "stopped $p"; done
        for i in $(seq 1 30); do sleep 2; [ -z "$(pgrep -f "colbert_server_wrappe[d].py --port $PORT|retrieval_server_colber[t].py --port $PORT")" ] && { echo "server :$PORT gone"; break; }; done ;;
  *) echo "usage: $0 start [GPU]|status|stop" >&2; exit 2 ;;
esac
