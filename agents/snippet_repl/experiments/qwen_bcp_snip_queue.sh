#!/usr/bin/env bash
# BrowseComp-Plus with Qwen3.6 through bcp_snip2.sh on BM25, dense and ColBERT, one after another, 8 workers each.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
D=${RUNS_DIR:?}/run5d/full
mkdir -p "$D"
for r in bm25:http://localhost:8044 dense:http://localhost:8001 colbert:http://localhost:8024; do
  "$ALG/experiments/bcp_snip2.sh" qwen ${r#*:} $D/bcp_snip_qwen_${r%%:*} 8 > $D/bcp_snip_qwen_${r%%:*}.out 2>&1
done
echo "$(date '+%F %T') QWEN SNIPPET QUEUE DONE"
