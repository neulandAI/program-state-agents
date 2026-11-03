#!/usr/bin/env bash
# BrowseComp-Plus, document REPL agent with GPT-5 at high effort on Reason-ModernColBERT (8,192-token index): batch 1
# of 4 (a seeded random 400 of the 830). --expect-served-model needs a checker script that is not part of this
# repository, so run_experiment.py aborts at that gate unless the flag is removed.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_azure.sh"
cd "$FULLDOC_ROOT" || exit 1
WAIT_PID=${WAIT_PID:-}
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') predecessor $WAIT_PID exited; starting high400"
exec "$PYTHON" "$SRC/run_experiment.py" \
  --name colbert8192_high400 \
  --design B \
  --ids-file data/prepared/colbert8192_high400.json \
  --workers 12 --stagger 0.4 \
  --server http://127.0.0.1:8766 \
  --root-model gpt-5 --sub-model gpt-5 \
  --root-deployment "$DEP" --sub-deployment "$DEP" \
  --expect-served-model gpt-5-2025-08-07 \
  --reasoning-effort high --sub-reasoning-effort high \
  --max-iterations 30 --max-searches 10 \
  --seed-context --compaction --grounding-gate --lean-prologue \
  --no-delegation --solo-prompt --no-orchestrator \
  --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 \
  --max-timeout 5400 --save-trajectory
