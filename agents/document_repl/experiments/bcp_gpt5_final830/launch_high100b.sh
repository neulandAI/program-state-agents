#!/usr/bin/env bash
# Batch 2 of 4 of the GPT-5 high-effort Reason-ModernColBERT run (100 more of the 830, seed 100100). Flags as in
# launch_high400.sh.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_azure.sh"
cd "$FULLDOC_ROOT" || exit 1
exec "$PYTHON" "$SRC/run_experiment.py" \
  --name colbert8192_high100b \
  --design B \
  --ids-file data/prepared/colbert8192_high100b.json \
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
