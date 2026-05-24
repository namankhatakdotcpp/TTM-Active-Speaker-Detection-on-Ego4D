#!/usr/bin/env bash
# Launch I3D-R50 TTM training in the background (SSH-safe).
#
# Usage:
#   chmod +x train_run.sh
#   ./train_run.sh
#
# Monitor:
#   tail -f logs/training.log

set -euo pipefail

PYTHON="/usershome/cs671_user13/miniconda3/envs/ttm/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/logs/training.log"

mkdir -p "$SCRIPT_DIR/logs"

echo "============================================" | tee -a "$LOG"
echo "Training start: $(date)"                      | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

nohup "$PYTHON" "$SCRIPT_DIR/train.py" >> "$LOG" 2>&1 &
PID=$!

echo "Training launched (PID=$PID)" | tee -a "$LOG"
echo "Monitor with: tail -f $LOG"
echo ""
echo "To stop:  kill $PID"
