#!/usr/bin/env bash
# Run the full audio pipeline: extract spectrograms → train ResNet-18.
# SSH-safe via nohup.
#
# Usage:
#   chmod +x run_audio_pipeline.sh
#   ./run_audio_pipeline.sh          # full run (all clips)
#   ./run_audio_pipeline.sh val      # val only — smoke test
#
# Monitor:
#   tail -f logs/audio_pipeline.log

set -euo pipefail

SPLIT="${1:-all}"
WORKERS=32
PYTHON="/usershome/cs671_user13/miniconda3/envs/ttm/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/logs/audio_pipeline.log"

mkdir -p "$SCRIPT_DIR/logs"

echo "==========================================" | tee -a "$LOG"
echo "Audio pipeline start: $(date)"              | tee -a "$LOG"
echo "Split: $SPLIT  |  Workers: $WORKERS"        | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

cd "$SCRIPT_DIR"

# ── Step A1: Extract Mel spectrograms ─────────────────────────────────────────
echo ""                                                   | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step A1: Extracting spectrograms..." | tee -a "$LOG"
"$PYTHON" step_a1_extract_audio.py --split "$SPLIT" --workers "$WORKERS" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step A1 done."                 | tee -a "$LOG"

# ── Step A2: Train ResNet-18 audio encoder ────────────────────────────────────
echo ""                                                   | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step A2: Training audio model..." | tee -a "$LOG"
"$PYTHON" train_audio.py 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step A2 done."                 | tee -a "$LOG"

echo ""                                                   | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "Audio pipeline complete: $(date)"           | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
