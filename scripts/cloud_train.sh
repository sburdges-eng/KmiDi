#!/usr/bin/env bash
# Launch cloud training — tmux session, Docker, auto-resume.
# Run ON the cloud instance after cloud_setup.sh.
# Reference: docs/CLOUD_TRAINING.md
#
# Usage: ./scripts/cloud_train.sh [CONFIG]
#   CONFIG defaults to /workspace/config.yaml

set -e

WORKSPACE="/workspace"
REPO="$WORKSPACE/repos/kmidi"
CONFIG="${1:-$WORKSPACE/config.yaml}"
SESSION="kmidi-train"

# Verify GPU inside container
verify_gpu() {
  docker run --rm --gpus all kmidi-trainer python -c "
import torch
ok = torch.cuda.is_available()
print('CUDA available:', ok)
if ok:
  print('GPU:', torch.cuda.get_device_name(0))
  print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
exit(0 if ok else 1)
"
}

cd "$REPO"

echo "[1/3] Verifying GPU in container..."
if ! verify_gpu; then
  echo "ERROR: GPU not available in container. Check nvidia-docker, drivers."
  exit 1
fi

echo "[2/3] Launching in tmux (session: $SESSION)..."
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "  Session exists. Attach with: tmux attach -t $SESSION"
  echo "  Or kill and restart: tmux kill-session -t $SESSION && $0 $*"
  exit 0
fi

# Run training inside tmux; use Docker with mount
tmux new-session -d -s "$SESSION" -c "$REPO" -- bash -c "
echo 'Training started at \$(date)'
echo 'Logs: $WORKSPACE/logs/'
mkdir -p $WORKSPACE/logs
docker run --rm --gpus all \
  -v $WORKSPACE/datasets:/workspace/datasets:ro \
  -v $WORKSPACE/checkpoints:/workspace/checkpoints \
  -v $WORKSPACE/logs:/workspace/logs \
  -v $REPO:/workspace/repos/kmidi:ro \
  -e PYTHONPATH=/workspace/repos/kmidi:/workspace/repos/kmidi/KmiDi_CANON/brain \
  -e WORKSPACE=$WORKSPACE \
  -w /workspace/repos/kmidi \
  kmidi-trainer \
  python -m KmiDi_CANON.training.train_jepa \
  --config $CONFIG \
  --output_dir $WORKSPACE/checkpoints \
  2>&1 | tee $WORKSPACE/logs/train_\$(date +%Y%m%d_%H%M%S).log
echo 'Training finished at \$(date)'
exec bash
"

echo "[3/3] Attach with: tmux attach -t $SESSION"
echo "  Detach: Ctrl+B then D"
echo "  Checkpoint path: $WORKSPACE/checkpoints"
