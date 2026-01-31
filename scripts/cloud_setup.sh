#!/usr/bin/env bash
# One-time cloud GPU setup — storage, clone, docker build.
# Run ON the cloud instance (Lambda Labs, RunPod, etc.).
# Reference: docs/CLOUD_TRAINING.md
#
# Usage: ./scripts/cloud_setup.sh [REPO_URL]
#   REPO_URL defaults to your canonical repo.

set -e

REPO_URL="${1:-https://github.com/sburdges-eng/KmiDi}"
WORKSPACE="/workspace"

echo "[1/5] Verifying workspace (persistent volume)..."
if [ ! -d "$WORKSPACE" ]; then
  echo "ERROR: /workspace not found. Mount persistent volume to /workspace."
  exit 1
fi
if mount | grep -q "on / type"; then
  ROOT_DISK=$(df / | tail -1 | awk '{print $1}')
  WORK_DISK=$(df "$WORKSPACE" 2>/dev/null | tail -1 | awk '{print $1}' || echo "none")
  if [ "$ROOT_DISK" = "$WORK_DISK" ]; then
    echo "WARNING: /workspace may be on ephemeral root disk. Prefer mounted volume."
  fi
fi

echo "[2/5] Creating directory layout..."
mkdir -p "$WORKSPACE"/{datasets/manifests,datasets/shards,checkpoints,logs,repos}

echo "[3/5] Cloning repo..."
if [ -d "$WORKSPACE/repos/kmidi/.git" ]; then
  echo "  Repo exists, pulling..."
  (cd "$WORKSPACE/repos/kmidi" && git pull)
else
  git clone "$REPO_URL" "$WORKSPACE/repos/kmidi"
fi

echo "[4/5] Building Docker image..."
cd "$WORKSPACE/repos/kmidi"
docker build -f docker/Dockerfile.kmidi-trainer -t kmidi-trainer .

echo "[5/5] Copying default cloud config..."
cp -n configs/cloud_training.yaml "$WORKSPACE/config.yaml" 2>/dev/null || true

echo ""
echo "Setup complete. Next:"
echo "  ./scripts/cloud_train.sh          # Launch training in tmux"
echo "  nvidia-smi                        # Verify GPU"
