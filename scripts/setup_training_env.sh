#!/usr/bin/env bash
# Create local learning dirs for JEPA + current training (DATA LAW).
# Does NOT create conda/venv inside repo; env lives in ~/envs (ENV LAW).
# See docs/TRAINING_ENV.md and docs/DATA_AND_TRAINING.md.
set -e
HOME="${HOME:-$HOME}"
DS="$HOME/Datasets"
MD="$HOME/Models"
mkdir -p "$DS/kmidi_jepa/manifests" "$DS/kmidi_learning" "$MD/checkpoints"
echo "[OK] $DS/kmidi_jepa/manifests"
echo "[OK] $DS/kmidi_learning"
echo "[OK] $MD/checkpoints"
echo "--- Local learning dirs ready. Create env outside repo: micromamba create -n kmidi-training python=3.11 && micromamba activate kmidi-training"
