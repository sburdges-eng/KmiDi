#!/usr/bin/env bash
# One-time path check for Next 90 days / DATA LAW.
# Verifies ~/Datasets and ~/Models (and ~/Models/checkpoints); does not create them.
# See docs/DATA_AND_TRAINING.md.
set -e
OK=0
for dir in "$HOME/Datasets" "$HOME/Models" "$HOME/Models/checkpoints"; do
  if [ -d "$dir" ]; then
    echo "[OK] $dir"
    OK=$((OK+1))
  else
    echo "[MISSING] $dir (create if using training: mkdir -p \"$dir\")"
  fi
done
echo "--- $OK/3 expected dirs present (optional unless training)."
