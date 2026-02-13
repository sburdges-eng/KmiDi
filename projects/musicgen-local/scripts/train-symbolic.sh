#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIDI_MANIFEST="$ROOT_DIR/ml/data/manifests/active/midi-approved.txt"
MODEL_MANIFEST="$ROOT_DIR/ml/data/manifests/active/models-approved.txt"

"$ROOT_DIR/scripts/check_training_gate.sh"
"$ROOT_DIR/scripts/build_training_manifests.sh"

python3 "$ROOT_DIR/ml/training/symbolic/train_symbolic_entrypoint.py" \
  --midi-manifest "$MIDI_MANIFEST" \
  --model-manifest "$MODEL_MANIFEST" \
  "$@"
