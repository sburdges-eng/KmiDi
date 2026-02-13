#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIDI_MANIFEST="$ROOT_DIR/ml/data/manifests/active/midi-approved.txt"
MODEL_MANIFEST="$ROOT_DIR/ml/data/manifests/active/models-approved.txt"

"$ROOT_DIR/scripts/check_training_gate.sh"
"$ROOT_DIR/scripts/build_training_manifests.sh"

echo "JEPA training entrypoint not implemented yet."
echo "Approved manifests refreshed and ready:"
echo "- $MIDI_MANIFEST"
echo "- $MODEL_MANIFEST"
