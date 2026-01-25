#!/usr/bin/env bash
# Run MIDI generator training with WASABI dataset.
# Usage:
#   ./scripts/run_wasabi_training.sh
#   ./scripts/run_wasabi_training.sh --epochs 2   # quick smoke test
#   ./scripts/run_wasabi_training.sh --epochs 15  # full run

set -e
cd "$(dirname "$0")/.."

# Generate WASABI manifest if missing
if [[ ! -f data/wasabi/processed/train.jsonl ]]; then
  echo "Generating WASABI manifest..."
  python3 scripts/generate_wasabi_manifest.py --samples 8000
fi

python3 -u training/cuda_session/train_midi_generator.py \
  --config midi_generator_training_config.yaml \
  --dataset wasabi \
  "$@"
