#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner:
# 1) Generate predictions from unified manifest (baseline or TorchScript model).
# 2) Evaluate against ground-truth CSV manifest.
#
# Usage:
#   bash scripts/run_emotion_eval.sh [--manifest PATH] [--csv PATH] [--out-dir DIR] [--pred-split SPLIT] [--eval-split SPLIT] [--checkpoint PATH]
#
# Defaults:
#   manifest:   datasets/validation/emotion_manifest.json
#   csv:        datasets/validation/audio_emotion_manifest.csv
#   out-dir:    output/audio_emotion_eval
#   pred-split: val          (split filter for unified manifest)
#   eval-split: val_gold     (split filter for CSV manifest)

MANIFEST="datasets/validation/emotion_manifest.json"
CSV="datasets/validation/audio_emotion_manifest.csv"
OUT_DIR="output/audio_emotion_eval"
PRED_SPLIT="val"
EVAL_SPLIT="val_gold"
CHECKPOINT=""

usage() {
  echo "Usage: $0 [--manifest PATH] [--csv PATH] [--out-dir DIR] [--pred-split SPLIT] [--eval-split SPLIT] [--checkpoint PATH]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --pred-split) PRED_SPLIT="$2"; shift 2 ;;
    --eval-split) EVAL_SPLIT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

PRED_PATH="${OUT_DIR}/predictions.jsonl"

echo "Generating predictions -> ${PRED_PATH}"
if [[ -n "${CHECKPOINT}" ]]; then
  python scripts/infer_emotion_from_manifest.py \
    --manifest "${MANIFEST}" \
    --split "${PRED_SPLIT}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${PRED_PATH}"
else
  python scripts/infer_emotion_from_manifest.py \
    --manifest "${MANIFEST}" \
    --split "${PRED_SPLIT}" \
    --output "${PRED_PATH}"
fi

echo "Evaluating predictions using ${CSV} (split=${EVAL_SPLIT})"
python scripts/eval_audio_emotion.py \
  --manifest "${CSV}" \
  --predictions "${PRED_PATH}" \
  --split "${EVAL_SPLIT}" \
  --output-dir "${OUT_DIR}"

echo "Done. Metrics at ${OUT_DIR}/metrics.json"
