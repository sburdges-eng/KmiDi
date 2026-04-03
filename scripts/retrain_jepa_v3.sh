#!/bin/bash
# Aggressive JEPA retrain — run after v2 completes
# Resumes from v2 best checkpoint with tuned hyperparams

set -e
cd /Users/seanburdges/Dev/KmiDi
source .venv-coreml/bin/activate

echo "=== JEPA Retrain v3 — Aggressive ==="
echo "  LR: 1e-3 (3x higher)"
echo "  Emotion weight: 0.2 (2x higher)"
echo "  Batch: 24"
echo "  Epochs: 100"
echo "  Patience: 20"
echo ""

python3 scripts/retrain_jepa.py \
  --resume checkpoints/audio_jepa_v2/best_model.pt \
  --output-dir checkpoints/audio_jepa_v3 \
  --epochs 100 \
  --batch-size 24 \
  --lr 1e-3 \
  --emotion-weight 0.2 \
  --patience 20 \
  --save-every 5

echo ""
echo "=== Done. Next steps: ==="
echo "  cp checkpoints/audio_jepa_v3/best_model.pt checkpoints/audio_jepa/best_model.pt"
echo "  python scripts/export_audio_jepa.py --checkpoint checkpoints/audio_jepa_v3/best_model.pt"
echo "  python scripts/extract_jepa_embeddings.py --checkpoint checkpoints/audio_jepa_v3/best_model.pt"
echo "  python scripts/train_emotion_probe.py"
echo "  python scripts/export_emotion_probe.py"
echo "  python scripts/demo_proof_of_life.py --backend coreml"
