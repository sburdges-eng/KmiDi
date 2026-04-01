# Training Pipeline Improvements

**Date:** 2026-04-01

## Context

The Audio JEPA encoder produces 512x256 latent embeddings from mel spectrograms, but the latent-to-emotion mapping is a hardcoded 4x256 linear transform in AudioEmotionRunner.cpp that saturates arousal/dominance to 1.0 regardless of input. The JEPA model itself trained for only 3 epochs (loss=0.074) with pure self-supervised MSE loss — no semantic grounding in emotion.

## Goal

1. Train a dedicated emotion probe on frozen JEPA latents that maps to meaningful valence/arousal
2. Improve JEPA model quality with longer training, more data, and an emotion auxiliary loss
3. Export the probe as a separate ONNX model for the C++ plugin

## Phase A: Emotion Probe (independent of JEPA retraining)

### Data

- **Speech emotion**: RAVDESS (~1.4k files, 7 emotions) + CREMA-D (~7.4k files) — already at ~/Datasets/
- **Music emotion**: Download DEAM (2058 excerpts, continuous valence/arousal annotations) and PMEmo (794 songs, continuous VA) to ~/Datasets/
- **Label format**: `{audio_path, valence: float[-1,1], arousal: float[-1,1], split: train|val|test}`
- **Unified dataset**: `music_brain/penta_core/ml/unified_emotion.py` already defines the metadata schema

### Probe architecture

```python
class EmotionProbe(nn.Module):
    # Input: pooled JEPA latent (256,)
    # Output: (valence, arousal) in [-1, 1]
    def __init__(self, latent_dim=256, hidden_dim=128):
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )
```

### Training

- Freeze JEPA encoder — extract latents once, train probe on cached embeddings
- Loss: MSE between predicted (valence, arousal) and ground-truth
- Optimizer: AdamW, lr=1e-3, weight_decay=0.01
- Epochs: 50, early stopping patience 10
- Train/val/test split: 70/15/15

### Export

- Export trained probe to ONNX: input `(1, 256)` → output `(1, 2)`
- Model size: ~100KB (tiny MLP)

### C++ integration

- AudioEmotionRunner loads two ONNX models: JEPA encoder + emotion probe
- After JEPA inference, pool latents (mean over time dim), run probe
- Replace hardcoded `mapLatentToEmotion()` weights with probe inference
- Dominance derived as: `dominance = 0.5 + 0.3 * arousal + 0.2 * abs(valence)`
- Confidence from probe softmax entropy or output magnitude

## Phase B: JEPA Retraining

### Multi-task loss

```python
loss = mse_loss(pred_latent, target_latent)           # JEPA reconstruction
     + 0.1 * mse_loss(probe(encoder_out), target_va)  # Emotion auxiliary
```

The emotion auxiliary steers latent geometry toward emotion-relevant features without overwhelming the self-supervised objective.

### Training improvements

- Train for 50+ epochs (current checkpoint stopped at epoch 3)
- Use mixed precision (AMP) on MPS/CUDA
- Add gradient clipping (already in config at 1.0)
- Larger batch size if memory allows (current: 12)
- Data: all audio from ~/Datasets/ (music + speech), not just the emotion-labeled subset

### After retraining

- Re-export ONNX and Core ML models
- Retrain emotion probe on new latents (latent geometry will have changed)
- Re-run benchmarks to verify latency stays under 8ms

## Files

| File | Action |
|------|--------|
| `music_brain/jepa/emotion_probe.py` | Create — probe model definition |
| `scripts/download_emotion_datasets.py` | Create — fetch DEAM/PMEmo |
| `scripts/train_emotion_probe.py` | Create — extract latents, train probe |
| `scripts/export_emotion_probe.py` | Create — export probe to ONNX |
| `music_brain/jepa/trainer.py` | Modify — add emotion auxiliary loss |
| `config/jepa_training.yaml` | Modify — emotion loss weight, longer training |
| `src/ml/AudioEmotionRunner.cpp` | Modify — load probe ONNX, replace hardcoded mapping |
| `include/penta/ml/AudioEmotionRunner.h` | Modify — add probe model path to config |

## Success criteria

- Probe: emotion classification accuracy > 60% on holdout test set
- Probe: meaningfully different valence/arousal for calm vs energetic audio (verified by demo script)
- JEPA retrained: reconstruction loss < 0.1, emotion cluster separation improved
- AudioEmotionRunner: probe ONNX loads and runs, parameters move meaningfully in demo
- Latency: JEPA + probe inference < 10ms total on Apple Silicon
