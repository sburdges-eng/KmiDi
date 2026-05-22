# JEPA — KmiDi

Workspace entry point for the JEPA (Joint-Embedding Predictive Architecture)
models used in KmiDi.

## Canonical source

All JEPA code lives in [`music_brain/jepa/`](../../music_brain/jepa/):

| Module | Purpose |
|--------|---------|
| `audio_jepa.py` | `AudioJEPAEncoder` (conv mel→latent), `LatentPredictor` (Transformer), `EMATargetEncoder` (teacher) |
| `chord_jepa.py` | `ChordJEPA` (Transformer, 170 chord classes), `ChordEmbedding` |
| `config.py` | `AudioJEPAConfig`, `ChordJEPAConfig`, `StemJEPAConfig`, `TrainingConfig` |
| `datasets.py` | `AudioMelDataset`, `ChordSequenceDataset` |
| `emotion_probe.py` | `EmotionProbe` (valence/arousal auxiliary MLP) |
| `masking.py` | `MultiBlockMasking`, `mask_latents` |
| `trainer.py` | `train_audio_jepa`, `train_chord_jepa` (full loops with AMP, EMA, checkpoints, ONNX export) |

## Usage

```python
from music_brain.jepa import AudioJEPAEncoder, ChordJEPA
from music_brain.jepa.config import AudioJEPAConfig, TrainingConfig
from music_brain.jepa.trainer import train_audio_jepa
```

## Related

- Experiment configs: `experiments/exp_001_ump_jepa/`, `exp_002_wavjepa_emotion/`, `exp_003_jepa_transcriber_probe/`
- Checkpoints: `checkpoints/audio_jepa/`, `checkpoints/chord_jepa/`
- Core ML export: `models/audio_jepa_v01.mlpackage`
- Training entrypoints: `training/scripts/train_jepa.py`, `scripts/train_jepa_local.py`
- SageMaker: `scripts/sagemaker_train.py`, `docs/SAGEMAKER_SETUP.md`
