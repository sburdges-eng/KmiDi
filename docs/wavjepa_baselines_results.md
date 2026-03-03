# WavJEPA vs HuBERT / Wav2Vec2 — baseline comparison results

Emotional-separability (and optional music-task) comparison: **frozen** encoders only, same train/val split, same shallow head, same metrics.

## Methods

- **Encoders (frozen):** WavJEPA base (~90M), HuBERT base (~95M), Wav2Vec2 base (~95M). No pretraining or fine-tuning in this experiment.
- **Data:** CREMA-D + RAVDESS; same train/val split (fixed seed) for all encoders.
- **Head:** Same shallow head (linear or 1-layer MLP) for all; embedding → emotion label.
- **Metrics:** NMI, adjusted Rand index, accuracy (and optional per-class). Inference-only; compute reported for same machine where applicable.

## Results (to fill after runs)

| Encoder    | NMI (emotion) | Adj. Rand (emotion) | Accuracy (emotion) | Notes |
|------------|----------------|----------------------|--------------------|-------|
| WavJEPA    | —             | —                    | —                  |       |
| HuBERT     | —             | —                    | —                  |       |
| Wav2Vec2   | —             | —                    | —                  |       |

Optional music tasks (genre / instrument): add columns or a second table after adding data and runs.

## Reproducibility

- **Dataset paths:** Set `KMIDI_DATASETS_PATH` or use `~/Datasets`; see [wavjepa_emotion_protocol.md](wavjepa_emotion_protocol.md).
- **Config:** `experiments/wavjepa_emotion/config.yaml` (encoder, checkpoints, split_ratio, seed).
- **Commands (after A2–B4 implemented):** e.g. `python run.py --encoder wavjepa`, `python run.py --encoder hubert_base`, `python run.py --encoder wav2vec2_base` (or equivalent from config).
