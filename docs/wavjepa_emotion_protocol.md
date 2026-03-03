# WavJEPA emotion experiment — protocol

Protocol for evaluating **emotional separability** of frozen WavJEPA (and baseline) embeddings on CREMA-D / RAVDESS. No fine-tuning of encoders.

## Protocol

- **Datasets:** CREMA-D (~7.4k WAVs), RAVDESS (~1.4k files). Paths via `KMIDI_DATASETS_PATH` or `~/Datasets`. Reuse path and filename parsing from `scripts/utilities/prepare_datasets.py` (e.g. `parse_ravdess_filename`, `parse_crema_emotion`).
- **Preprocessing:** Resample to 16 kHz; 2 s windows (or document mismatch if different). RMS or instance norm as per WavJEPA README. No randomness at inference.
- **Encoders:** Frozen only (e.g. `labhamlet/wavjepa-base`). For baseline comparison, same protocol with HuBERT base, Wav2Vec2 base; same train/val split and metrics.
- **Evaluation:** Same fixed train/val split (fixed seed) for clustering and for optional shallow head.

## Dataset sizes

| Dataset   | ~Files | ~Size  |
|----------|--------|--------|
| CREMA-D  | ~7.4k  | ~6–8 GB |
| RAVDESS  | ~1.4k  | ~2–3 GB |

(Combined ~8–11 GB; exact counts depend on download and any filtering.)

## Data readiness

- **Root:** The experiment uses **one** data root, resolved in this order: `KMIDI_DATASETS_PATH` → `AUDIO_DATA_ROOT` → `~/Datasets`. Set either env var (or both to the same path) so the loader and `scripts/utilities/prepare_datasets.py` agree.
- **Subpaths (match prepare_datasets.py):** Under the root, expect:
  - **RAVDESS:** `emotions/ravdess` (WAVs; RAVDESS is downloaded from Kaggle `uwrfkaggler/ravdess-emotional-speech-audio`).
  - **CREMA-D:** `emotions/cremad` (WAVs; CREMA-D from GitHub zip, unpacked into this dir).
- **Prepare data:** From repo root, with root set (e.g. `export AUDIO_DATA_ROOT=~/Datasets`):
  ```bash
  python scripts/utilities/prepare_datasets.py --dataset emotion_ravdess --download
  python scripts/utilities/prepare_datasets.py --dataset emotion_cremad --download
  ```
  RAVDESS requires Kaggle CLI and credentials. CREMA-D downloads from URL and may need unpacking into `emotions/cremad`.
- **Check:** Under the chosen root, `emotions/ravdess` and `emotions/cremad` should contain `.wav` files; filename parsing is as in `prepare_datasets.py` (`parse_ravdess_filename`, `parse_crema_emotion`).
- **Git LFS:** If the dataset was cloned with Git LFS, the repo may contain only LFS pointer files (text stubs), not real WAVs. The pipeline detects these and errors with a clear message. In the dataset repo (e.g. CREMA-D on external drive), run `git lfs pull` to fetch the actual audio, or use a copy of the data that has real WAVs (e.g. from `prepare_datasets.py --download` or manual unpack).

## Metrics

- **Clustering:** k-means, k = number of emotion classes. Report **NMI**, **adjusted Rand index**. Optional: accuracy by assigning each cluster to majority class.
- **Optional shallow head:** One linear or 1-layer MLP (embedding → emotion label). Report **accuracy** and per-class metrics (e.g. precision/recall/F1 per emotion).

## Conclusion (to fill after run)

- **Separability:** [ Sufficient for use case / Weak — treat as compressor only / TBD after first run ]
- **Compressor vs semantic:** If clustering and shallow-head accuracy are low, treat WavJEPA as a good acoustic compressor rather than a semantic goldmine for emotion; consider spectrogram baselines (HuBERT/Wav2Vec2) per [wavjepa_baselines_results.md](wavjepa_baselines_results.md).
