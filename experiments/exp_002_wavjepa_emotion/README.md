# WavJEPA emotional separability experiment

Test whether **frozen** WavJEPA embeddings separate emotion (e.g. discrete labels from CREMA-D / RAVDESS) on speech/music emotion data. If they don’t, treat WavJEPA as a compressor, not a semantic goldmine.

## Datasets

- **CREMA-D:** ~7.4k WAVs, emotion labels from filenames.
- **RAVDESS:** ~1.4k files, emotion (and other metadata) from filenames.

**Data (in place):** By default the loader uses this experiment’s **`data/`** folder. Put or symlink WAVs there:
- `data/emotions/ravdess/` — RAVDESS WAVs  
- `data/emotions/cremad/` or `data/raw/emotions/cremad/` — CREMA-D WAVs (loader checks both).

Override: set `KMIDI_DATASETS_PATH` or `AUDIO_DATA_ROOT` to use another root. **CREMA-D note:** The GitHub zip has LFS stubs only. For real WAVs, clone the repo and run `git lfs pull`, then `bash scripts/finish_cremad_from_clone.sh` (see that script for full steps). See [wavjepa_emotion_protocol.md](../../docs/wavjepa_emotion_protocol.md) and `scripts/utilities/prepare_datasets.py` for path and filename parsing.

## Metrics

- **Clustering:** k-means with k = number of emotion classes; NMI, adjusted Rand index (optional: accuracy by assigning cluster id to majority class).
- **Optional:** One shallow head (linear or 1-layer MLP) embedding → emotion label; report accuracy and per-class metrics.

## Encoders (Option B: baselines)

Same experiment supports multiple **frozen** encoders for comparison:

- `wavjepa` — e.g. `labhamlet/wavjepa-base`
- `hubert_base` — e.g. `facebook/hubert-base-ls960`
- `wav2vec2_base` — e.g. `facebook/wav2vec2-base`

Set `encoder` in `config.yaml` (or CLI). Same dataset, split, and metrics for all.

## Protocol and results

- **Protocol:** [docs/wavjepa_emotion_protocol.md](../../docs/wavjepa_emotion_protocol.md)
- **Baseline comparison results:** [docs/wavjepa_baselines_results.md](../../docs/wavjepa_baselines_results.md)

## Dependencies

See `requirements.txt`. Core: `transformers`, `torch`, `scikit-learn`, `soundfile` or `librosa` (resample to 16 kHz). WavJEPA: `labhamlet/wavjepa-base` on Hugging Face or official WavJEPA repo; preprocessing (resample, RMS/instance norm) per their README.
