# WavJEPA vs KmiDi — File-level task list (C → A → B)

Concrete tasks for the three workstreams from the plan. **Do not execute** until you choose to; this is a checklist only.

**References:** [plan](.cursor/plans/wavjepa_and_kmidi_515ed035.plan.md), [UMP_JEPA_EXPRESSIVE_CONDITIONING.md](UMP_JEPA_EXPRESSIVE_CONDITIONING.md), [apple-silicon-low-latency.md](apple-silicon-low-latency.md), [mt3-transcription-baseline.md](mt3-transcription-baseline.md).

---

## Option C: JEPA-style latent layer in MIDI pipeline (sketch)

**Goal:** One design doc that locks placement, determinism, and “no encoder training.” No code.

### C1. Create design doc

- **File:** `docs/WAVJEPA_LATENT_PIPELINE.md`
- **Contents:**
  - **Placement:** Input (16 kHz, fixed chunk or streaming 2 s) → Frozen WavJEPA context (or target) encoder → one embedding per time step (e.g. 100 Hz) → downstream: latents → optional linear map → token head (MT3-style/REMI) or UMP/JEPA conditioning per [UMP_JEPA_EXPRESSIVE_CONDITIONING.md](UMP_JEPA_EXPRESSIVE_CONDITIONING.md).
  - **Diagram:** One ASCII or Mermaid diagram: `Audio → WavJEPA (frozen) → latents → [optional: linear map] → token head / conditioning`.
  - **Determinism (3–5 bullets):** Frozen encoder + fixed preprocessing (resample, norm; no randomness at inference) ⇒ same audio ⇒ same latent sequence; no EMA or stochastic masking at inference; teacher/target used only as fixed feature extractor; any “contract” (e.g. latent band = intensity) defined *after* the frozen encoder (fixed linear map or frozen adapter).
  - **Where it does *not* go:** Do not train WavJEPA predictor or target in KmiDi stack; use WavJEPA only as frozen feature extractor.
  - **Optional:** Short note on drift detector (embedding sequence vs reference; comparison metric fixed ⇒ deterministic).
- **Cross-links:** Add “See [WAVJEPA_LATENT_PIPELINE.md](WAVJEPA_LATENT_PIPELINE.md)” in [UMP_JEPA_EXPRESSIVE_CONDITIONING.md](UMP_JEPA_EXPRESSIVE_CONDITIONING.md) and [apple-silicon-low-latency.md](apple-silicon-low-latency.md) where the “encode” step / frozen front-end is mentioned.

### C2. (Optional) Add pipeline sketch to UMP_JEPA doc

- **File:** `docs/UMP_JEPA_EXPRESSIVE_CONDITIONING.md`
- **Change:** Add a short subsection or bullet that says “Frozen audio front-end (e.g. WavJEPA) can feed the context encoder; see [WAVJEPA_LATENT_PIPELINE.md](WAVJEPA_LATENT_PIPELINE.md).”

---

## Option A: Emotional separability (WavJEPA only)

**Goal:** One experiment: freeze WavJEPA, extract embeddings on CREMA-D/RAVDESS, measure emotion separability, document protocol and conclusion.

### A1. Experiment directory and deps

- **Dir:** `experiments/exp_002_wavjepa_emotion/`
- **Files to add:**
  - `README.md` — Purpose (emotional separability of WavJEPA embeddings), datasets (CREMA-D, RAVDESS), metrics (NMI, adjusted Rand, optional shallow-head accuracy), and pointer to protocol doc.
  - `requirements.txt` or note in README: `transformers`, `torch`, `scikit-learn`, `soundfile`/`librosa` (resample 16 kHz), and reference to `labhamlet/wavjepa-base` or official WavJEPA repo for loading.

### A2. Data loading and preprocessing

- **File:** `experiments/exp_002_wavjepa_emotion/dataset.py` (or `load_emotion_data.py`)
- **Responsibilities:**
  - Load CREMA-D and/or RAVDESS from paths (env or config: `KMIDI_DATASETS_PATH` or `~/Datasets`; reuse logic from `scripts/utilities/prepare_datasets.py` / `KmiDi_PROJECT/scripts/prepare_datasets.py` for paths and filename parsing — see `parse_ravdess_filename`, `parse_crema_emotion`).
  - Output: (audio_path, emotion_label) per sample; optional (audio_path, valence, arousal) if you add V-A later.
  - Preprocessing: resample to 16 kHz, 2 s windows (or document mismatch if different), RMS or instance norm as per WavJEPA README; no randomness at inference.

### A3. Embedding extraction

- **File:** `experiments/exp_002_wavjepa_emotion/extract_embeddings.py`
- **Responsibilities:**
  - Load frozen WavJEPA (e.g. `labhamlet/wavjepa-base` via Hugging Face or official repo).
  - For each (audio_path, label): load audio → preprocess → forward → save (embedding, label) per clip (or per 2 s window); optionally pool (mean) per file.
  - Output: one `.npy` or DataFrame with embeddings + labels, or a small cache dir (e.g. `experiments/exp_002_wavjepa_emotion/cache/embeddings.npz`).

### A4. Evaluation: clustering and optional shallow head

- **File:** `experiments/exp_002_wavjepa_emotion/evaluate.py`
- **Responsibilities:**
  - Load cached embeddings + labels.
  - Clustering: k-means with k = number of emotion classes; compute NMI, adjusted Rand index (and optionally accuracy by assigning cluster id to majority class).
  - Optional: train one linear or 1-layer MLP (embedding → emotion label); report accuracy and per-class metrics (e.g. sklearn).
  - Same train/val split for clustering and classifier (fixed seed).

### A5. Config and entrypoint

- **File:** `experiments/exp_002_wavjepa_emotion/config.yaml` (or `config.json`)
- **Contents:** Dataset roots (CREMA-D, RAVDESS), split ratio or fixed split file, 16 kHz, window length (e.g. 2 s), WavJEPA checkpoint id, output dir for metrics.

- **File:** `experiments/exp_002_wavjepa_emotion/run.py` (or single CLI script)
- **Responsibilities:** Parse config, run dataset load → extract_embeddings → evaluate; print or save metrics (NMI, adj Rand, accuracy).

### A6. Protocol and conclusion doc

- **File:** `docs/wavjepa_emotion_protocol.md` (or a section in an existing doc)
- **Contents:** Protocol (datasets, preprocessing, 2 s windows, no fine-tuning), dataset sizes, metrics (NMI, adjusted Rand, optional accuracy), and conclusion (separability sufficient for use case or not; “compressor vs semantic” takeaway).

---

## Option B: Compare WavJEPA to HuBERT / Wav2Vec baselines

**Goal:** Same emotional-separability protocol as A, plus 1–2 frozen baseline encoders; matched eval (same split, same shallow head, same metrics). Optionally 1–2 music tasks (genre/instrument).

### B1. Extend experiment layout

- **Dir:** Reuse `experiments/exp_002_wavjepa_emotion/` or add `experiments/wavjepa_baselines/` (recommended: same dir with encoder key in config).
- **Config:** Add `encoder: wavjepa | hubert_base | wav2vec2_base` (or explicit checkpoint ids). Same `dataset`, `split`, `metrics` for all.

### B2. Encoder abstraction and baseline loaders

- **File:** `experiments/exp_002_wavjepa_emotion/encoders.py` (or `experiments/wavjepa_baselines/encoders.py`)
- **Responsibilities:**
  - Common interface: `load_encoder(name)`, `extract(encoder, audio_16k)` → embedding matrix (same pooling strategy as A: e.g. per-clip mean or per-window).
  - Implement for: WavJEPA (existing extract script), HuBERT base (e.g. `facebook/hubert-base-ls960`), Wav2Vec2 base (e.g. `facebook/wav2vec2-base`). All frozen; preprocessing per model (WavJEPA: their norm; HuBERT/Wav2Vec2: standard 16 kHz).

### B3. Embedding extraction for all encoders

- **File:** Extend `experiments/exp_002_wavjepa_emotion/extract_embeddings.py` (or add `extract_all_encoders.py`)
- **Responsibilities:** Loop over `encoder in [wavjepa, hubert_base, wav2vec2_base]`; run same dataset → same splits → save embeddings per encoder (e.g. `cache/embeddings_wavjepa.npz`, `cache/embeddings_hubert.npz`, `cache/embeddings_wav2vec2.npz`).

### B4. Matched evaluation

- **File:** Extend `experiments/exp_002_wavjepa_emotion/evaluate.py`
- **Responsibilities:** For each encoder’s cache, run same clustering + same shallow head (linear or 1-layer MLP), same metrics (NMI, adj Rand, accuracy). Output one table: rows = encoders, columns = metrics (emotion task). Same train/val split and seed for all.

### B5. (Optional) Music tasks

- **Data:** Add 1–2 music datasets (e.g. genre or instrument from a standard set — e.g. GTZAN subset or a small MusicBrainz-tagged set) if readily available; otherwise skip and note “music tasks TBD.”
- **File:** `experiments/exp_002_wavjepa_emotion/dataset_music.py` or extend `dataset.py` with `task: emotion | genre | instrument`.
- **Eval:** Same pipeline: extract embeddings per encoder → same shallow head → accuracy/NMI. Add rows (or a second table) for music tasks.

### B6. Results table and methods paragraph

- **File:** `docs/wavjepa_baselines_results.md` (or section in `docs/wavjepa_emotion_protocol.md`)
- **Contents:**
  - Methods: “Frozen encoders (WavJEPA base ~90M, HuBERT base ~95M, Wav2Vec2 base ~95M); same train/val split; same shallow head (linear); inference-only, no pretraining.”
  - Table: Encoder | CREMA-D/RAVDESS NMI | Adj Rand | Accuracy | [optional: genre acc | instrument acc].
  - Short reproducibility note: dataset paths, preprocessing, and script commands (e.g. `python run.py --encoder wavjepa` etc.).

---

## Execution order (recommended)

1. **C1** → **C2** (optional): Design doc and cross-links.
2. **A1** → **A2** → **A3** → **A4** → **A5** → **A6**: WavJEPA-only emotion experiment and protocol doc.
3. **B1** → **B2** → **B3** → **B4** → **B5** (optional) → **B6**: Baseline comparison and results doc.

---

## What this task list does *not* include

- Training or fine-tuning WavJEPA.
- New datasets beyond CREMA-D/RAVDESS (and optional music set for B5).
- ARCH/HEAR full benchmark suites.
- Any implementation of the pipeline in C beyond the design doc.
