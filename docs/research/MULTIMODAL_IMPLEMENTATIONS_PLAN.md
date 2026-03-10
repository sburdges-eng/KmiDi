# Multimodal Pipeline Implementations — Plan

**Scope:** Concrete implementations derived from `MULTIMODAL_REPRESENTATIONS_2026.md` for KmiDi’s audio ↔ symbolic ↔ control stack.  
**Status:** Planning (execution to be done in follow‑up PRs).  

## 1. Perch + REMI‑BPE Prototype Pipeline (Top Priority)

**Goal:** Stand up a minimal, end‑to‑end research pipeline that:

- Computes **Perch audio embeddings** on MAESTRO (or a small subset).  
- Uses **REMI‑BPE tokenization** (MidiTok + Maestro‑REMI‑bpe20k) for symbolic targets.  
- Demonstrates a simple **alignment task** (e.g., nearest‑neighbour retrieval or shallow supervised head) without full JEPA training yet.

### 1.1 Repository Layout

- `experiments/perch_remi_pipeline/`
  - `embed_perch.py` — CLI to:
    - Read audio files (MAESTRO or small test set).  
    - Run Perch encoder to produce 1536‑dim embeddings on fixed windows (e.g., 4–8 s).  
    - Emit JSONL/NPZ with:
      - `audio_path`, `start`, `duration`, `embedding` (or reference to `.npy` file), `sha1_audio`.  
  - `remi_bpe_demo.py` — CLI / small notebook to:
    - Load MidiTok REMI tokenizer and `Natooz/Maestro-REMI-bpe20k`.  
    - Tokenize a MIDI file, run short generation, decode to MIDI.  
    - Log token sequence lengths and sanity‑check round‑trip.

Optional (stretch):

- `align_perch_to_remi.py` — quick script that:
  - Loads Perch embeddings and aligned REMI‑BPE token sequences (via a shared manifest).  
  - Trains a shallow projection (e.g., MLP) to predict next REMI token distribution from the audio embedding or to score matching vs non‑matching pairs.

### 1.2 Interfaces & Config

- CLI pattern (both scripts):

```bash
python embed_perch.py \
  --audio-root ~/Datasets/maestro/audio \
  --pattern "**/*.wav" \
  --window-seconds 8.0 \
  --stride-seconds 4.0 \
  --output embeddings_maestro_perch.jsonl

python remi_bpe_demo.py \
  --midi-path ~/Datasets/maestro/midi/2011/track.mid \
  --max-new-tokens 256 \
  --output generated_track.mid
```

- Config via a small `yaml`/`toml` in `experiments/perch_remi_pipeline/config/`:
  - Model paths/checkpoints for Perch (local or from HF).  
  - Tokenizer/model IDs for REMI‑BPE.

### 1.3 Data Contracts

- Perch embedding JSONL fields (per window):
  - `id`: unique id (e.g., `maestro_2011_01_w000123`).  
  - `audio_path`: absolute or dataset‑relative path.  
  - `start`, `duration`: seconds.  
  - `embedding_path`: path to `.npy` with a `1536` vector (or inline array for small experiments).  
  - `sha1_audio`: hash of the full source audio file.

- REMI‑BPE token JSONL fields (aligned or standalone):
  - `midi_path`.  
  - `token_ids`: list of ints.  
  - `num_tokens`.  
  - Optional: `bars`, `tempo` metadata extracted by MidiTok.

### 1.4 Execution Plan

- **Step 1:** Vendor Perch dependency into `requirements-experiments.txt` or a dedicated env file; confirm a minimal example runs against a local audio file.  
- **Step 2:** Implement `embed_perch.py` with:
  - Deterministic windowing.  
  - Parallelism that is simple but safe (e.g., `multiprocessing` with bounded workers).  
- **Step 3:** Implement `remi_bpe_demo.py` that:
  - Loads Maestro‑REMI‑bpe20k from HF.  
  - Tokenizes and round‑trips a handful of MIDI files.  
- **Step 4 (optional):** Implement `align_perch_to_remi.py` for a trivially small supervised/contrastive alignment experiment; log metrics and plots to `experiments/perch_remi_pipeline/logs/`.

---

## 2. JEPA Manifest Generator (Lhotse + DataLad)

**Goal:** Define and generate a reproducible JEPA manifest for MAESTRO and similar datasets using Lhotse and DataLad, matching the schema in `MULTIMODAL_REPRESENTATIONS_2026.md`.

### 2.1 Script & Location

- `KmiDi_PROJECT/scripts/make_jepa_manifest.py`
  - Uses Lhotse Python API to:
    - Build `RecordingSet` from audio paths.  
    - Build `SupervisionSet` from aligned MIDI sidecars and optional emotion / BPM metadata.  
    - Derive a `CutSet` with fixed windows or supervision‑aligned segments.
  - Writes:
    - `recordings.jsonl`, `supervisions.jsonl`, `cuts.jsonl` under a specified `manifests/` directory.  
    - An additional `cuts_with_hashes.jsonl` that includes `sha1_audio` and `sha1_midi`.

### 2.2 DataLad Integration

- Recommended pattern:

```bash
datalad create kmidi-mini-maestro
cd kmidi-mini-maestro
mkdir manifests

python -m KmiDi_PROJECT.scripts.make_jepa_manifest \
  --audio-root /path/to/maestro/audio \
  --midi-root /path/to/maestro/midi \
  --out-dir manifests

datalad save -m "Add Lhotse manifests for JEPA windows"

datalad run -m "Regenerate 8s windows from supervisions" \
  "python -m KmiDi_PROJECT.scripts.make_jepa_manifest --...same-args..."
```

### 2.3 Schema Alignment

- Ensure fields match the manifest sketch from the research doc:
  - Recording + Supervision + Cut info, plus `custom` dict with:
    - `midi_sidecar`, `emotion_label`, `bpm`.  
  - Top‑level `sha1_audio` and `sha1_midi`.

---

## 3. Shared C ABI Engine + RT Harness

**Goal:** Introduce a single C ABI (`engine.h`, `libengine`) around the C++ DSP/latent engine and a headless real‑time harness with CI guardrails.

### 3.1 Engine Library

- New directory: `engine/`
  - `include/engine.h` — C header with:
    - `kmidi_engine_config_t`, `kmidi_engine_t` (opaque).  
    - `kmidi_engine_create/destroy`, `kmidi_engine_prepare`, `kmidi_engine_process`.  
    - Optional `kmidi_engine_query_scratch`, `kmidi_engine_last_error`.
  - `src/` — minimal implementation wrapping existing DSP graph.
  - `CMakeLists.txt` — builds `libengine` (static) with install rules.

### 3.2 Consumers

- **JUCE AUv3 / plugin:**  
  - In `prepareToPlay`, call `kmidi_engine_prepare`.  
  - In `processBlock`, map JUCE buffers to `float* const*` and call `kmidi_engine_process`.
- **Rust/Tauri:**  
  - New crate (e.g., `crates/engine-ffi`) with `extern "C"` bindings and a safe wrapper.  
  - Linked against `libengine` using `build.rs`.

### 3.3 RT Harness + CI

- New target: `rt_harness/`:
  - Runs in headless mode, loads a “golden” preset, simulates audio callbacks at a fixed buffer size and SR.  
  - Measures callback durations (P50/P90/P99) and writes `callback_stats.json`.
- CI job:
  - Builds `engine` + `rt_harness`.  
  - Runs harness for a fixed duration (e.g., 30–60 s).  
  - Fails if `p90_us` exceeds a configured threshold.

---

## 4. MIDI 2.0 PE + UMP Affect Channel

**Goal:** Establish an official MIDI 2.0 affect channel for KmiDi using Property Exchange (PE) for schema and UMP 32‑bit controllers for live control lanes.

### 4.1 PE Resource

- Implement `com.sburdges.kmidi/affect.v1` resource in the MIDI I/O layer:
  - Properties: `valence`, `arousal`, `dynamics`, `timestamp`, `mode`.  
  - JSON schema stored in a small helper (e.g., `midi/pe_affect_schema.json`).

### 4.2 UMP Channel Voice Mapping

- Choose vendor controller indices (e.g. `0x28`..`0x2A`) for:
  - `valence`, `arousal`, `dynamics`.  
- Implement helpers to:
  - Map floats to 32‑bit ints.  
  - Pack and send UMP 32‑bit CC messages at control‑rate.

### 4.3 Test Harness

- Small script under `KmiDi_PROJECT/scripts/` (Python or C++):
  - Connects to a local UMP endpoint.  
  - Streams synthetic affect curves (e.g., sine waves, ramps).  
  - Lets you verify lanes in a DAW that supports MIDI 2.0.

---

## 5. Expressive Controller Prototyping (Sensel Morph / MPE)

**Goal:** Use devices like Sensel Morph or K‑Board Pro 4 to capture high‑resolution, per‑note expressive control mapped into the affect channel and/or MPE parameters.

**Status:** Implemented. Profile doc: `midi/sensel_kmidi_profile.md`. Bridge: `scripts/morph_affect_bridge.py` (MIDI-in and optional Sensel API; optional UMP forward). See `midi/README.md` § Expressive Controller.

### 5.1 Sensel Morph Profile

- Define a SenselApp profile:
  - Map key/zone pressure and position to:  
    - MPE channels.  
    - Secondary CCs for valence/arousal/dynamics, or for intermediate features feeding the Brain.

### 5.2 Integration

- Add a small “Morph bridge” module:
  - Normalizes Morph data to KmiDi’s internal control schema.  
  - Optionally forwards values into the PE + UMP affect path.

