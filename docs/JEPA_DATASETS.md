# JEPA Datasets — Acquire / Create Real Data

JEPA training configs (`configs/jepa_*.yaml`) expect a manifest at `~/Datasets/kmidi_jepa/manifests/aligned.jsonl`. Each line is one aligned triple: audio, MIDI, optional Spectocloud. **No data in repo** — everything under `~/Datasets` (DATA LAW).

Create dirs first: `./scripts/setup_training_env.sh`

---

## Real data sources (research-backed)

### 1. MAESTRO (aligned piano audio + MIDI)

- **What:** ~200 hours piano, ~3 ms alignment. WAV + MIDI, train/val/test split.
- **License:** CC BY-NC-SA 4.0.
- **URLs (v3.0.0):**
  - Full (audio + MIDI): `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip` — **~101 GB** download, ~120 GB uncompressed.
  - MIDI only: `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip` — **~56 MB** (81 MB unpacked). Use this + render to WAV for a small local copy.
  - Metadata CSV: `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv` — filenames, split, duration.
- **Use:** Full zip → unpack → `python scripts/acquire_maestro.py --build-manifest`. Or midi-only zip → unpack → render MIDI to WAV (fluidsynth) → build manifest (see scripts below).

### 2. Lakh MIDI (aligned to audio)

- **What:** 45,129 MIDI files matched to Million Song Dataset; LMD-aligned has 7digital preview MP3s.
- **URLs:** [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/) — LMD-aligned (audio): `http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz`; LMD-matched (MIDI only): `http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz`.
- **Use:** Download aligned tarball, unpack to `~/Datasets/kmidi_jepa/lakh_aligned/`, then a script can build `aligned.jsonl` from the directory layout (format varies; see Lakh docs).

### 3. Slakh2100 (multi-track + MIDI)

- **What:** 2100 tracks, mixture + stems + MIDI. **~105 GB** download, ~500 GB as WAV.
- **URL:** [Zenodo 4599666](https://zenodo.org/record/4599666). BabySlakh subset exists for prototyping.
- **Use:** For JEPA you can use mixture (or stem) + corresponding MIDI; build manifest from Slakh folder structure.

---

## Recommended path when nothing exists locally

**Option A — Small footprint (MAESTRO midi-only + synthetic audio)**  
No 101 GB download. You get real MIDI and aligned synthetic WAV.

1. Run: `python scripts/acquire_real_data.py`  
   - Downloads MAESTRO midi-only zip (~56 MB) to `~/Datasets/kmidi_jepa/maestro_midi/`.
   - Unpacks MIDI.
   - Optionally renders MIDI → WAV with **fluidsynth** (requires `fluidsynth` and a SF2 soundfont; see below).
   - Writes `~/Datasets/kmidi_jepa/manifests/aligned.jsonl`.

2. For WAV: install fluidsynth (`brew install fluidsynth`) and set `FLUIDSYNTH_SF2` to a .sf2 path (e.g. FluidR3_GM.sf2). Script skips render if fluidsynth/sf2 missing and still builds manifest with real midi_path and empty audio_path.

**Option B — Full MAESTRO (real piano audio)**  
~101 GB download. Run in **tmux** so the job survives disconnects.

**Monitoring, throttling, resume:**
- **Progress:** Download and unzip report progress (bytes/GB, speed, ETA every 5 s).
- **Throttling:** Cap bandwidth with `--throttle-mbps N` (e.g. `10` for 10 MB/s) to avoid saturating the link.
- **Resume:** By default, a partial zip is resumed; use `--no-resume` to start over.

**Steps:**

1. CSV (required for manifest):  
   `python scripts/acquire_maestro.py --csv-only`

2. Download full zip (progress + optional throttle):  
   `python scripts/acquire_maestro.py --download [--throttle-mbps 10]`  
   Resume is automatic if the same zip path already exists.

3. Unzip with progress:  
   `python scripts/acquire_maestro.py --unzip`

4. Build manifest:  
   `python scripts/acquire_maestro.py --build-manifest`

5. Analysis (rows, duration, split, disk usage, path validation):  
   `python scripts/acquire_maestro.py --analyze [--analyze-samples 10]`

**One-shot Option B (all steps):**  
`python scripts/acquire_maestro.py --csv-only && python scripts/acquire_maestro.py --download --throttle-mbps 10 && python scripts/acquire_maestro.py --unzip && python scripts/acquire_maestro.py --build-manifest && python scripts/acquire_maestro.py --analyze`

---

## Scripts

| Script | Purpose |
|--------|--------|
| `scripts/setup_training_env.sh` | Create `~/Datasets/kmidi_jepa/manifests`, `kmidi_learning`, `~/Models/checkpoints`. |
| `scripts/create_jepa_manifest_stub.py` | Empty/minimal `aligned.jsonl` for config validation. |
| `scripts/acquire_maestro.py` | Option B: CSV, **download** (progress, `--throttle-mbps`, resume), **unzip** (progress), **build manifest**, **analyze** (rows, duration, split, disk, validation). |
| `scripts/acquire_real_data.py` | Option A: MAESTRO midi-only zip, unpack, optional MIDI→WAV render, build manifest. |

---

## Manifest format

See `data/manifests/README.md`. One JSON object per line:

```json
{"audio_path": "/path/to/audio.wav", "midi_path": "/path/to/midi.mid", "specto_path": "", "start_offset": 0.0, "tempo": 120, "timebase": 480}
```

`specto_path` can be `""`; training can skip Spectocloud. Tempo/timebase can be defaulted (e.g. 120, 480) if unknown.
