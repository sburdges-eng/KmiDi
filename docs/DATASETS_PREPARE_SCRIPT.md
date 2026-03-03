# Datasets in prepare_datasets.py

`scripts/utilities/prepare_datasets.py` can **download** and prepare **16 datasets**. Output goes under `AUDIO_DATA_ROOT` (or `KMIDI_DATA_ROOT/Datasets` if set in `.env`): `raw/<output_dir>`, `downloads/`, and optionally `processed/`.

## All datasets (--list)

| Key | Name | Task | Contains audio? | Notes |
|-----|------|------|-----------------|--------|
| **emotion_ravdess** | RAVDESS | emotion | Yes (WAV) | Kaggle |
| **emotion_cremad** | CREMA-D | emotion | Yes (WAV) | URL zip |
| **emotion_tess** | TESS | emotion | Yes | Kaggle |
| **gtzan** | GTZAN | emotion/genre | Yes | 10 genres, ~22050 Hz |
| **groove_midi** | Groove MIDI | groove | No (MIDI only) | Magenta |
| **maestro** | MAESTRO | melody | No (MIDI only) | Magenta |
| **lakh_midi** | Lakh MIDI (Clean) | harmony | No (MIDI only) | URL |
| **musicnet** | MusicNet | melody | Yes + MIDI | ~168 GB, Zenodo |
| **fma_small** | FMA Small | all | Yes | 8k tracks, 7.2 GB |
| **fma_medium** | FMA Medium | all | Yes | 25k tracks, 22 GB |
| **fma_full** | FMA Full | all | Yes | ~106k tracks, ~900 GB |
| **mtg_jamendo** | MTG-Jamendo | all | Metadata (TSV) | ~1 TB if you fetch audio separately |
| **nsynth_full** | NSynth (Full) | instrument | Yes (WAV) | ~30 GB |
| **musdb18** | MUSDB18 | source_separation | Yes | ~10 GB |
| **local_music** | Local Music | all | Yes | Copies from `~/Music` |

**Audio-capable (download gives you audio):** emotion_ravdess, emotion_cremad, emotion_tess, gtzan, musicnet, fma_small, fma_medium, fma_full, nsynth_full, musdb18, local_music.

**Faster MusicNet download:** The MusicNet tarball (~11 GB) is slow with a single connection. On your machine (with Ethernet and DNS), use multi-connection download: `./scripts/download_musicnet_aria2.sh` (or with `-c` to resume a partial). Requires `brew install aria2`.  
**MIDI-only (no audio in package):** groove_midi, maestro, lakh_midi.  
**Metadata only (audio elsewhere):** mtg_jamendo.

## Where files go

- **Default root** is chosen from: `AUDIO_DATA_ROOT` → `KMIDI_DATA_ROOT/Datasets` → sbdrive → Extreme SSD → Sean's SSD/Datasets → fallback `./kmidi_audio_data`.
- **Downloads:** `<root>/downloads/`
- **Raw (extracted):** `<root>/raw/<output_dir>/` (e.g. `raw/emotions/ravdess`, `raw/raw/fma_small`)

So if you use `KMIDI_DATA_ROOT="/Volumes/Sean's SSD"` in `.env`, the script writes under `/Volumes/Sean's SSD/Datasets/raw/` and `.../Datasets/downloads/`.

## Commands

```bash
# List datasets
python scripts/utilities/prepare_datasets.py --list

# Download one dataset (to AUDIO_DATA_ROOT / KMIDI_DATA_ROOT/Datasets)
python scripts/utilities/prepare_datasets.py --dataset emotion_ravdess --download

# Use a specific root
python scripts/utilities/prepare_datasets.py --root "/Volumes/Sean's SSD/Datasets" --dataset gtzan --download

# Download all (can be very large)
python scripts/utilities/prepare_datasets.py --dataset all --download
```

So yes: **more audio datasets are available via the script** than what’s currently on disk; they appear only after you run `--download` (and optionally point `--root` or env at your external Datasets folder).
