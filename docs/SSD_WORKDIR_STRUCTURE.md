# SSD workdir structure (KMIDI_DATA_ROOT)

When the external SSD is mounted (e.g. **KmiDi-external** or **Sean's SSD**), set `KMIDI_DATA_ROOT` in `.env` to its mount path. Scripts then use these folders without hardcoding the volume name.

```
$KMIDI_DATA_ROOT  (e.g. /Volumes/KmiDi-external  or  /Volumes/Sean's SSD)
│
├── Datasets/                          ← AUDIO_DATA_ROOT when KMIDI_DATA_ROOT is set
│   ├── downloads/                     ← prepare_datasets + aria2 put tarballs here
│   │   └── musicnet.tar.gz
│   ├── raw/                            ← extracted datasets by prepare_datasets
│   │   ├── emotions/
│   │   │   ├── ravdess/
│   │   │   ├── cremad/
│   │   │   └── tess/
│   │   ├── melodies/
│   │   │   └── musicnet/
│   │   ├── raw/
│   │   │   ├── fma_small/
│   │   │   └── ...
│   │   └── ...
│   ├── processed/                     ← optional post-processed output
│   │   └── ...
│   ├── COLD_STORAGE/                  ← archive / long-term (KMIDI_DATASETS_PATH often here)
│   │   └── kmidi_datasets/
│   └── cache/                         ← KMIDI_CACHE_ROOT (when set)
│       ├── pip/
│       ├── npm/
│       └── cargo/
│
├── build/                             ← out-of-tree CMake builds (when using volume)
│   └── <project>/
│
└── Models/                            ← or ~/Models; checkpoints, weights
    └── checkpoints/
```

**Env (in `.env` when SSD is connected):**

| Var | Typical value |
|-----|----------------|
| `KMIDI_DATA_ROOT` | `/Volumes/KmiDi-external` or `/Volumes/Sean's SSD` |
| `KMIDI_DATASETS_PATH` | `$KMIDI_DATA_ROOT/Datasets/COLD_STORAGE/kmidi_datasets` |
| `KMIDI_CACHE_ROOT` | `$KMIDI_DATA_ROOT/Datasets/cache` |

**Note:** Repo and active code stay in `~/Dev/KmiDi`. The SSD holds data, cache, and build outputs only.

---

## Actual layout on Sean's SSD (live view)

Scanned from `/Volumes/Sean's SSD` — full workdir **outside** the repo:

```
/Volumes/Sean's SSD
│
├── Datasets/
│   ├── 00_catalog/
│   ├── 10_geospatial/
│   ├── 20_audio_music/
│   ├── 30_audio_speech/
│   ├── 40_multimodal/
│   ├── 50_cache/
│   ├── 60_staging_unclassified/
│   ├── 70_quarantine/
│   ├── 90_archive_readonly/
│   ├── CO_NAIP_2021_9667/
│   ├── consolidated/
│   ├── downloads/                     ← prepare_datasets / aria2
│   │   ├── groove-v1.0.0-midionly.zip
│   │   ├── lmd_matched.tar.gz
│   │   ├── maestro-v3.0.0-midi.zip
│   │   ├── master.zip
│   │   └── musicnet.tar.gz.partial
│   ├── processed/
│   └── raw/
│       ├── chord_progressions/
│       ├── emotions/
│       ├── grooves/
│       └── melodies/
│
├── COLD_STORAGEEXTERNAL/
│   ├── audio/
│   ├── ml-training-suite/
│   ├── models/
│   └── recovered_intelligence/
│
├── Dev/
├── DevEXTERNAL/
│   ├── KmiDi MIDI Companion/
│   ├── Models/
│   ├── .cursor/
│   └── ...
├── KmiDi/
├── KmiDiEXTERNAL/
├── KmiDi_MASTER_VAULTEXTERNAL/
├── GH_REPOS/
├── Models/                            (empty at top level)
├── ModelsEXTERNAL/
├── build/                             (empty at top level)
├── cacheEXTERNAL/
│   └── vehicle_geometry/
├── iDAW_Samples/
├── MusicEXTERNAL/
├── hf_home/
├── CLEANUP_RECOVERY_20260225-032025/
├── KmiDi-Backup-* (zip + dirs)
└── ... (other EXTERNAL / project folders)
```

So on this SSD, **COLD_STORAGE** and **cache** are at volume root as `COLD_STORAGEEXTERNAL` and `cacheEXTERNAL`; **Datasets** matches the doc (downloads, raw, processed). Use `KMIDI_DATA_ROOT="/Volumes/Sean's SSD"` and point `KMIDI_DATASETS_PATH` / `KMIDI_CACHE_ROOT` at the actual folder names you use (e.g. `COLD_STORAGEEXTERNAL`, `cacheEXTERNAL`) if they differ from the canonical names.
