# Dataset download/prepare scripts on external drive (Sean's SSD)

Found on **`/Volumes/Sean's SSD`** (the currently mounted drive). Use for reference when locating or consolidating dataset tooling.  
*Currently mounted:* Yes — this catalog was generated from the drive at `/Volumes/Sean's SSD`.

---

## KmiDi_MASTER_VAULTEXTERNAL / KmiDi (canonical-style copies)

| Path | Type |
|------|------|
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/prepare_datasets.py` | prepare |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/utilities/prepare_datasets.py` | prepare |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/download_emotion_datasets.sh` | download |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/training/download_datasets_background.py` | download |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/hash_emotion_datasets.sh` | hash |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/normalize_emotion_datasets.py` | normalize |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/preprocess_emotion_datasets.py` | preprocess |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/KmiDi_PROJECT/scripts/prepare_datasets.py` | prepare |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/ML Kelly Training/backup/scripts/prepare_datasets.py` | prepare |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/_ARCHIVE_AUDIT/redundant/src_kmidi/scripts/prepare_datasets.py` | prepare |
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/tests/training/test_dataset_download.py` | test |

---

## GH_REPOS / kelly-project

| Path | Type |
|------|------|
| `GH_REPOS/kelly-project/brain-python/scripts/prepare_datasets.py` | prepare |
| `GH_REPOS/kelly-project/brain-python/scripts/download_emotion_datasets.py` | download |
| `GH_REPOS/kelly-project/brain-python/scripts/download_premium_datasets.py` | download |
| `GH_REPOS/kelly-project/brain-python/scripts/process_emotion_datasets.py` | process |
| `GH_REPOS/kelly-project/brain-python/scripts/prepare_g2p_dataset.py` | prepare |
| `GH_REPOS/kelly-project/ml_training/prepare_g2p_dataset.py` | prepare |
| `GH_REPOS/kelly-project/audio-engine-cpp/ml_training/prepare_datasets.py` | prepare |
| `GH_REPOS/kelly-project/audio-engine-cpp/ml_training/prepare_g2p_dataset.py` | prepare |

---

## Dev (recovery / other)

| Path | Type |
|------|------|
| `Dev/recovery_quality_patch/final_kel/scripts/download_datasets.py` | download |
| `Dev/swif:xcode/KmiDi/scripts/prepare_webdataset_shards.py` | prepare |

---

## My MacEXTERNAL / KmiDi MIDI Companion

| Path | Type |
|------|------|
| `My MacEXTERNAL/KmiDi MIDI Companion/FINAL_KMIDI/scripts/...` | (download_datasets_background cache/tests) |
| `My MacEXTERNAL/KmiDi MIDI Companion/CANONICAL_REBUILD/KmiDi/scripts/utilities/prepare_datasets.py` | prepare |
| `My MacEXTERNAL/KmiDi MIDI Companion/Desktop/final kel/CONSOLIDATED_CODE/ml_training/prepare_datasets.py` | prepare |

---

## Download/output directories on external

| Path |
|------|
| `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/scripts/downloads` |
| `COLD_STORAGEEXTERNAL/audio/kelly-audio-data/downloads` |
| `KmiDi/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/downloads` |
| `KmiDi/KmiDi-1/scripts/downloads` |
| `KmiDi/KmiDi/scripts/downloads` |
| `Datasets` (root-level; contains 20_audio_music, 60_staging_unclassified, etc.) |

---

**Active repo:** This workspace uses `scripts/utilities/prepare_datasets.py` (in repo). External copies are for recovery or legacy; prefer the single script in `~/Dev/KmiDi` and `AUDIO_DATA_ROOT` or `KMIDI_DATASETS_PATH` for paths.
