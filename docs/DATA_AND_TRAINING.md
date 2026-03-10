# Data & Training Governance

**Governance: DATA LAW, CUDA TRAINING GOVERNANCE**

All contributors and automation must follow these rules when working with datasets,
model weights, and GPU training runs.

---

## 1. Dataset Paths — DATA LAW

| Rule | Detail |
|------|--------|
| **Canonical root** | `~/Datasets` — all raw and processed datasets live here only |
| **No data in repo** | Never commit audio, MIDI, video, or large binary dataset files |
| **Symlinks OK** | If a dataset lives on an external drive or SSD, symlink it: `ln -s /Volumes/Extreme\ SSD/kelly-audio-data ~/Datasets/kelly-audio-data` |
| **Env override** | Set `KELLY_AUDIO_DATA_ROOT` to point to an alternative root; scripts read this variable |

**Dataset root resolution order** (used by experiments and dataset loaders): `KMIDI_DATASETS_PATH` → `AUDIO_DATA_ROOT` → config `dataset_root` → `~/Datasets`. See [DISCOVERY_DATASETS_KMIDI_PATH.md](DISCOVERY_DATASETS_KMIDI_PATH.md). Optional location `~/Datasets/by_source/kmidi` is not referenced in code; it may hold KmiDi-sourced audio if you use it locally.

### Standard layout under `~/Datasets/`

```
~/Datasets/
  kelly-audio-data/         # main audio corpus
    raw/                    # immutable originals
    processed/              # preprocessed / augmented
    splits/                 # train / val / test manifests
  groove_midi/
  maestro/
  fma_small/
  ...
```

---

## 2. Model Weights & Checkpoints — DATA LAW

| Rule | Detail |
|------|--------|
| **Canonical root** | `~/Models` — all saved weights and checkpoints live here |
| **Checkpoints subdir** | `~/Models/checkpoints/<experiment_name>/` |
| **No weights in repo** | Never commit `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, or similar binary weight files |
| **Env override** | Set `KELLY_MODEL_ROOT` to point to an alternative root |

### Standard layout under `~/Models/`

```
~/Models/
  checkpoints/
    <experiment_name>/
      run_manifest.yaml     # required — see §4
      epoch_001.ckpt
      epoch_002.ckpt
      best.ckpt
  exported/                 # ONNX / TorchScript exports
  pretrained/               # downloaded third-party weights
```

---

## 3. Training Run Requirements — CUDA TRAINING GOVERNANCE

Every GPU training event **must** satisfy all of the following before launch:

- [ ] **Experiment name** — unique, descriptive slug (e.g. `wavjepa-emotion-v2-2026-03`)
- [ ] **Run manifest** — `run_manifest.yaml` placed in the checkpoint dir (see §4)
- [ ] **Config file** — training hyperparameters committed to `configs/` *before* the run
- [ ] **No repo output** — checkpoint and log output must go to `~/Models/` and `~/Datasets/`, never into the repo tree
- [ ] **Reproducibility** — random seeds, dataset split hashes, and dependency versions recorded in the manifest
- [ ] **Budget cap** — estimated GPU hours and cost captured in manifest; stop if exceeded

### Recommended directory layout inside the repo

```
configs/          # training/model configs (YAML / JSON) — committed
experiments/      # lightweight result summaries, NOT weights — committed selectively
```

---

## 4. Run Manifest Template

Create `~/Models/checkpoints/<experiment_name>/run_manifest.yaml` before each run:

```yaml
# run_manifest.yaml — required for every GPU training event
schemaVersion: "1.0"

experiment:
  name: "<slug>"                 # unique, descriptive (no spaces)
  description: ""
  started_at: ""                 # ISO-8601, filled at launch
  finished_at: ""                # ISO-8601, filled at completion

reproducibility:
  random_seed: 42
  dataset_split_hash: ""         # sha256 of the splits manifest
  config_file: "configs/<name>.yaml"
  git_commit: ""                 # filled automatically by launch script
  python_version: ""
  cuda_version: ""
  torch_version: ""

paths:
  data_root: "~/Datasets"
  checkpoint_dir: "~/Models/checkpoints/<experiment_name>"
  log_dir: "~/Models/checkpoints/<experiment_name>/logs"

budget:
  estimated_gpu_hours: 0
  max_gpu_hours: 24              # run is stopped if exceeded
  cloud_provider: ""             # e.g. lambda, aws, local

status: pending                  # pending | running | completed | failed
notes: ""
```

---

## 5. Training Safety Checklist

Run through this checklist before every training job:

1. **Data root set** — `echo $KELLY_AUDIO_DATA_ROOT` returns a path under `~/Datasets`
2. **No repo writes** — checkpoint and log dirs are outside the repo (`~/Models/…`)
3. **Config committed** — the YAML/JSON config for this run is in `configs/` and pushed
4. **Manifest created** — `run_manifest.yaml` exists in the checkpoint dir
5. **Seeds fixed** — `random_seed` is explicitly set (not `None` or random)
6. **Budget cap set** — `max_gpu_hours` is a finite positive number
7. **Experiment name unique** — no existing dir with the same name in `~/Models/checkpoints/`
8. **`.gitignore` checked** — no accidental large-file staging (`git status` is clean)

---

## 6. `.gitignore` Reminders

The repo `.gitignore` already excludes common dataset and weight extensions.
If you add a new format, ensure it is covered:

```
# datasets — never commit
*.wav  *.mp3  *.flac  *.ogg  *.aiff  *.aac
*.tfrecord  *.parquet  *.h5  *.hdf5

# model weights — never commit
*.pt  *.pth  *.ckpt  *.safetensors  *.onnx  *.bin
```

---

## References

- `configs/README.md` — conventions for the `configs/` directory
- `.cursor/rules/engineering-governance.mdc` — Cursor AI enforcement rules
- `docs/ENVIRONMENT.md` — all environment variables (`KELLY_AUDIO_DATA_ROOT`, etc.)
- `docs/DATASETS_PREPARE_SCRIPT.md` — how to download and prepare datasets
