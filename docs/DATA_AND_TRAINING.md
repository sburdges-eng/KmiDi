# Data Paths and Training Governance

Datasets and model weights live **outside** the repo. Every training event must be traceable. (Governance: DATA LAW, CUDA TRAINING GOVERNANCE.)

## Paths

| Purpose        | Location                  | Notes |
|----------------|---------------------------|--------|
| **Datasets**   | `~/Datasets`              | No large data in repo; use symlinks if needed |
| **Model weights** | `~/Models`             | |
| **Checkpoints**   | `~/Models/checkpoints` | Per-run checkpoints; never in repo |

Never store large datasets inside the repository. Never duplicate datasets. Warn if dataset sprawl appears.

**JEPA + current training env:** See [TRAINING_ENV.md](TRAINING_ENV.md). **JEPA datasets (acquire/create):** See [JEPA_DATASETS.md](JEPA_DATASETS.md). **Cloud GPU training:** See [CLOUD_TRAINING.md](CLOUD_TRAINING.md). Local learning dirs: `./scripts/setup_training_env.sh`.

## Repo layout for training

- **configs/** — Training/config YAML or JSON (in repo).
- **experiments/** — Experiment dirs `exp_NNN_description` (in repo); see [experiments/README.md](../experiments/README.md).
- **Checkpoints and logs** — Always under `~/Models/checkpoints` (or a documented subdir), not in repo.

## Before any GPU / large training run

1. **Dataset location** — Confirm data under `~/Datasets` (or symlink).
2. **Checkpoint destination** — Set to `~/Models/checkpoints/<experiment_name>`.
3. **Experiment naming** — Use `exp_NNN_description` and reference in run manifest.
4. **Run manifest** — One file per run: experiment name, config path, dataset path, checkpoint path, seed, timestamp (see template below).
5. **Reproducibility** — Record seed, env (e.g. conda/venv), and config hash or git commit.
6. **No output into repo** — Logs and checkpoints go to `~/Models` (or documented path), not `./logs` or `./checkpoints` in repo.

## Run manifest template

Create e.g. `experiments/exp_NNN_description/manifest_run_YYYYMMDD_HHMM.json` (or `.yaml`):

```json
{
  "experiment": "exp_NNN_description",
  "config": "configs/exp_NNN_config.yaml",
  "dataset_path": "~/Datasets/...",
  "checkpoint_dir": "~/Models/checkpoints/exp_NNN_description",
  "seed": 42,
  "git_commit": "<hash>",
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
}
```

## Training safety checklist

Before large jobs, confirm:

- [ ] Disk space
- [ ] Checkpoint path (e.g. `~/Models/checkpoints/...`)
- [ ] Logging path
- [ ] Resume capability (interval, path)
- [ ] Fail-safe checkpoint interval to avoid silent storage exhaustion
