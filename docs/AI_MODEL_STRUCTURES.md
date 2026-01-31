# AI Model Structures — Configs, Registry, Experiments

**Purpose:** Single reference for where AI/ML model configs, manifests, and run artifacts live. Aligns with DATA LAW, TRAINING SAFETY LAW, and penta_core model_registry.

**References:** [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md), [TRAINING_ENV.md](TRAINING_ENV.md), [experiments/README.md](../experiments/README.md), [configs/README.md](../configs/README.md).

---

## 1. Where things live

| What | Location | Notes |
|------|----------|--------|
| **Model weights / checkpoints** | `~/Models/checkpoints/<exp_name>` | Never in repo. Per DATA LAW. |
| **Exported models** | `~/Models/exported/<exp_name>` | ONNX, CoreML, etc. |
| **Logs** | `~/Models/logs/<exp_name>` or `~/Models/logs/shadow` | Training logs; shadow JEPA logs. |
| **Datasets** | `~/Datasets` (e.g. `kmidi_learning`, `kmidi_jepa`) | Symlink or manifest path; no large data in repo. |
| **Configs (in repo)** | `configs/*.yaml` | JEPA: `jepa_audio`, `jepa_midi`, `jepa_multimodal`, `jepa_specto`. Experiment: copy from `configs/model_exp_template.yaml`. |
| **Registry manifest** | `data/manifests/registry.json` | penta_core discovers models from this (or load_registry_manifest path). Paths in manifest point to `~/Models` or relative. |
| **Registry schema** | `data/manifests/registry.schema.json` | JSON schema for `registry.json`; used by model_registry when `jsonschema` is present. |
| **Run manifests** | `experiments/exp_NNN_*/manifest_run_*.json` | One per run: experiment name, config, dataset, checkpoint dir, seed, timestamp. |
| **Data manifests** | `data/manifests/aligned.jsonl` | Aligned triples (audio, MIDI, spectocloud); path can override to `~/Datasets/...`. |

---

## 2. Model registry (penta_core)

The Brain uses **penta_core.model_registry** to discover and load models. It can:

- **Load a manifest:** `load_registry_manifest(path)` where `path` is to a `registry.json`.
- **Validate:** If `registry.schema.json` exists next to the manifest and `jsonschema` is installed, the manifest is validated.

### registry.json format

- **registry_version** (optional): integer for schema evolution.
- **model_dirs** (optional): list of dirs to scan (e.g. `~/Models/checkpoints`, `~/Models/exported`).
- **models**: list of entries. Each entry:
  - **id** (required): unique name for `get_model(id)`.
  - **task**: one of `emotion_embedding`, `melody_generation`, `harmony_prediction`, `dynamics_mapping`, `groove_prediction`, `intent_mapping`, `audio_classification`, `chord_prediction`, `chord_detection`, `key_detection`, `tempo_estimation`, `style_transfer`, `emotion_classification`, `audio_generation`, `onset_detection`, `beat_tracking`, `custom`.
  - **format**: `onnx`, `coreml`, `pytorch`, `torchscript`, `rtneural-json`, `tflite`, etc.
  - **file** (or **onnx_path** / **coreml_path**): path to model file (absolute or relative to manifest dir).
  - **version**, **input_size**, **output_size**, **sample_rate**, **inference_target_ms**, **note**, **license**, **status**: optional metadata.

Example: see `data/manifests/registry.json`. Point **file** at real paths under `~/Models` once you have trained/exported models.

---

## 3. Experiment config template

- **Template:** `configs/model_exp_template.yaml`.
- **Use:** Copy to `configs/exp_NNN_short_name.yaml`. Fill in `experiment`, `paths` (dataset, checkpoint_dir, log_dir, export_dir), and copy modality-specific `data`/`model`/`train` blocks from `configs/jepa_*.yaml` or from `ML_TRAINED_MODELS/ml/ml-training-suite/configs/config.yaml`.
- **Paths:** All run artifacts go under `~/Models` and `~/Datasets`; no checkpoints or large data in repo.

---

## 4. Run manifest (traceability)

For each training run, create a run manifest in the experiment dir:

- Path: `experiments/exp_NNN_description/manifest_run_YYYYMMDD_HHMM.json`.
- Fields: `experiment`, `config`, `dataset_path`, `checkpoint_dir`, `seed`, `git_commit`, `timestamp`.
- Template: [DATA_AND_TRAINING.md § Run manifest template](DATA_AND_TRAINING.md#run-manifest-template).

---

## 5. Model types in this repo

| Type | Config / location | Registry task / use |
|------|-------------------|----------------------|
| **JEPA (audio)** | `configs/jepa_audio.yaml` | Training; not yet in registry until exported. |
| **JEPA (MIDI)** | `configs/jepa_midi.yaml` | Same. |
| **JEPA (multimodal / specto)** | `configs/jepa_multimodal.yaml`, `jepa_specto.yaml` | Same. |
| **Emotion / voice (LoRA, etc.)** | `ML_TRAINED_MODELS/ml/ml-training-suite/configs/config.yaml` | `emotion_classification`, `emotion_embedding`; register exported model in `registry.json`. |
| **penta_core (inference)** | `KmiDi_CANON/brain/penta_core/ml/` | model_registry, inference.py; load `registry.json` or add model_dirs. |
| **LLM (intent)** | Env: `KMI_DI_LLM_MODEL_PATH` | Not in penta_core registry; used by mcp_workstation LLM engine. |
| **Image / audio gen** | Env: `KMI_DI_IMAGE_MODEL_PATH`, `KMI_DI_AUDIO_MODEL_ID` | Stub or optional; see BOOT.md. |

---

## 6. Quick checklist for a new AI model run

1. Create experiment dir: `experiments/exp_NNN_short_name/` with README.
2. Copy `configs/model_exp_template.yaml` → `configs/exp_NNN_short_name.yaml`; set paths to `~/Datasets` and `~/Models/checkpoints/exp_NNN_short_name`.
3. Create run manifest after run: `experiments/exp_NNN_short_name/manifest_run_YYYYMMDD_HHMM.json`.
4. If the model is used by the Brain at inference time: add an entry to `data/manifests/registry.json` (or a manifest under `~/Models`) with path to exported weights under `~/Models`.

**Stability > novelty. No large artifacts in repo.**
