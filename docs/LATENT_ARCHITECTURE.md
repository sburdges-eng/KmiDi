# Latent Architecture: Six High-Leverage Tools

Reference for the unified multimodal strategy (KmiDi / TOTaLi). All six tools are implemented or scaffolded in this repo.

## Phase 1: Bare-Metal Execution & Hardware Control

### 1.1 Stateful KV-Cache Harness (mlmodelc)

**Purpose:** Sub-15 ms/token on M4 by keeping the KV cache in hardware-bound MLState so it is never copied back to host memory between decode steps.

**Location:** [scripts/export_llm_coreml.py](../scripts/export_llm_coreml.py), [tools/coreml_llm_runner/](../tools/coreml_llm_runner/)

**Usage:** Export with `--coreml-enable-state`, `--coreml-quantize b4w`, `--max-context-length 2048`; compile to mlmodelc; run Swift runner with state threading. See [tools/coreml_llm_runner/README.md](../tools/coreml_llm_runner/README.md) and [FULL_STACK_BUILD.md](FULL_STACK_BUILD.md).

### 1.2 MIDI-CI Microformat-to-SysEx Daemon

**Purpose:** Zero-overhead bridge from the orchestrator’s JSON microformat to physical gear via MIDI-CI Property Exchange SysEx (and optionally UMP).

**Location:** [tools/midi_ci_daemon/](../tools/midi_ci_daemon/)

**Usage:** Build with `-DBUILD_MIDI_CI_DAEMON=ON`; run the daemon and feed line-delimited JSON (`{"op":"set","target":"cutoff","val":85}`). See [tools/midi_ci_daemon/README.md](../tools/midi_ci_daemon/README.md).

---

## Phase 2: Latent Space Management & Agent Reasoning

### 2.1 Latent Canonicalization Adapter (Vector DB Hot-Swap)

**Purpose:** After encoder upgrades, avoid re-embedding the whole vector DB by fitting an orthogonal Procrustes map from ~500 anchor pairs and applying it at query time so new queries match the legacy index.

**Location:** [music_brain/penta_core/ml/canonicalize_embeddings.py](../music_brain/penta_core/ml/canonicalize_embeddings.py), [scripts/canonicalize_embeddings.py](../scripts/canonicalize_embeddings.py)

**Usage:** `fit_orthogonal_map(anchor_old, anchor_new)`; save map; at retrieval use `apply_map(query_new, R, center_old, center_new, normalize=True)`. CLI: `scripts/canonicalize_embeddings.py --old-embeddings ... --new-embeddings ... --output map.npz`.

### 2.2 APSC Multi-Stem Orchestrator Wrapper

**Purpose:** Mitigate positional bias on multi-stem prompts by permuting stem order, running multiple inferences at low temperature, and aggregating orchestration decisions via majority vote.

**Location:** [music_brain/penta_core/ml/apsc_wrapper.py](../music_brain/penta_core/ml/apsc_wrapper.py)

**Usage:** `run_apsc(prompt, stems, model_invoke, temperature=0.2, num_permutations=6)` returns a single list of `{track_id, action, params}`. Call from the orchestration layer when the request has multiple audio stems.

---

## Phase 3: Alignment Stabilization & Diagnostics

### 3.1 StructXLIP Symbolic Preprocessor

**Purpose:** Ground the LALM in temporal boundaries (e.g. “drop at bar 16, beat 3”) by extracting Audio Edge Maps (onset envelope, spectral flux) and aligning them to structural text with dedicated losses.

**Location:** [music_brain/penta_core/ml/structxlip/](../music_brain/penta_core/ml/structxlip/)

**Usage:** `extract_audio_edge_maps(y, sr)` for training data; add `global_structure_loss(edge_features, text_structure_features)` to the fine-tuning loss. See [docs/STRUCTXLIP_TRAINING.md](STRUCTXLIP_TRAINING.md).

### 3.2 PID Flow Modality Spectrometer

**Purpose:** Detect modality collapse (model “deaf” to audio) via layer-wise Partial Information Decomposition: Redundant, Text-Unique, Audio-Unique, Synergy. Acts as a “check engine light” for training.

**Location:** [music_brain/penta_core/ml/diagnostics/pid_flow.py](../music_brain/penta_core/ml/diagnostics/pid_flow.py), [scripts/run_pid_flow.py](../scripts/run_pid_flow.py)

**Usage:** `run_pid_flow_report(model, get_activations, layer_names)`; `check_modality_collapse(report)` for warnings. Run `scripts/run_pid_flow.py --dummy` to validate (use `PYTHONPATH=music_brain`).

---

## Summary

| Phase | Tool | Path |
|-------|------|------|
| 1.1 | Stateful KV-cache | scripts/export_llm_coreml.py, tools/coreml_llm_runner/ |
| 1.2 | MIDI-CI daemon | tools/midi_ci_daemon/ |
| 2.1 | Canonicalization | penta_core.ml.canonicalize_embeddings, scripts/canonicalize_embeddings.py |
| 2.2 | APSC wrapper | penta_core.ml.apsc_wrapper |
| 3.1 | StructXLIP | penta_core.ml.structxlip |
| 3.2 | PID Flow | penta_core.ml.diagnostics.pid_flow, scripts/run_pid_flow.py |

See **AGENTS.md** (“On-device and alignment tools”) and **docs/FULL_STACK_BUILD.md** for build and run details.
