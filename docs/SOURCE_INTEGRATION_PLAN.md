# Source Integration and Acquisition Plan (KmiDi)

Historical note
- This plan includes Tauri-era repo descriptions and should not be treated as current architecture authority.
- Use it for source-acquisition context only.
- When it conflicts with the current repo architecture, follow `docs/ARCHITECTURE.md`, `docs/REPO_MODULE_MAP.md`, and `AGENTS.md`.

**Status:** Plan only. No implementation, no file edits beyond this doc, no downloads.  
**Source material:** The block was provided as placeholder `[paste the full source block here]`; the 15 source items below are taken from the task list. Anything not verified from repo or from pasted source is marked UNKNOWN, UNVERIFIED, or ASSUMPTION.

KmiDi vocabulary preserved: KmiDi, KellyBrain, Brain–Body, kellyharness, Archive of Integrated Wounds, witness, residual, void, Autotroph.

---

# 1. Repo scan summary

## Relevant directories

| Directory | Role |
|-----------|------|
| **src/** | React (Vite) UI; intent builder, emotion wheel. No direct external-research refs. |
| **engine/intent_ir/** | Tauri 2 + Rust; FFI to KellyFFI/KellyCore. |
| **music_brain/** | FastAPI, `/generate`, session, generative; **music_brain/jepa/** (audio_jepa, chord_jepa, trainer); emotion, groove, harmony, penta_core, engine_api. |
| **shared_schemas/** | CompleteSongIntentRequest.json → sync_entities.py → TS/Rust/Python. |
| **config/** | jepa_training.yaml, models.yaml, emotion/harmony/groove/dynamics, dataset_manifest_schema.json (DatasetManifest 2.0). |
| **configs/** | Storage/experiments (DATA_AND_TRAINING). |
| **scripts/** | make_jepa_manifest.py (Lhotse), train_jepa_local.py, launch_jepa_sagemaker.py, prepare_datasets.py, download_all_datasets.sh, download_musicnet_aria2.sh, build_manifests.py, package_dataset.py. No Label Studio→Lhotse; no generic external-source manifest. |
| **experiments/** | exp_002_wavjepa_emotion (WavJEPA emotion), perch_remi_pipeline (REMI-BPE), exp_001_ump_jepa. Lightweight code; no weights in repo. |
| **docs/** | DEVELOPMENT.md, ENVIRONMENT.md, DATA_AND_TRAINING.md, SAGEMAKER_SETUP.md; WAVJEPA_*, REMI_BPE_TOKENIZATION.md, mt3-transcription-baseline.md, apple-silicon-low-latency.md, research/, specs/. |
| **engine/, include/, rt_harness/** | C++ engine; BUILD_RT_HARNESS (headless RT). KellyBrain documented in DEVELOPMENT.md; paths may be aspirational. |
| **manifests/** | JEPA Lhotse output (recordings/supervisions/cuts JSONL). |

## Relevant existing modules or docs

- **Open song generation:** music_brain session/generative and `/generate`; no product name “SongGeneration v2” in repo.
- **MidiTok / tokenization:** docs/REMI_BPE_TOKENIZATION.md, docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md, experiments/perch_remi_pipeline (remi_bpe_demo.py, make_fixtures.py). Autotroph: no refs.
- **KellyBrain / Brain–Body / kellyharness:** KellyBrain in DEVELOPMENT.md and AGENTS.md; rt_harness exists (headless RT). Brain–Body and kellyharness: no refs.
- **JEPA:** config/jepa_training.yaml, scripts/make_jepa_manifest.py, train_jepa_local.py, launch_jepa_sagemaker.py, music_brain/jepa/. Lhotse in make_jepa_manifest and pyproject [jepa].
- **WavJEPA / transcriber:** docs/WAVJEPA_KMIDI_TASKS.md, WAVJEPA_LATENT_PIPELINE.md, wavjepa_emotion_protocol.md; experiments/exp_002_wavjepa_emotion; docs/mt3-transcription-baseline.md. MoE: no refs.
- **MIDI / emotion corpora:** MAESTRO-like in make_jepa_manifest; CREMA-D, RAVDESS in exp_002 and DATA_AND_TRAINING.
- **Apple silicon:** docs/apple-silicon-low-latency.md (Audio Workgroup, QoS, buffer sweep, ANE vs GPU). XPC, UDS, shared rings: no refs.
- **Label Studio, Core ML:** No refs.

## Likely integration points

- **Open song generation:** docs (overview); music_brain session/generative and config if APIs or formats align (after verification).
- **MidiTok / Autotroph:** experiments/perch_remi_pipeline and docs/REMI_BPE_TOKENIZATION.md; new experiment or config for “Autotroph tokenizers” once source is clear.
- **Creative seed (Bbm / void):** docs or intent/creative-seed config if framed as architectural; otherwise research reference.
- **Brain–Body / kellyharness:** docs and possibly rt_harness or new harness concept under scripts/ or experiments/; depends on source definition.
- **JEPA / WavJEPA / transcriber / MoE:** music_brain/jepa, docs/WAVJEPA_LATENT_PIPELINE.md, experiments/exp_002; transcriber probe and MoE bridge as experiments or design docs first.
- **WavJEPA pretrained encoder:** Hugging Face (e.g. labhamlet/wavjepa-base) referenced in WAVJEPA_KMIDI_TASKS; checkpoint to KELLY_MODELS_PATH or env-directed path.
- **MIDI datasets / emotion corpora:** config and scripts/prepare_datasets, make_jepa_manifest; manifest entries; KMIDI_DATASETS_PATH.
- **Apple silicon microbench / XPC-UDS-rings:** docs/apple-silicon-low-latency.md; new microbench script or experiment; IPC choice = design doc then optional implementation.
- **Label Studio→Lhotse:** new script under scripts/; Lhotse format already in make_jepa_manifest.py.
- **Emotion corpora (license-safe):** config + manifest; reuse CREMA-D/RAVDESS path pattern.
- **AI tagging / benchmarks:** docs and optional config or manifest for standards; evaluation in experiments or research.
- **Core ML export/quantization:** new doc and optional experiment; no current code.

## Likely non-goals (current repo shape)

- Implementing SongGeneration v2 as a product without a verified spec or API.
- Adding Autotroph code until “Autotroph tokenizers” and MidiTok recipes are defined in source.
- Training WavJEPA encoder in-repo (use frozen only; per WAVJEPA_LATENT_PIPELINE).
- Committing large weights, datasets, or binaries to git.
- Implementing XPC/UDS/shared rings before a documented choice and human approval.
- Core ML pipeline before encoder export path and license are verified.

---

# 2. Structured briefings for each source item

*Source text was not pasted; only item titles were provided. “Source text / data” is UNKNOWN or derived from item name and repo only.*

---

## SongGeneration v2 open release overview

- **Task:** Integrate overview of “SongGeneration v2 open release” into KmiDi context.
- **Secondary tasks:** Align with `/generate` and session/generative if APIs or formats match; document relationship.
- **Context:** Open song generation; music_brain API and session/generator.
- **Known facts:** Repo has `/generate` and intent-driven generation; no “SongGeneration v2” product name or release doc in repo.
- **Unknowns:** What “SongGeneration v2” is; canonical URL; license; API/schema; whether it is a model, service, or spec.
- **Assumptions:** None beyond that it is an “open release” (UNVERIFIED).
- **Constraints:** UNKNOWN.
- **Ambiguities:** Whether “open release” means open weights, open API, or open spec; whether it is third-party or internal.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Doc in docs/ (e.g. docs/research/sources/) summarizing overview and link; optional config/manifest entry after verification.

---

## MidiTok recipes for Autotroph tokenizers

- **Task:** Capture MidiTok recipes used for “Autotroph tokenizers” and plan integration.
- **Secondary tasks:** Align with existing REMI-BPE pipeline (perch_remi_pipeline, REMI_BPE_TOKENIZATION.md); add Autotroph-specific recipes if defined.
- **Context:** Symbolic tokenization; MidiTok; Autotroph (KmiDi term; no repo refs).
- **Known facts:** Repo uses MidiTok REMI+BPE; experiments/perch_remi_pipeline; docs/REMI_BPE_TOKENIZATION.md; Autotroph not referenced in repo.
- **Unknowns:** What “Autotroph tokenizers” are; which MidiTok recipes; whether Autotroph is a config variant, vocabulary, or separate codebase.
- **Assumptions:** None.
- **Constraints:** UNKNOWN.
- **Ambiguities:** Relationship between “recipes” and “Autotroph”; whether recipes are public or internal.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Doc (recipes, vocab, config); optional config YAML or experiment under experiments/; manifest if external assets.

---

## Creative seed: Bbm as the shape of the void

- **Task:** Integrate the concept “creative seed: Bbm as the shape of the void” as context (and optionally as architecture).
- **Secondary tasks:** Map to intent/creative-seed or emotion if framed as technical; otherwise keep as research reference.
- **Context:** Creative seed ingestion; KmiDi vocabulary (void); possibly key/affect in intent.
- **Known facts:** “Void” is preserved KmiDi vocabulary; shared_schemas and Intent have key, structure, emotion; no “creative seed” or “Bbm” semantic in repo.
- **Unknowns:** Whether “Bbm” is key (B flat minor), a literal seed value, or metaphor; whether “shape of the void” is a data shape, latent space, or poetic; whether it implies a config field or only documentation.
- **Assumptions:** None.
- **Constraints:** Treat emotional/creative language as potentially architectural per instructions.
- **Ambiguities:** Literal vs metaphorical; configurable vs reference-only.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Doc (briefing + interpretation); optional config or intent extension only if source later specifies a technical contract.

---

## Brain–Body test harness concept for KmiDi

- **Task:** Plan integration of “Brain–Body test harness” concept.
- **Secondary tasks:** Relate to KellyBrain and existing rt_harness; distinguish from kellyharness if they differ.
- **Context:** KellyBrain / Brain–Body / kellyharness; rt_harness (headless RT) exists.
- **Known facts:** DEVELOPMENT.md and AGENTS.md reference KellyBrain; rt_harness has BUILD_RT_HARNESS; no Brain–Body or kellyharness refs in repo.
- **Unknowns:** Definition of Brain–Body; whether kellyharness is the same as “Brain–Body test harness”; what is tested (latency, correctness, DAW safety); where it runs (C++, Python, both).
- **Assumptions:** None.
- **Constraints:** Plugin safety and DAW stability as technical constraints if stated in source.
- **Ambiguities:** Brain–Body vs kellyharness vs rt_harness; scope of “test harness.”
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Design doc in docs/; optional script or target under scripts/ or experiments/ after definition; manifest if external harness code.

---

## JEPA-to-MIDI transcriber probe implementation

- **Task:** Plan implementation of a “JEPA-to-MIDI transcriber probe” (latent → MIDI/symbolic).
- **Secondary tasks:** Tie to WavJEPA latent pipeline and MT3 baseline; document placement and determinism.
- **Context:** JEPA / WavJEPA / transcriber probes; docs/WAVJEPA_LATENT_PIPELINE.md, mt3-transcription-baseline.md; music_brain/jepa.
- **Known facts:** WAVJEPA_LATENT_PIPELINE: Audio → frozen WavJEPA → latents → optional linear map → token head/conditioning; MT3 as token-decoder baseline; no “probe” implementation in repo.
- **Unknowns:** Exact probe architecture (linear head, small MLP, MT3 adapter); training data; evaluation protocol; whether “probe” is trainable head on frozen JEPA only.
- **Assumptions:** ASSUMPTION: probe = lightweight trainable head on frozen JEPA encoder (not re-training encoder).
- **Constraints:** Determinism and “no encoder training” per WAVJEPA_LATENT_PIPELINE.
- **Ambiguities:** Probe vs full transcriber; which JEPA (audio-JEPA, chord-JEPA, WavJEPA).
- **Source text / data:** UNKNOWN (not pasted); repo design doc only.
- **Output format:** Design doc update or new doc; experiment under experiments/; config for data and eval.

---

## Prototype a frozen JEPA encoder + MoE bridge

- **Task:** Plan a prototype “frozen JEPA encoder + MoE bridge.”
- **Secondary tasks:** Relate to WavJEPA latent pipeline; clarify MoE role (routing, capacity, conditioning).
- **Context:** JEPA / WavJEPA; MoE; WAVJEPA_LATENT_PIPELINE (frozen encoder only).
- **Known facts:** Repo has frozen WavJEPA design; music_brain/jepa (audio/chord); no MoE references in repo.
- **Unknowns:** What “MoE bridge” is (router, expert layout, interface); whether MoE is trained or fixed; input/output of bridge; which JEPA encoder (WavJEPA vs in-repo JEPA).
- **Assumptions:** None.
- **Constraints:** Encoder must stay frozen per existing design.
- **Ambiguities:** MoE as latent-space bridge vs token-side; training vs inference-only.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Design doc; experiment under experiments/; config and manifest if checkpoints or code are external.

---

## WavJEPA pretrained waveform encoder released

- **Task:** Integrate and plan acquisition of “WavJEPA pretrained waveform encoder” release.
- **Secondary tasks:** Use as frozen front-end per WAVJEPA_LATENT_PIPELINE; document checkpoint location and license.
- **Context:** WavJEPA; exp_002_wavjepa_emotion references labhamlet/wavjepa-base (Hugging Face).
- **Known facts:** WAVJEPA_KMIDI_TASKS and exp_002 reference loading WavJEPA (e.g. Hugging Face); design uses frozen encoder only.
- **Unknowns:** Official release URL; license; exact artifact (HF model id, file list, checksums); whether “released” is public or gated.
- **Assumptions:** ASSUMPTION: “released” implies at least one publicly loadable checkpoint (UNVERIFIED until source or primary source checked).
- **Constraints:** Use only as frozen feature extractor (no training in KmiDi).
- **Ambiguities:** Which variant (base/large); compatibility with 16 kHz, 2 s chunks in repo docs.
- **Source text / data:** UNKNOWN (not pasted); repo references to HF and WavJEPA only.
- **Output format:** Manifest entry (URL, license, checksum policy); doc; download to KELLY_MODELS_PATH or env path; config for model id.

---

## New MIDI datasets & models

- **Task:** Integrate “new MIDI datasets & models” into discovery and acquisition plan.
- **Secondary tasks:** Add to config/manifest; align with make_jepa_manifest and prepare_datasets; respect DATA LAW.
- **Context:** MIDI datasets; aligned audio-MIDI; config/dataset_manifest_schema.json; scripts/make_jepa_manifest.py.
- **Known facts:** Repo uses MAESTRO-like audio+MIDI, Lhotse manifests; DATA_AND_TRAINING, KMIDI_DATASETS_PATH; DatasetManifest 2.0.
- **Unknowns:** Which datasets and models; URLs; licenses; size; alignment format (audio-MIDI pairs, MIDI-only).
- **Assumptions:** None.
- **Constraints:** No large data in repo; use env-directed paths.
- **Ambiguities:** “New” vs existing (MAESTRO, groove_midi); whether “models” are tokenizers, checkpoints, or both.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Manifest entries; docs; config and scripts for download/prep; storage outside git.

---

## Microbench for one-way latency on Apple silicon

- **Task:** Plan a microbench for one-way latency on Apple silicon.
- **Secondary tasks:** Tie to apple-silicon-low-latency.md; inform buffer and QoS choices.
- **Context:** Apple silicon latency; docs/apple-silicon-low-latency.md (buffer sweep, Audio Workgroup, QoS, ANE vs GPU).
- **Known facts:** Doc exists; targets sub-10 ms; 64/128 samples at 48 kHz; Instruments/Xcode; no dedicated microbench script in repo.
- **Unknowns:** Exact metric (“one-way” = I/O only, or encode, or full loop); target process (plugin, standalone, Python); whether microbench is script, Xcode template, or doc procedure.
- **Assumptions:** None.
- **Constraints:** Realtime safety (no alloc/locks in callback) per doc.
- **Ambiguities:** Scope of “one-way”; which component is measured.
- **Source text / data:** UNKNOWN (not pasted); repo doc only.
- **Output format:** Doc (procedure or spec); optional script in scripts/ or experiments/; no new binaries in repo.

---

## Choosing between XPC, UDS, and shared rings

- **Task:** Plan how to choose and document IPC: XPC vs UDS vs shared rings.
- **Secondary tasks:** Relate to local-service boundaries and DAW/plugin safety; document decision.
- **Context:** Apple silicon / IPC; local-service boundaries (technical constraint).
- **Known facts:** No XPC, UDS, or shared rings refs in repo; apple-silicon-low-latency.md does not specify IPC.
- **Unknowns:** Use case (which processes communicate); latency and throughput requirements; platform (macOS only or cross-platform); whether “shared rings” is Apple-specific or generic.
- **Assumptions:** None.
- **Constraints:** Human approval before implementation; plugin/DAW stability.
- **Ambiguities:** Which boundary (e.g. UI ↔ engine, Python ↔ C++) the choice applies to.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Design doc (options, tradeoffs, decision); no code until decision and approval.

---

## Convert Label Studio exports to Lhotse manifests

- **Task:** Plan conversion from Label Studio exports to Lhotse manifests.
- **Secondary tasks:** Reuse Lhotse format from make_jepa_manifest.py; support audio/supervision/cuts.
- **Context:** Label Studio → Lhotse; scripts/make_jepa_manifest.py produces RecordingSet/SupervisionSet/CutSet.
- **Known facts:** make_jepa_manifest.py outputs Lhotse JSONL to manifests/; no Label Studio in repo.
- **Unknowns:** Label Studio export schema (JSON, CSV); mapping of labels to Lhotse supervisions; whether audio paths are local or URIs; which task (transcription, emotion, segments).
- **Assumptions:** None.
- **Constraints:** UNKNOWN.
- **Ambiguities:** Export format version; handling of multi-task labels.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Script under scripts/ (e.g. label_studio_to_lhotse.py); doc; optional config for paths and field mapping.

---

## Small, license-safe emotion-labeled music corpora

- **Task:** Identify and plan acquisition of small, license-safe emotion-labeled music corpora.
- **Secondary tasks:** Integrate with emotion config and WavJEPA emotion experiment; manifest and config.
- **Context:** Emotion corpora; exp_002 (CREMA-D, RAVDESS); config/emotion_*.yaml; DATA_AND_TRAINING.
- **Known facts:** CREMA-D and RAVDESS used in exp_002; speech/emotion; repo mentions emotions/ravdess, emotions/cremad under dataset root; “music” corpora not enumerated.
- **Unknowns:** Which corpora are “music” (not speech); licenses; size; label schema; public URLs.
- **Assumptions:** None.
- **Constraints:** License-safe only; small preferred; no repo storage of data.
- **Ambiguities:** “Music” vs “speech with emotion”; whether existing CREMA-D/RAVDESS suffice or additional corpora are required.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Manifest and doc; config paths; download/prep script; storage via KMIDI_DATASETS_PATH.

---

## Apple Music's new AI-content tagging rules

- **Task:** Integrate Apple Music’s AI-content tagging rules as context (compliance / provenance).
- **Secondary tasks:** Map to evaluation/provenance/transparency tags; document for export or distribution.
- **Context:** Evaluation / provenance / transparency tags; possible future distribution or store submission.
- **Known facts:** No Apple Music or AI-tagging refs in repo.
- **Unknowns:** Exact rules; URL or doc; whether they apply to generated music, training data, or both; required metadata format.
- **Assumptions:** None.
- **Constraints:** UNKNOWN.
- **Ambiguities:** Scope (Apple only vs industry); enforcement mechanism.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Doc (summary + link); optional config or schema for tags if rules specify format; no implementation until verified.

---

## New standards & music-AI benchmarks

- **Task:** Integrate “new standards & music-AI benchmarks” into docs and optional manifest.
- **Secondary tasks:** Relate to evaluation harness; avoid duplicate effort.
- **Context:** Evaluation / benchmarks; KellyFFIBenchmark, wavjepa baselines, research guardrails; no single benchmark manifest.
- **Known facts:** Scattered eval refs; no central benchmark list in repo.
- **Unknowns:** Which standards and benchmarks; URLs; licenses; task coverage (transcription, generation, emotion); whether datasets are public.
- **Assumptions:** None.
- **Constraints:** UNKNOWN.
- **Ambiguities:** “New” vs existing; adoption vs reference only.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Doc (list + links); optional manifest for benchmark assets; config or experiment only if we run them.

---

## Exporting and quantizing a JEPA encoder for Core ML

- **Task:** Plan export and quantization of a JEPA encoder for Core ML.
- **Secondary tasks:** Document encoder source (WavJEPA vs in-repo JEPA); quantization levels; runtime path.
- **Context:** Core ML export/quantization; WavJEPA frozen encoder; no Core ML refs in repo.
- **Known facts:** Intent “quantization” in repo is rhythmic (swing); no ML model quantization or Core ML code.
- **Unknowns:** Which encoder (WavJEPA, audio_jepa, chord_jepa); export path (PyTorch → Core ML, ONNX intermediate); quantization (weights, activations, ANE); license of encoder and tools.
- **Assumptions:** None.
- **Constraints:** Encoder remains frozen (no training); plugin/DAW and Apple silicon context.
- **Ambiguities:** ANE vs GPU; whether “JEPA encoder” is WavJEPA or in-repo JEPA.
- **Source text / data:** UNKNOWN (not pasted).
- **Output format:** Design doc; optional experiment (export script, quant config); manifest for exported artifact; storage outside repo.

---

# 3. Source inventory / verification matrix

| item | category | status | downloadable artifacts available? | artifact classes | likely canonical source(s) | likely license or usage constraints | manual review needed | estimated storage impact | KmiDi integration relevance | notes / ambiguity |
|------|----------|--------|-----------------------------------|------------------|----------------------------|--------------------------------------|------------------------|---------------------------|-----------------------------|--------------------|
| SongGeneration v2 open release | product/spec | UNVERIFIED | UNKNOWN | docs, possibly API/spec | UNKNOWN | UNKNOWN | YES | UNKNOWN | open song generation | No pasted source; may be conceptual or third-party. |
| MidiTok recipes for Autotroph | config/recipes | PARTIALLY VERIFIED | YES (MidiTok pypi) | repo, docs, config | MidiTok: PyPI/GitHub; Autotroph: UNKNOWN | MidiTok: UNKNOWN until checked; Autotroph: UNKNOWN | YES for Autotroph | small (recipes) | symbolic tokenization / Autotroph | “Recipes” may be internal; Autotroph not in repo. |
| Creative seed Bbm/void | concept | UNVERIFIED | NO (conceptual) | other (doc) | UNKNOWN | N/A | NO | none | creative seed ingestion | Conceptual; may inform config or doc only. |
| Brain–Body test harness | concept/tool | UNVERIFIED | UNKNOWN | repo, docs, or other | UNKNOWN | UNKNOWN | YES | UNKNOWN | KellyBrain / Brain–Body / kellyharness | kellyharness not in repo; may be internal. |
| JEPA-to-MIDI transcriber probe | implementation | PARTIALLY VERIFIED | UNKNOWN | code, config, docs | Repo design (WAVJEPA_LATENT_PIPELINE, MT3); training data UNKNOWN | MT3: Apache-2.0 / GPL forks; probe: UNKNOWN | YES if external data | medium if checkpoints | JEPA / transcriber | Probe not implemented; “implementation” = plan. |
| Frozen JEPA + MoE bridge | prototype | UNVERIFIED | UNKNOWN | code, weights, docs | UNKNOWN | UNKNOWN | YES | UNKNOWN (MoE can be large) | JEPA / MoE bridge | MoE not in repo; no primary source. |
| WavJEPA pretrained encoder | weights/model | PARTIALLY VERIFIED | YES (e.g. HF) | weights, model card, docs | Hugging Face (e.g. labhamlet/wavjepa-base); official WavJEPA: UNKNOWN | UNKNOWN until primary source | YES (license) | large | WavJEPA / latent pipeline | Repo references HF; official release URL unverified. |
| New MIDI datasets & models | datasets/models | UNVERIFIED | UNKNOWN | dataset, weights, docs | UNKNOWN | UNKNOWN | YES | large (datasets) | MIDI datasets / emotion | “New” unspecified; no URLs. |
| Apple silicon one-way latency microbench | benchmark | PARTIALLY VERIFIED | NO (procedure) | docs, possibly script | Repo: apple-silicon-low-latency.md | N/A | NO | none | Apple silicon latency | Procedure in doc; “microbench” may be script or doc. |
| XPC vs UDS vs shared rings | design choice | UNVERIFIED | N/A | docs | UNKNOWN | N/A | YES (decision) | none | Apple silicon / IPC | Conceptual; no implementation until decision. |
| Label Studio → Lhotse | pipeline/script | UNVERIFIED | YES (Label Studio export format) | docs, code | Label Studio docs; repo: make_jepa_manifest Lhotse format | Label Studio: UNKNOWN; Lhotse: Apache-2.0 | YES (export schema) | small | Label Studio → Lhotse | Conversion not in repo; export schema to verify. |
| Emotion-labeled music corpora | dataset | PARTIALLY VERIFIED | YES (e.g. CREMA-D, RAVDESS) | dataset, docs | CREMA-D/RAVDESS (in repo docs); “music” corpora: UNKNOWN | Dataset-specific; check each | YES (license) | medium | emotion corpora | “Music” vs speech ambiguity. |
| Apple Music AI-content tagging | policy/spec | UNVERIFIED | UNKNOWN | docs | UNKNOWN | UNKNOWN | YES | none | evaluation / provenance | No ref in repo; need primary source. |
| New standards & music-AI benchmarks | standards/benchmarks | UNVERIFIED | UNKNOWN | docs, benchmark packages, datasets | UNKNOWN | UNKNOWN | YES | variable | evaluation / benchmarks | “New” unspecified. |
| JEPA encoder Core ML export | pipeline/weights | UNVERIFIED | UNKNOWN (encoder yes; export tooling varies) | weights, export script, quant config, docs | WavJEPA or music_brain/jepa; Core ML tools: Apple | Encoder license + Apple tooling | YES | large (encoder) + export artifacts | Core ML export / quantization | Which encoder and toolchain unverified. |

---

# 4. Repo-aware integration plan (by domain)

## Open song generation

- **Proposed place:** docs/research/sources/ or docs/external_sources/ (overview); music_brain and config only if API/schema verified.
- **Docs:** One briefing + link to canonical “SongGeneration v2” if found.
- **Manifests:** Only if there are downloadable assets (e.g. API spec, SDK).
- **Configs:** Optional API base URL or feature flags after verification.
- **Experiments:** None until spec exists.
- **Research references only:** Yes until verified.
- **Dependencies:** Verification of product/spec and license.
- **Human decisions:** Whether to adopt or only reference.
- **Phase priority:** Low (after verification).
- **Non-goals for now:** Implementing a full SongGeneration v2 client or product.

## Symbolic tokenization / MidiTok / Autotroph

- **Proposed place:** docs/ (REMI_BPE_TOKENIZATION.md, new Autotroph brief); config/ (tokenizer configs); experiments/perch_remi_pipeline and optional experiments/autotroph_*.
- **Docs:** MidiTok recipes; Autotroph definition and recipe list (when known).
- **Manifests:** External vocab or recipe files if any.
- **Configs:** YAML for tokenizer params and dataset paths (existing pattern).
- **Experiments:** Autotroph recipe runs or comparisons once defined.
- **Research references only:** Autotroph until defined.
- **Dependencies:** Clarification of “Autotroph tokenizers” and recipe source.
- **Human decisions:** Which recipes to adopt; vocab size and schema.
- **Phase priority:** Medium (MidiTok already in use); Autotroph when defined.
- **Non-goals for now:** New tokenizer training pipeline until recipes are verified.

## Creative seed ingestion

- **Proposed place:** docs/ (briefing); optional shared_schemas or config if “creative seed” gets a technical contract.
- **Docs:** “Bbm as shape of the void” briefing; interpretation (literal vs architectural).
- **Manifests:** None unless seed is an asset.
- **Configs:** Optional creative-seed or key/affect default if specified by source.
- **Experiments:** None for now.
- **Research references only:** Yes unless source specifies a data shape or config.
- **Dependencies:** Source clarification.
- **Human decisions:** Whether to add config fields or keep reference-only.
- **Phase priority:** Low.
- **Non-goals for now:** Implementing seed semantics without a clear contract.

## KellyBrain / Brain–Body / kellyharness

- **Proposed place:** docs/ (Brain–Body and kellyharness design); scripts/ or experiments/ for harness if it is a script/target; rt_harness remains separate (headless RT).
- **Docs:** Definition of Brain–Body and kellyharness; relationship to KellyBrain and rt_harness.
- **Manifests:** If harness code or fixtures are external.
- **Configs:** Harness config (paths, targets) if needed.
- **Experiments:** Optional harness run or smoke test.
- **Research references only:** Until definition is clear.
- **Dependencies:** Source definition; plugin/DAW safety constraints.
- **Human decisions:** Whether kellyharness is same as Brain–Body harness; where it lives.
- **Phase priority:** Medium after definition.
- **Non-goals for now:** Implementing a new harness until concept is defined.

## JEPA / WavJEPA / transcriber probes / MoE bridge

- **Proposed place:** music_brain/jepa (existing); docs (WAVJEPA_LATENT_PIPELINE, new transcriber/MoE docs); experiments/ (transcriber probe, MoE bridge prototype).
- **Docs:** Transcriber probe design; MoE bridge design; update WAVJEPA_LATENT_PIPELINE if needed.
- **Manifests:** Checkpoints (WavJEPA, probe, MoE if released); Lhotse for training data.
- **Configs:** jepa_training.yaml; new probe/MoE configs.
- **Experiments:** exp_002 pattern; new exp for probe and MoE.
- **Research references only:** MoE until design is fixed.
- **Dependencies:** WavJEPA checkpoint verification; probe/MoE design from source.
- **Human decisions:** Which JEPA encoder for probe; MoE scope.
- **Phase priority:** High for WavJEPA and probe (align with existing design); MoE after design.
- **Non-goals for now:** Training JEPA encoder; implementing MoE without a design.

## MIDI datasets / aligned audio-MIDI / emotion corpora

- **Proposed place:** config/ (dataset entries); scripts/ (prepare_datasets, make_jepa_manifest, optional download); manifests/ (Lhotse); KMIDI_DATASETS_PATH.
- **Docs:** Dataset list; license and provenance per dataset.
- **Manifests:** config/dataset_manifest_schema.json and Lhotse; optional source_manifest for external refs.
- **Configs:** dataset_root, paths per dataset (existing pattern).
- **Experiments:** Use in exp_002 and future experiments.
- **Research references only:** “New” datasets until identified.
- **Dependencies:** Dataset URLs and licenses; emotion “music” corpus clarification.
- **Human decisions:** Which corpora to adopt; license sign-off.
- **Phase priority:** High for existing CREMA-D/RAVDESS path; medium for “new” MIDI/music-emotion.
- **Non-goals for now:** Adding unverified or license-unsafe datasets.

## Evaluation / provenance / transparency tags

- **Proposed place:** docs/ (benchmarks, standards, Apple Music tagging); config or manifest for benchmark list and tag schema.
- **Docs:** Apple Music AI-tagging rules; new standards and benchmarks (list + links).
- **Manifests:** Benchmark assets if we run them; tag schema if defined.
- **Configs:** Eval runs; tag field names if needed.
- **Experiments:** Benchmark runs or tag export experiments.
- **Research references only:** Until standards and tagging rules are verified.
- **Dependencies:** Primary sources for Apple Music and benchmarks.
- **Human decisions:** Which benchmarks to run; whether to emit tags.
- **Phase priority:** Medium (docs first); implementation after verification.
- **Non-goals for now:** Full compliance implementation without verified rules.

## Apple silicon latency / IPC

- **Proposed place:** docs/apple-silicon-low-latency.md; optional scripts/ or experiments/ for microbench; new doc for IPC choice.
- **Docs:** Microbench procedure or spec; XPC/UDS/shared rings comparison and decision.
- **Manifests:** None.
- **Configs:** Buffer size, QoS, or IPC choice if configurable.
- **Experiments:** Microbench run (script or documented procedure).
- **Research references only:** IPC until decision.
- **Dependencies:** Human decision on IPC; no automatic implementation.
- **Human decisions:** IPC choice; microbench scope.
- **Phase priority:** Medium (microbench doc/script); IPC after decision.
- **Non-goals for now:** Implementing XPC/UDS/rings before documented decision.

## Label Studio → Lhotse

- **Proposed place:** scripts/ (e.g. label_studio_to_lhotse.py); docs (mapping, export format).
- **Docs:** Label Studio export format; field mapping to Lhotse.
- **Manifests:** Output = Lhotse JSONL (same as make_jepa_manifest).
- **Configs:** Input/output paths; field mapping.
- **Experiments:** Optional run on sample export.
- **Research references only:** No.
- **Dependencies:** Label Studio export schema verification.
- **Human decisions:** Which tasks and label types to support.
- **Phase priority:** Medium.
- **Non-goals for now:** Supporting every Label Studio task; only those needed for Lhotse.

## Core ML export / quantization

- **Proposed place:** docs/ (design); experiments/ (export/quant script); KELLY_MODELS_PATH or env for exported artifact.
- **Docs:** Encoder choice; export path (PyTorch → Core ML / ONNX); quantization levels; ANE vs GPU.
- **Manifests:** Encoder source; exported .mlpackage or quantized artifact (outside repo).
- **Configs:** Quantization config; model id.
- **Experiments:** Export and quant run; latency test.
- **Research references only:** Until encoder and toolchain are chosen.
- **Dependencies:** Encoder license; Apple tooling license.
- **Human decisions:** Which encoder; quantization aggressiveness; ANE targeting.
- **Phase priority:** Lower (after WavJEPA and probe).
- **Non-goals for now:** Shipping Core ML in plugin until path is verified.

---

# 5. Download / acquisition plan

## Staged plan

### Stage 1 — Verify first (no download)

- **Artifacts to verify:**  
  - SongGeneration v2: existence, URL, license.  
  - WavJEPA: official or HF release URL, license, checksum policy.  
  - MidiTok: PyPI/GitHub license; Autotroph recipe source.  
  - Label Studio: export schema (JSON/CSV).  
  - Apple Music AI tagging: official rules URL.  
  - New MIDI datasets & models: names, URLs, licenses.  
  - Emotion music corpora: list, licenses (CREMA-D/RAVDESS already referenced).  
  - Standards & benchmarks: names, URLs, licenses.  
  - JEPA encoder for Core ML: which encoder, export tooling, license.

### Stage 2 — Fetch first after approval

- **Order:**  
  1. Docs and small specs (no large binaries).  
  2. MidiTok (already used; ensure version and license).  
  3. WavJEPA checkpoint (after license and path approval) to KELLY_MODELS_PATH or env path.  
  4. Label Studio export sample (if available) for converter development.  
  5. License-safe emotion corpora (after license check).  
  6. Benchmark metadata/specs (not full datasets yet).  
  7. Code repos (clone to env path; no heavy blobs in repo).

### Stage 3 — Large, gated, license-sensitive, or train-from-scratch

- **Likely large:** WavJEPA weights; “new” MIDI datasets; Core ML exported model.  
- **Likely gated:** Any “released” item that requires sign-up or approval (mark in manifest).  
- **License-sensitive:** All datasets and weights; Apple Music rules; third-party code.  
- **Train-from-scratch:** Probe head; MoE bridge (if no released checkpoint). Do not download “probe weights” unless a public release exists.

### Checksum / hash strategy

- Add optional `checksum` / `hashAlgorithm` to source manifest (align with config/dataset_manifest_schema.json where applicable).  
- For weights and large datasets: record SHA-256 or provider’s checksum when available; verify on first fetch.  
- Store checksums in manifest (in repo); store blobs outside repo.

### Provenance recording strategy

- Each manifest entry: source URL, date obtained, version/tag if applicable.  
- Docs: cite primary source and date.  
- Per DATA_AND_TRAINING and run manifests for training: record dataset and checkpoint provenance.

### License recording strategy

- Required field in source manifest: `license` or `usage_constraints`; UNKNOWN until verified.  
- Doc per source: license summary and link to full text.  
- No automated download for items with UNKNOWN or restrictive license until human approval.

### Storage strategy outside git

- All weights, datasets, and large binaries: KELLY_MODELS_PATH, KMIDI_DATASETS_PATH, or EXTERNAL_SOURCES_ROOT (env).  
- Optional: KMIDI_DATA_ROOT (external SSD) for large assets.  
- Repo: manifests, configs, small fixtures, docs only.  
- .gitignore: any local clone or download dir under repo if needed.

### Manifest strategy

- Single source manifest (e.g. config/source_manifest.yaml or docs/source_manifest.yaml) with: item name, category, primary source URL, license, downloadable (yes/no/unknown), checksum policy, storage path (env var name), integration domain.  
- Dataset-specific: keep using config/dataset_manifest_schema.json and Lhotse JSONL.  
- Add “manual review” and “verified date” fields.

### Red flags that block automatic download

- License UNKNOWN or restrictive.  
- Primary source not verified (no URL or broken).  
- Gated or sign-up required (manual only).  
- Size above threshold without explicit approval.  
- Conflicting license with existing assets (e.g. GPL vs Apache).

---

## By artifact type

- **Code repos:** Clone to env path; do not put in repo; add to manifest with commit/tag and license.  
- **Papers:** Doc with citation and link; no full-text in repo unless license and size allow.  
- **Model cards:** Doc or manifest entry; link to primary.  
- **Weights / checkpoints:** Manifest entry; download to KELLY_MODELS_PATH or env path; checksum; license required.  
- **Datasets:** Manifest + Lhotse or DatasetManifest; KMIDI_DATASETS_PATH; license and checksum.  
- **Example assets:** Small only in repo (e.g. tests/fixtures); else env path and manifest.  
- **Benchmark packages:** Verify license; fetch to env path; manifest.  
- **Metadata specs / docs:** Doc in repo; link to canonical spec.

---

# 6. Risks, ambiguities, blockers

## Factual uncertainty

- No pasted source text: all 15 items lack verbatim context; many “known facts” are repo-only or ASSUMPTION.  
- SongGeneration v2, Autotroph recipes, Brain–Body/kellyharness, MoE bridge, “new” datasets/models, Apple Music rules, “new” standards/benchmarks: no primary source in repo.  
- WavJEPA “released” and “pretrained waveform encoder”: repo points to HF; official release and license unverified.

## License uncertainty

- Every external asset: license UNKNOWN until primary source checked.  
- MT3 forks (GPL) vs Apache-2.0: compatibility if combined with other code.  
- Apple Music and Core ML: distribution and tooling terms unknown.

## Storage / infra risk

- Large datasets and weights: must not go into git; env and external SSD discipline required.  
- setup-workspace.sh missing: dataset symlinks and external SSD workflow may need doc or script.

## Integration risk

- Autotroph and Brain–Body/kellyharness: undefined in repo; wrong placement if guessed.  
- MoE and Core ML: no current code; design and encoder choice needed first.  
- Label Studio export format: must be verified or converter may break on real exports.

## Repo pollution risk

- Adding binaries or large data to repo. Mitigation: manifest-only in repo; downloads to env paths; .gitignore.  
- Duplicate or conflicting tokenizer/config patterns if Autotroph and MidiTok recipes are added without a single convention.

## Architecture ambiguity

- “Void” and “Bbm as shape of the void”: literal (key, seed) vs metaphorical; affects whether config or only doc.  
- Brain–Body vs kellyharness vs rt_harness: one concept or several.  
- “Probe” vs “transcriber”: lightweight head vs full MT3-style model.  
- MoE “bridge”: routing layer vs multi-encoder fusion vs other.

## Human approval points

- Any download of weights or datasets (license and path).  
- IPC choice (XPC vs UDS vs shared rings).  
- Which “new” datasets and benchmarks to adopt.  
- Core ML encoder choice and quantization level.  
- Whether to implement SongGeneration v2 integration or only document.

---

# 7. Approval-ready next actions

1. **Obtain pasted source material**  
   - Replace placeholder with the full source block so briefings and inventory can be updated from primary claims and citations.

2. **First files to create (after or in parallel with verification)**  
   - `docs/research/sources/` (or `docs/external_sources/`) and one briefing file per source item (can use §2 as draft; update when source is pasted).  
   - `config/source_manifest.yaml` (or equivalent) with schema: item, category, status, primary_source, license, downloadable, storage_path, manual_review, notes.

3. **First manifests to draft**  
   - source_manifest.yaml with one row per §3 item; status = UNVERIFIED/PARTIALLY VERIFIED; license = UNKNOWN until checked; downloadable = per §3 table.

4. **First source families to verify**  
   - WavJEPA: official or HF URL, license, checksum.  
   - MidiTok: license and version.  
   - Label Studio: export schema (official docs).  
   - CREMA-D / RAVDESS: confirm license and path for “music” vs speech use.  
   - MT3: official vs fork licenses.

5. **First human decisions needed before any download**  
   - Approve storage paths (KELLY_MODELS_PATH, KMIDI_DATASETS_PATH, EXTERNAL_SOURCES_ROOT).  
   - Approve WavJEPA checkpoint download and license.  
   - Approve any dataset download (license and size).  
   - Decide whether to implement Label Studio→Lhotse script (and which tasks).

6. **Phase 0 contents**  
   - Paste and parse source material.  
   - Extract discrete items and citations.  
   - Verify primary source (URL, title, license) for each item; update status to VERIFIED/PARTIALLY VERIFIED/UNVERIFIED.  
   - Populate source_manifest.yaml (no downloads).  
   - Update briefings (§2) with any new facts from source; keep UNKNOWN where still unverified.  
   - List human decisions and red flags for Stage 2.

---

# 8. Phase 4 — Optional experimental integration (implemented)

**Scope:** Lightweight wiring of `config/source_manifest.yaml` into data-prep scripts and experiment stubs. Use only verified, license-cleared assets for any future implementation.

## 8.1 Optional source-manifest wiring

- **make_jepa_manifest.py**
  - `--source-manifest PATH` — path to source_manifest.yaml (default: repo `config/source_manifest.yaml`).
  - `--list-from-manifest` — list adopted JEPA/dataset sources (`adoption_decision=adopted` and integration_domain or artifact_classes indicating JEPA/dataset). Prints `source_item`, `proposed_storage_path`, `storage_env_var` and exits. No download or manifest generation.
  - When adoption_decision and URLs/paths are set for dataset-like items, these entries can drive `--audio-root` / `--midi-root` (user invokes script per adopted source).

- **prepare_datasets.py**
  - `--source-manifest PATH` — path to source_manifest.yaml.
  - `--list-from-manifest` — list adopted dataset sources (`adoption_decision=adopted` and artifact_classes containing `dataset` or integration_domain containing midi/emotion). Prints same fields and exits. No SSD or root required.

- **Dependencies:** Optional `pyyaml` for manifest parsing; scripts fail with a clear message if YAML is missing when using `--list-from-manifest`.

## 8.2 Experiment stubs

- **exp_003_jepa_transcriber_probe** — README only. Points to WAVJEPA_LATENT_PIPELINE, mt3-transcription-baseline, SOURCE_INTEGRATION_PLAN briefing. Implementation deferred until encoder/data adopted and probe architecture defined.
- **exp_004_moe_bridge** — README only. Points to WAVJEPA_LATENT_PIPELINE and SOURCE_INTEGRATION_PLAN briefing. Implementation deferred until “MoE bridge” is defined and assets adopted.

Both stubs state: use only verified, license-cleared assets; no code or weights in repo until design and adoption are set.

## 8.3 Future: driving paths from manifest

Once `adoption_decision: adopted` and `proposed_storage_path` (and optional per-dataset audio_root/midi_root) are set for specific sources, scripts can be extended to accept a single “adopted source” key and resolve paths from the manifest (e.g. `KMIDI_DATASETS_PATH` + `proposed_storage_path`). Not required for Phase 4.

---

*End of plan.*
