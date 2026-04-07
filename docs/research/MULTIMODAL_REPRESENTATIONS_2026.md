# Multimodal Representation & JEPA Ecosystem (2026)

**Author:** Internal synthesis (KmiDi)  
**Research Date:** 2026‑03‑10  
**Status:** Planning / Design  

## Executive Summary

Recent work in multimodal representation learning (early 2026) and practical audio/symbolic tooling lines up well with KmiDi’s cross‑modal goals (audio ↔ symbolic ↔ control / affect).  
The broad trend is **refining CLIP‑style contrastive learning**, **stabilizing alignment geometry**, and **structuring shared vs modality‑specific spaces**, rather than replacing contrastive objectives outright.

For KmiDi, the most actionable pieces are:

- **Perch audio embeddings** as a ready‑to‑run JEPA‑style acoustic latent backbone.  
- **REMI‑BPE tokenization (MidiTok + Maestro‑REMI‑bpe20k)** as a concrete, interoperable symbolic target space.  
- **Lhotse + DataLad** for reproducible JEPA manifests (audio+MIDI windows with hashes).  
- **BNNS Graph + Audio Workgroups** patterns for safe real‑time inference on Apple silicon.  
- **Shared C ABI engine shim** to expose C++ DSP/latent code once to both JUCE AU and Rust/Tauri.  
- **MIDI 2.0 Property Exchange (PE) + UMP** for live affect control lanes (valence / arousal / dynamics).

These can be layered on top of evolving academic work on hierarchical alignment, canonicalization, and continual cross‑modal learning without blocking on any single paper.

## Technical Details

### 1. Alignment / Representation Learning Landscape (papers)

- **Hierarchical alignment:**  
  - **CLCR (Cross-Level Semantic Collaborative Representation, CVPR 2026)** — proposes hierarchical semantic alignment across modalities to reduce misalignment between feature levels (local vs global semantics).  
  - **DecAlign** — disentangles modality‑shared vs modality‑specific representations and aligns them using optimal transport and distribution matching.

- **Contrastive pretraining refinements:**  
  - **Dual-domain contrastive frameworks** model spectral structure in embeddings to improve large‑scale multimodal contrastive training (CLIP‑style), rather than replacing contrastive objectives.

- **Latent geometry / canonicalization:**  
  - **Canonicalizing Multimodal Contrastive Representation Learning** finds that embeddings from independently trained models (CLIP‑like) can often be related by orthogonal transforms plus mean shift.  
  - Implication: cross‑model latent interoperability (e.g., mapping Perch latents into KmiDi’s own latent space) may be realizable via linear / near‑linear maps.

- **Continual multimodal learning:**  
  - **StructAlign** uses an Equiangular Tight Frame (ETF) prior plus relation‑preserving loss to reduce feature drift and catastrophic forgetting in continual text‑video retrieval.  
  - Directly relevant to any long‑running KmiDi pretraining/fine‑tuning regime where we care about not “forgetting” core audio–symbolic correspondences.

- **Representation analysis / preferences:**  
  - **PID Flow** applies information‑theoretic decomposition to analyze how modalities contribute to learned features.  
  - **MAPLE (Modality‑Aligned Preference Learning for Embeddings)** uses multimodal LLM priors to guide embedding learning with human / model preferences.  

- **Industry multimodal models:**  
  - **Phi‑4‑reasoning‑vision‑15B** emphasizes: heavily curated data, mixed reasoning vs non‑reasoning batches, and careful curriculum — all applicable to constructing future KmiDi multimodal corpora (audio, MIDI, text prompts, affect labels).

### 1.1 User briefing update (2026-03-31)

Status: external briefing captured for planning; source verification still required before implementation claims.

- **Disentangled latent spaces are moving from theory to architecture guidance:** user-provided notes on **DecAlign** reinforce a split between modality-shared and modality-specific latents, aligned with prototype-guided optimal transport and distribution matching.
- **Non-autoregressive multimodal generation is becoming more credible:** **CrossFlows** was highlighted as a flow-matching approach over joint discrete + continuous spaces, which is relevant to KmiDi if symbolic tokens, audio latents, and control signals are modeled in one program.
- **Robustness is now a first-class training objective:** **DMAST** was described as a two-player multimodal safety regime using imitation learning, supervised fine-tuning, and RL self-play to harden systems against cross-modal attacks.
- **Prototype abstraction is a scaling lever:** **MFCPL** uses shared prototypes plus contrastive alignment to keep training useful when modalities are missing, which matters for KmiDi where some corpora have audio only, symbolic only, or sparse affect labels.
- **Hybrid objective stacking appears to be the new default:** the briefing called out a converged recipe of contrastive loss for alignment, masked prediction for intra-modal structure, and matching/translation losses for cross-modal mapping.
- **Incremental modality onboarding matters more than full retraining:** **CACARA** was cited as evidence that new modalities can be attached relative to an anchor modality without full multimodal retraining.
- **Generalization techniques may shift from dataset scaling to architecture normalization:** **OmniVaT** was described as mapping modalities into a shared embedding-frequency space to reduce domain and modality gaps.
- **Workshop themes confirm the direction of travel:** CVPR 2026 workshop topics reportedly center on self-supervised multimodal learning, multimodal diffusion, cross-modal transfer/adaptation, and multimodal LLM integration.

### 1.2 Implications for KmiDi

- Maintain a two-part latent design where shared emotion/structure space is separated from modality-private residue rather than forcing every modality into one collapsed embedding.
- Keep training pipelines multi-objective by default: contrastive alignment, masked/predictive structure losses, and explicit translation heads should be treated as complementary rather than mutually exclusive.
- Preserve the option to add control modalities incrementally (gesture, MIDI 2.0 PE streams, tactile surfaces, visual cues) without re-running the full training stack from scratch.
- Add robustness and missing-modality evaluation to every cross-modal benchmark; "works when audio and symbolic are both present" is not an adequate acceptance gate.
- Treat flow-style multimodal generation as a research branch, not the default runtime path, until export/runtime support is clearer for Apple-silicon deployment.

### 1.3 Caveats from the same briefing

- Many of these directions still build on contrastive alignment, so this is an evolution of CLIP-like paradigms rather than a clean replacement.
- Molecular, tactile, or federated multimodal papers may not transfer directly to music/audio-control stacks.
- Prototype-heavy methods can trade away instance-level fidelity.
- Flow-based multimodal generation is promising but not yet the obvious production default.
- Emergent-alignment schemes may rely heavily on a strong anchor modality such as text.

### 2. Practical Audio / Symbolic Backbone

#### 2.1 Perch audio embeddings

- **Repo:** `google-research/perch`  
- **Capability:** produces high‑dimensional audio embeddings (e.g. 1536‑d vectors) and ships `embed_audio.ipynb` for large‑scale embedding extraction.  
- **Key property:** designed for bioacoustics but functionally provides **JEPA‑style acoustic latents** that are already widely used for search, clustering, and classification.

For KmiDi, Perch can be treated as:

- A **frozen audio encoder** that maps waveforms or spectrogram segments to 1536‑d embeddings.  
- A building block for: audio retrieval, affect estimation, alignment to symbolic REMI tokens, and condition vectors for generation.

#### 2.2 REMI‑BPE symbolic tokenization

- **Library:** `MidiTok` — tokenizes MIDI/ABC into deep‑learning‑friendly token sequences.  
- **Format:** **REMI** with chords, tempo, durations, etc., plus subword models (BPE/Unigram/WordPiece) via `tokenizers`.  
- **Example model:** `Natooz/Maestro-REMI-bpe20k` — GPT‑style causal Transformer trained on Maestro with a ~20k BPE vocab over REMI.

This gives KmiDi:

- A **concrete output vocabulary** (REMI‑BPE) for symbolic generation heads.  
- Ready‑made code paths:
  - `REMI.from_pretrained(...)` for tokenization.
  - `AutoModelForCausalLM.from_pretrained("Natooz/Maestro-REMI-bpe20k")` for generation.  
- A replicable recipe via the **BPE‑Symbolic‑Music** repo (companion to “Byte Pair Encoding for Symbolic Music”, EMNLP).

### 3. Dataset Manifests: Lhotse + DataLad

Goal: deterministic, JEPA‑ready audio+MIDI windows with full provenance.

- **Lhotse:** Recording/Supervision/Cut abstractions + CLI to create **CutSets** (fixed windows, supervision‑aligned, etc.).  
- **DataLad:** Git+Git‑Annex wrapper for **byte‑exact versioning and provenance** with `datalad run` for command tracking.

Suggested schema (JSONL, one row per window):

- `recording`: audio info (path, sr, duration).  
- `supervisions`: timed annotations with `custom` fields (`midi_sidecar`, `emotion_label`, `bpm`, etc.).  
- `cut`: actual window (start, duration, channel).  
- `sha1_audio`, `sha1_midi`: hashes for reproducibility.

### 4. Real‑time Inference: BNNS Graph + Audio Workgroups

On Apple silicon:

- **BNNS Graph** supports:
  - Single‑thread targets at compile time (`TargetSingleThread`) to bound work.  
  - Preallocated workspace via custom allocation callbacks (no runtime heap).  
  - Streaming contexts for continuous audio.
- **Audio Workgroups:** coordinate BNNS worker threads with the Core Audio I/O thread so they share the same deadline and scheduling budget.

Pattern:

- **Compile** BNNS graph single‑threaded with RT‑safe options.  
- **Preallocate** all workspace/output buffers during init.  
- **Run** inference inside the Audio Workgroup, with a fixed‑size, lock‑free ring buffer for audio handoff between I/O and BNNS worker.

### 5. Shared Engine Boundary: Stable C ABI

Design:

- One **C ABI** over the core C++ DSP/latent engine (`engine.h`, `libengine.*`).  
- Two main faces:
  - **Realtime face:** `prepare` and `process` with raw float pointers, no allocs/locks.  
  - **Control face:** create/destroy, load model/assets, set params, get capabilities.
- AUv3 (JUCE) and Rust/Tauri both link to the same static library or via `dlopen`, ensuring:
  - Single, audited DSP core.  
  - No UI / host‑specific code in the engine dir.  
  - CI policies guard import edges and RT safety (no `new`/`malloc` in process).

### 6. MIDI 2.0 PE + UMP for Affect Control

- **Property Exchange (PE):** JSON resources over MIDI‑CI that describe **semantic properties** (names, ranges, units).  
- **UMP Channel Voice 32‑bit controllers:** hi‑res lanes for live control and DAW automation.

Proposed PE resource for KmiDi:

- Resource type `com.sburdges.kmidi/affect.v1`.  
- Properties:
  - `valence` (float, −1..1).  
  - `arousal` (float, 0..1).  
  - `dynamics` (float, 0..1).  
  - `timestamp`, `mode` (`ride` vs `snapshot`).

Mapping to UMP:

- Assign vendor channel controller indices (e.g. `0x28`..`0x2A`) for valence/arousal/dynamics.  
- Scale floats to 32‑bit ints and emit 32‑bit CCs at control‑rate (100–250 Hz).

### 7. Expressive Controllers (Sensel Morph etc.)

- **Sensel Morph:** pressure‑sensitive, MPE‑capable controller with high scan rates (125–500 Hz) and flexible mappings via SenselApp.  
- Well suited to map touch/pressure/slide gestures into PE + UMP affect lanes or MPE per‑note modulation.

## Integration Opportunities

1. **Audio backbone:** use Perch embeddings as the default **audio latent space** for JEPA‑style tasks (emotion, structure, retrieval, alignment to REMI‑BPE).  
2. **Symbolic backbone:** standardize on a **REMI‑BPE vocabulary** (20k tokens) for generative models and alignment targets.  
3. **Dataset layer:** define a **KmiDi JEPA manifest format** using Lhotse + DataLad for MAESTRO and future datasets (emotion‑tagged corpora).  
4. **Runtime layer:** codify BNNS + Audio Workgroup + Safe‑Mode traits as the canonical pattern for Apple‑silicon inference in KellyCore/KmiDi.  
5. **Engine boundary:** converge on one C ABI shim for all engine consumers (AU, Tauri, tools), enforced via CI.  
6. **Control semantics:** adopt PE + UMP affect schema as the **official live affect API** between devices, KmiDi, and DAWs.  
7. **Hardware prototyping:** standardize test harnesses around Morph / MPE controllers for expressive affect capture.

## Implementation Plan (Phased)

### Phase 1 — Documentation & Prototypes

- **R1.1:** Finalize this document and link it from `docs/research/README.md`.  
- **R1.2:** Create a minimal **Perch embedding script** (Python) that:
  - Reads MAESTRO audio.  
  - Outputs 1536‑d segment embeddings with timestamps and hashes.  
- **R1.3:** Add a **REMI‑BPE integration notebook/script**:
  - Load `Natooz/Maestro-REMI-bpe20k`.  
  - Tokenize sample MIDI, generate, and round‑trip decode.  
- **R1.4:** Prototype a **Lhotse manifest generator** for MAESTRO:
  - Emits Recording/Supervision/Cut JSONL as per the JEPA manifest sketch.  
  - Wrap generation in `datalad run` for provenance.

### Phase 2 — Core Pipeline & Training

- **R2.1:** Define a **KmiDi JEPA dataset spec** (separate doc) pointing to:
  - Manifest fields.  
  - Emotion/BPM labels.  
  - Audio/MIDI root paths and hash requirements.  
- **R2.2:** Implement JEPA‑style training scaffolding that:
  - Consumes Lhotse CutSets + Perch embeddings.  
  - Aligns to REMI‑BPE tokens / Maestro‑REMI‑bpe20k as a supervised or semi‑supervised head.  
- **R2.3:** Integrate representation analysis hooks:
  - Logging for cross‑modal similarity, linear canonicalization diagnostics.  
  - Optional PID‑style decomposition hooks for future work.

### Phase 3 — Runtime & Engine Integration

- **R3.1:** Implement the **C ABI shim** (`engine.h`, `libengine`) and:
  - Port at least one existing DSP/latent path into the engine core.  
  - Wire AUv3 and Tauri/Rust consumers through the ABI only.  
- **R3.2:** Add **BNNS Graph‑based inference** with:
  - Single‑thread compile target, preallocated workspace, and lock‑free ring buffers.  
  - Audio Workgroup integration where available.  
- **R3.3:** Implement a **Safe Mode** harness:
  - Headless RT harness that runs a golden session and records callback P50/P90/P99.  
  - CI gate that fails when P90 exceeds threshold.

### Phase 4 — MIDI 2.0 Affect Channel

- **R4.1:** Implement **PE resource** `com.sburdges.kmidi/affect.v1` in the MIDI I/O layer.  
- **R4.2:** Add **UMP 32‑bit controller emitters** for valence/arousal/dynamics in the engine / host glue.  
- **R4.3:** Build a small **JUCE + Python “affect sender” harness**:
  - Python Brain emits affect values to a local UMP endpoint.  
  - Plugin/host records and displays lanes.  
- **R4.4:** Prototype a **Morph mapping** configuration (SenselApp profile) that:
  - Maps grid regions + pressure to affect dimensions and optionally MPE per‑note modulation.

## References

- **Perch (audio embeddings):**  
  - Repo: `https://github.com/google-research/perch`  
  - `embed_audio.ipynb` for large‑scale embeddings.
- **Symbolic/BPE music:**  
  - `MidiTok`: `https://github.com/Natooz/MidiTok`  
  - Maestro‑REMI‑bpe20k model on Hugging Face.  
  - “Byte Pair Encoding for Symbolic Music” (EMNLP).  
- **JEPA / manifests:**  
  - `Lhotse`: `https://github.com/lhotse-speech/lhotse`  
  - `DataLad`: `https://www.datalad.org/`  
- **MIDI 2.0 & PE:**  
  - MIDI Association: State of MIDI 2.0 updates (2025–2026).  
  - Korg Keystage MIDI 2.0 PE examples.  
- **Expressive controllers:**  
  - Sensel Morph documentation and community mappings.  
- **Academic work (representative, non‑exhaustive):**  
  - CLCR (CVPR 2026).  
  - Dual‑domain contrastive frameworks for multimodal learning.  
  - Canonicalizing Multimodal Contrastive Representation Learning.  
  - StructAlign (ETF‑based continual multimodal learning).  
  - DecAlign (modality‑shared vs specific representation alignment).  
  - MAPLE (modality‑aligned preference learning).  
  - PID Flow (information‑theoretic decomposition of multimodal representations).  
  - Phi‑4‑reasoning‑vision‑15B training report.
