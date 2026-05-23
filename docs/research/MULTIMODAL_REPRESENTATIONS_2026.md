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

## Latent Control Core (in‑tree, 2026‑05‑22)

The latent stack in `music_brain/` shipped a set of real, individually tested primitives (audio JEPA, chord JEPA, emotion probe, cross‑attention, LoRA adapter, world‑model, vector store, streaming WAV writer) but no shared envelope. Without that envelope, downstream goals — multimodal fusion, persistent latent world‑state, chunked latent prediction, the conditioning bridge between `IntentFrame` and `CrossAttention` — could not be composed end‑to‑end. The latent control core (`music_brain/latent/`) closes that gap with a minimum compositional surface: a frozen `LatentFrame` dataclass that carries `audio_z (T, D)`, optional `chord_z (T_c, D_chord)`, an `emotion_va` tag, a monotonic `time_index`, the canonical `IntentProvenance`, and an immutable metadata bag. The `time_index` deliberately mirrors `StreamingWavWriter.chunk_index` so the audio‑side and latent‑side stream protocols stay reasoned about under one ordering rule.

Two regularizers ship alongside the contract. `L2NormProjection(radius)` is the safety net against unbounded latent growth on long world‑model rollouts: row norms above the radius are radially scaled back to the boundary, rows inside the ball pass through, zero vectors stay zero, and the boundary subgradient points inward so it is safe at training time. `VarianceFloor(min_var)` is a loss‑only regularizer that returns `(z_unchanged, aux_loss)`, where `aux_loss = relu(min_var - var(z, dim=batch)).sum()`. Hosts compose `loss + λ * aux` to prevent JEPA‑style collapse without distorting the inference forward pass. Both are pure functional operators — no trainable parameters of their own — so they slot anywhere without touching the host model's optimizer.

The conditioning bridge (`ConditioningProjection`) materializes an `IntentFrame` as four cross‑attention "slots" (emotion, music, time, provenance) rather than one concatenated token. Per‑slot Linears keep facet identities distinct and turn the cross‑attention weight into a 4‑vector diagnostic surface: each query position reads "I listened X% to emotion, Y% to musical preference, …". The featurizer is deterministic and total over `IntentFrame` — the `-1` "immediate/open" sentinel for bar indices, the `+inf` default for `max_event_rate`, and the integer mode‑preference field are all pre‑normalized before any Linear sees them, so no NaN/inf reaches the trainable surface. The thin `stream_decode(frames, *, start_index, on_chunk)` generator drains a `LatentFrame` producer in strictly monotonic order using the same heap‑buffered pattern as `StreamingWavWriter._pending`; out‑of‑order producers are tolerated, duplicates and gaps surface as errors.

### Wave 2 — Decoding (KV‑cache + constraints + sub‑16ms scheduler)

The decoding side of the latent core ships three composable primitives in `music_brain/latent/`. `KVCache(num_heads, head_dim, max_len)` pre‑allocates a `(1, H, max_len, D)` buffer pair and exposes `append` / `truncate` / `snapshot`. The hot path never allocates after construction, and `truncate(target_length)` is O(1) — the rollback ring uses that to revert a speculative branch without ever recomputing K/V. `DecodeConfig(temperature, top_k, top_p)` parameterizes the constraint sampler `sample_with_constraints(logits, cfg, forbidden_tokens, generator)`; the companion `greedy_argmax` is the zero‑jitter deterministic path used whenever any sampling‑induced output divergence is unacceptable. `JitterBoundedScheduler(budget_ms=16.0, greedy_threshold_ms=2.0)` keeps a rolling window of step latencies, computes a `mean + p95_factor * stdev` proxy for the p95 latency, and on each `next_step()` returns a `StepDecision(within_budget, recommend_greedy, headroom_ms, observed_p95_ms)`. The realtime `incremental_decode(logit_fn, max_steps, cfg, scheduler)` loop ties the three together: it asks the scheduler for budget headroom, switches to `greedy_argmax` when headroom drops below the threshold (preserving zero jitter on the audio thread), and reports each step's elapsed time back to the scheduler.

### Wave 3 — Closed‑loop personalization

`LatentMemory(dim)` wraps the existing `VectorStore` and mean‑pools `LatentFrame.audio_z` on insert; `remember(id, frame, metadata)` stores the pooled vector with a metadata blob that always carries `time_index`, `emotion_va`, and the `IntentProvenance` snapshot. `recall(query_frame, top_k, emotion_weight, user_id)` blends cosine similarity with a (valence, arousal) proximity score so emotion‑conditioned retrieval works without a separate index, and a `user_id` filter scopes the lookup to one listener's longitudinal history. `UserModel(user_id, ema_alpha, memory)` maintains the closed loop: every `FeedbackEvent(emotion_va, satisfaction, frame, frame_id)` advances a satisfaction‑weighted EMA of the user's preferred emotion bias, *and* re‑embeds the accepted frame into the attached memory. `calibrate_emotion_va(requested, blend)` is how downstream IntentFrame emission pulls personalization back into the next generation — small blend values preserve composer intent, larger ones let learned bias dominate. `RollbackRing(capacity)` checkpoints `(LatentFrame, KVCache.length)` pairs and on `rollback_to(time_index)` rewinds *both* the ring and the attached KV‑cache in O(1) — the audition‑safe regeneration loop the spec calls out.

### Wave 4 — Structure, motif, MIDI 2.0, scope

`SectionPlanner.plan_default()` emits the canonical 16‑bar intro/verse/chorus/outro `SectionPlan`; `section_at_bar`, `phrase_boundary_bars(emit_internal=True)`, and `bars_until_next_boundary` give the decoder bar/beat structural awareness so generation can anticipate section transitions one or two bars ahead. `MotifTracker(similarity_threshold)` is an online cosine clusterer over pooled `audio_z`: each `observe(frame)` either matches an existing motif (incrementing recurrence count and time‑index history) or seeds a new entry. `most_recurrent()` exposes the dominant motif so the orchestrator can deliberately repeat or vary it. `music_brain/latent/ump.py` ships the minimum MIDI 2.0 Universal MIDI Packet contract: `pack_midi1_note_on`, `pack_midi2_note_on` (16‑bit velocity + attribute), and `validate_ump(words)` which decodes the top‑nibble message type and enforces the spec word‑count‑per‑MT rule. `GenerationScope(start_bar, end_bar, intent_id)` is a half‑open bar range an orchestrator hands one per concurrent intent — `enforce_bar` raises `ScopeViolation` when a generator tries to write outside its lane, `collect` filters frames to in‑scope, and `overlap` returns the intersection so the orchestrator can detect collisions before dispatch.

### Wave 5 — Predictors, fusion, emotion trajectory, companion

`PredictionEngine(world_model, name)` is the generic trajectory wrapper around `WorldModel.rollout_frames`; the named subclasses `GroovePredictor`, `HarmonyPredictor`, `DynamicsPredictor` stamp the predictor identity into the trajectory metadata so the orchestrator can route concurrent predictions without inspecting weights. `MultimodalFusion(chord_dim)` concatenates `audio_z` and `chord_z` (zero‑padding when chord is absent) into one unified `(T, D_audio + D_chord)` tensor downstream models consume as the joint representation. `StemBundle(stems)` groups per‑instrument frames at one `time_index` and exposes `mix()` (concat — preserves stem identity) and `average()` (mean — low‑bandwidth cross‑stem summary). `EmotionTrajectory(waypoints)` linearly interpolates between (bar, valence, arousal) waypoints, clamps to the unit box, and accepts an optional `blend_with` user bias so composer‑specified arcs and per‑user calibration mix without a separate post‑process. `CompanionSession(user_id, calibration_blend)` is the human‑in‑the‑loop surface: `normalize_intent(valence, arousal, mood)` maps a short mood adjective to canonical `MusicalIntent` adjustments; `propose(frame, horizon)` rolls the world model forward with bias‑calibrated emotion; `checkpoint(frame)` / `accept(...)` / `reject(...)` close the loop — accept records positive feedback into `UserModel` + `LatentMemory`, reject rewinds via the `RollbackRing` and *still* records the feedback so the model learns from rejections too.
