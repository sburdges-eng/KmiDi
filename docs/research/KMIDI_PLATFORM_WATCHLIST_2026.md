# KmiDi Platform Watchlist (2026)

**Author:** User briefings captured in repo  
**Date:** 2026-03-31  
**Status:** Informational / planning input  

## Purpose

This document stores user-provided 2026 platform notes that affect KmiDi's local runtime, export pipeline, symbolic data stack, and multi-agent orchestration. Treat each item as planning input until source verification is attached.

## 1. Structured multi-agent handoffs

Recommended operating pattern:

- **Architect:** turns a goal into a spec, constraints, and explicit acceptance tests.
- **Builder:** produces the artifact without mutating prior intent.
- **Critic:** evaluates the artifact against deterministic tests and failure rubrics.
- **Auditor:** records lineage, signatures, and compliance metadata.

Recommended baton properties:

- content-addressed refs only for large artifacts
- append-only writes
- signed outputs per role
- acceptance tests declared before build
- replayable runs with stored environment metadata

KmiDi mapping:

- Brain-side planning can play the Architect role
- Body-side realtime generation/rendering can play the Builder role
- rule checks, embedding-distance checks, and latency budgets fit the Critic role
- manifest/lineage/signature recording fits the Auditor role

Implication:

- if KmiDi formalizes baton handoffs, the baton should become a deterministic schema with explicit invariants, canonical ordering, and content hashes rather than ad hoc JSON blobs.

## 2. Safe desktop updates with Tauri

User-provided guidance favors enabling the Tauri v2 updater early:

- start with a static `latest.json` endpoint
- verify updater signatures in-app
- keep updater private key in CI secrets and ship only the public key in config
- use notarized signed macOS bundles for smooth post-update launches

KmiDi-specific note:

- desktop app can check for updates directly
- AU/plugin binaries still travel through the installer path, so the standalone app should surface update availability rather than attempting to hot-swap host-loaded binaries

## 3. Export and runtime watchlist for Apple silicon

### 3.1 PyTorch 2.11, TorchAO, ExecuTorch

- PyTorch 2.11 and TorchAO were highlighted as improving low-bit operators and export-adjacent quantization support.
- ExecuTorch remains the on-device export/deployment path to watch for Apple targets, including Core ML delegation and backend-specific partitioning.
- MPS support continues to evolve; visibility flags and profiler hooks matter when a model misses the fastest path.

### 3.2 Core ML state, KV cache, and optimization

- Core ML stateful models now support embedded state inputs/outputs for KV-cache style workloads.
- Newer `coremltools.optimize` APIs were described as supporting blockwise quantization, grouped/channel palettization, joint compression, and experimental activation quantization.
- Practical advice from the briefing: keep shapes fixed, benchmark compile times and p50/p99 latency, and quantize selectively rather than uniformly.

### 3.3 Converter/runtime caveats

- Rotary-attention and some `einsum` patterns can still be problematic in Core ML export paths.
- Beta OS releases may regress previously working Core ML behavior.
- CI should pin known-good exporter/runtime tuples and explicitly test delegate/fallback behavior.

### 3.4 Artifact trust

- Apple codesign/notarization is necessary for app delivery, but model artifacts benefit from separate Sigstore/Cosign-style attestations and transparency-log verification.
- The strongest model-loading policy is fail-closed verification of both app signature and model provenance.

## 4. Stateful streaming harness guidance

User-provided implementation guidance for JEPA/token and decoder loops:

- pre-allocate KV/state once
- micro-batch small token groups (for example 4-16 tokens)
- use stateful Core ML models or equivalent mutable-buffer export paths
- fix compile shapes up front
- keep decode work off the audio thread
- measure p50/p95/p99 plus allocation count, not only throughput

Recommended KmiDi adaptation:

- helper thread performs decode and writes control bursts into the C++ side
- audio thread consumes bounded, prevalidated control messages only
- long sessions should use a semantic ring buffer for KV eviction rather than unbounded growth

Segmented KV-cache idea:

- divide context into semantic segments such as bars, phrases, or control turns
- evict oldest sealed segments as a unit
- never evict the active segment
- preserve relative positions and attention masks after eviction

## 5. Foundation-model and representation watchlist

- **V-JEPA 2.1** was highlighted as a dense latent backbone worth studying for cross-modal prediction and probe-style downstream tasks.
- The implication for KmiDi is not "use video directly" but "reuse JEPA-style dense latent design patterns when bridging audio, symbolic, and control modalities."

## 6. Symbolic tokenizer and dataset notes

### 6.1 Compact REMI + BPE recipe

Saved defaults from the briefing:

- normalize MIDI to TPQN 480
- use REMI-style events
- use 32 velocity bins and tempo bins for compact expressivity
- keep an explicit special-token set
- train a BPE layer around a 16k vocabulary
- preserve bar/section markers if BPE is applied

### 6.2 Drum and expressive MIDI corpora

- **Groove MIDI Dataset (GMD):** small, expressive, drum-focused corpus for fast iteration
- **E-GMD:** larger drum transcription-oriented expansion with velocity labels
- **GigaMIDI:** very large expressive MIDI collection with heuristic expressive filtering

Implication:

- these corpora are good candidates for tokenizer bakeoffs, expressive-timing evaluation, and lightweight symbolic baselines before moving to larger mixed corpora

## 7. High-resolution controller note

- Haken 10.72 Beta firmware/editor notes were saved as a watch item for high-resolution MIDI and modulation workflows.
- If KmiDi is used with Continuum/EaganMatrix-class controllers, validate pitch-bend, pressure, and high-resolution controller density with the local CoreMIDI capture path before assuming throughput gains.

## 8. Audio-latency tracing and realtime IPC

User-provided low-latency guidance for AU/AUv3 work:

- timestamp host render entry, AU render in/out, and the plugin's own generation path
- track p50/p99/p999 over long callback runs
- use Audio System Trace and Time Profiler together
- keep p99 below roughly 70-80 percent of the device period for safety margin

Workgroup and scheduling notes:

- AU paths should use the host/device workgroup when available
- helper threads may need explicit realtime policy before `os_workgroup_join`
- VST3 may need fallback behavior because the same workgroup access is not always exposed
- QoS bumps help, but they do not replace proper workgroup membership and bounded work

IPC pattern saved from the briefing:

- use lock-free SPSC rings in shared memory
- keep capacities power-of-two and fixed
- use acquire/release ordering only
- no locks, allocation, logging, or file I/O on the realtime side
- batch bursty control data into fixed-size records so the RT side does O(1) copies

KmiDi implication:

- audio thread should capture and apply minimal DSP only
- feature extraction, model inference, and heavy coordination stay on worker threads joined to the same realtime cadence when possible

## 9. Emotion-lane control protocol

Saved interoperability defaults:

- expose **valence** and **arousal** as continuous lanes
- support both MIDI 1.0 CC14 and MIDI 2.0 32-bit UMP mappings
- advertise them through a small Property Exchange resource rather than hard-coded controller lore

Suggested defaults from the user briefing:

- valence range `[-1.0, 1.0]`
- arousal range `[0.0, 1.0]`
- MIDI 1.0 CC14:
  - valence -> CC 20/52 on channel 1
  - arousal -> CC 21/53 on channel 1
- MIDI 2.0 assignable parameter:
  - valence -> index 0
  - arousal -> index 1

Implementation note:

- if KmiDi promotes this into a repo schema, use deterministic field ordering and an explicit `schemaVersion`; do not ship the ad hoc example payloads as-is.

## 10. Emotion-thesaurus schema design

The user supplied a richer emotion-descriptor schema that is worth preserving as a design reference. Core fields:

- stable emotion id and human label
- valence and arousal
- optional coarse intensity
- tags
- cue groups for harmony, rhythm, and timbre
- optional small symbolic motifs
- optional human-readable descriptors and situations

Good storage pattern from the briefing:

- one file per emotion
- one manifest/index for discovery
- Python-side validation and API serving
- C++-side packed structs and prebuilt motif caches for fast access

Governance note:

- before this becomes production schema, normalize it to repo conventions:
  - `schemaVersion` first
  - explicit invariants
  - deterministic key order
  - no ambiguous free-form maps

## 11. Controllers worth testing

Additional controller notes captured:

- **Haken Continuum Fingerboard:** strongest continuous-surface option for ultra-high-resolution per-finger control
- **Expressive E Osmose:** more familiar keyboard form factor with strong MPE integration and wider deployability

Useful mapping heuristics from the briefing:

- pressure -> arousal/intensity
- front-to-back motion -> timbral evolution or brightness/valence proxies
- lateral pitch motion -> micro-intonation and expressive bends

## 12. Immediate actions if these notes become active work

1. Promote any adopted baton/update/model schema into a canonical repo schema with invariants and deterministic ordering.
2. Add fixed-shape, parity-checked export tests for any new Core ML or ExecuTorch model path.
3. Keep tokenizer and dataset experiments offline and reproducible with manifests and hashes.
4. Treat Tauri updater rollout, model attestation, and desktop notarization as one release-management lane rather than three disconnected tasks.
5. Build the emotion-thesaurus and emotion-lane specs together so training, control, and UI all share the same canonical vocabulary.
6. Benchmark AU/AUv3 workgroup plus SPSC IPC paths before assuming model inference is the main latency bottleneck.
