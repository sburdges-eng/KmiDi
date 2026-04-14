# KmiDi 90-Day Demo Roadmap (2026 Q2)

**Author:** User briefing captured in repo  
**Date:** 2026-03-31  
**Status:** Proposed  

## Executive Summary

The near-term path to a demo-ready KmiDi stack is:

1. lock a minimal emotion-and-intent contract,
2. prove a JEPA-style audio encoder on short clips,
3. ship a local AU helper that maps intent to stable DSP parameters under a strict latency budget.

This roadmap is intentionally narrow. It favors deterministic schemas, fixed-shape exports, and local-only runtime paths over broad feature expansion.

## Governance Notes

- The user-provided `intentframe` example should not be copied into production unchanged.
- If promoted into repo schema, it must use canonical ordering, explicit invariants, and a `schemaVersion` field as the first key.
- Cross-language parity must be enforced across Python, C++, and Rust if the contract moves into `shared_schemas/`.
- The runtime path stays offline-first: no cloud calls in the AU helper, no HTTP on the hot path.

## Phase 1: Contracts (Weeks 1-3)

Target outputs:

- `emotion_schema.yaml` as a single source of truth
- a strict `intentframe` schema
- validators and fixtures in Python and C++
- 10-20 golden fixtures under `tests/fixtures/intent/`

Minimal emotion thesaurus:

- valence in `[-1.0, 1.0]`
- arousal in `[0.0, 1.0]`
- tags drawn from a short controlled vocabulary such as `tension`, `release`, `warm`, `cold`, `bright`, `dark`, `drive`, `float`
- confidence in `[0.0, 1.0]`
- no more than 3 tags per clip/frame

Minimal intentframe fields:

- schema version
- monotonic timestamp in milliseconds
- emotion payload
- music hints: key, tempo, chord bias, section role
- DSP targets: filter cutoff, drive, reverb send
- explicit latency budget in milliseconds

Acceptance gates:

- Python and C++ validators reject unknown fields or apply documented safe defaults
- parity tests prove both languages read and write the same canonical structure
- fixtures are deterministic and checked in

## Phase 2: Lightweight Audio JEPA (Weeks 2-6)

Goal:

- infer valence, arousal, tags, and confidence from 2-6 second audio windows
- keep forward latency around 5-8 ms on Apple silicon for batch size 1

Plan:

- start with a tiny CNN or AST-style front-end over log-mel features
- use JEPA-style latent prediction for self-supervised pretraining
- fine-tune on small labeled emotion sets plus fast internal labels
- export fixed-shape ONNX and/or Core ML artifacts
- benchmark warm-started inference only; no mid-run allocations

Expected artifacts:

- `training/train_audio_jepa.py`
- `models/audio_jepa_v01.onnx` or equivalent local export
- `bench/latency_report.md`
- `inference/AudioEmotionRunner.{h,cpp}` with zero-allocation handoff

Acceptance gates:

- Spearman >= 0.75 on held-out valence/arousal
- tag F1 >= 0.6 on the small tag set
- <= 8 ms forward time on fixed windows

## Phase 3: Local AU Helper (Weeks 5-9)

Goal:

- consume the intent contract over FFI or shared memory
- map intent to stable DSP parameters every block without violating RT constraints

Architecture notes:

- JUCE AU remains real-time safe: no heap allocation or locks in the audio callback
- mapper stays tiny: quantized MLP or LUT plus deterministic smoothing
- bridging should prefer in-process FFI; shared memory queue is the fallback; HTTP is excluded from the hot path

Required features:

- per-parameter slew limiting to avoid zipper noise
- latency watchdog and last-known-good fallback on low confidence or missed deadlines
- dry-run/debug panel showing recent intent frames and current mapped state

Acceptance gates:

- no XRuns at a 64-sample buffer in the target demo chain
- end-to-end intentframe to audible change under 30 ms

## Phase 4: Demo Slice (Weeks 8-12)

Demo story:

- play or capture a short audio clip
- JEPA predicts emotion
- Brain emits a signed local intent frame
- AU helper updates filter, reverb, drive, and optional chord-bias visualization
- operator can switch among a small preset bank to show controlled morphs

Packaging:

- one AU component
- one local CLI sender such as `kmidi-intent-sender`
- one bench-and-verify script that runs latency checks, schema parity checks, and canned replay scenarios

## Tracker Copy

Milestone A:

- write emotion schema
- write intent schema
- add Python and C++ validators
- add golden fixtures and CI schema checks

Milestone B:

- pretrain JEPA on short unlabeled windows
- fine-tune on valence/arousal plus tags
- export fixed-shape artifacts
- add latency regression harness

Milestone C:

- stand up JUCE AU skeleton
- implement quantized intent mapper
- add slew limiter, watchdog, and shared buffer/FFI bridge

Milestone D:

- create 6 demo scenes
- add one-click build scripts
- add bench-and-verify automation and recorded artifacts

## Guardrails

- Fix tensor and message shapes early.
- Keep the tag vocabulary small.
- Always propagate confidence and define low-confidence fallback behavior.
- Log only on background threads.
- Use lock-free queues for telemetry.
- Keep the roadmap tied to deterministic fixtures and reproducible exports.
