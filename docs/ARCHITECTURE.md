# KmiDi Architecture Handoff

Status: canonical handoff for completed workbook passes A, B, C, D, E, F, and G
Last updated: 2026-06-08

Purpose
- Preserve approved architecture decisions in repo documents so future work can continue without re-deriving them.
- Replace older high-level architecture narratives that no longer match the canonical tree.
- Make remaining open architecture gaps explicit instead of letting later work guess.

This handoff is authoritative for:
- global architecture principles
- repository/module mapping
- Intent IR authority and contract rules
- native runtime ownership and lifetime rules
- JUCE / plugin / real-time rules
- FFI / ABI / cross-language ownership rules
- project state / persistence / save-load architecture

This handoff currently leaves implementation drift and future feature details open, but no workbook pass remains open from the approved architecture interview set captured here.

## Executive summary

KmiDi is a hybrid AI-assisted music creation system for beginners and experts. The product is plugin-first in launch priority, but the internal architecture must preserve a future path to a standalone native engine. Intent IR is canonical for engine-facing and persisted intent. The native engine owns live runtime audio/music state. The plugin/runtime project layer owns user project/session state and host-facing persistence. Old persisted projects from stable releases should remain loadable via migration.

AI is additive, not foundational to core usability. Plugin loading, core editing, playback, saved project loading, and non-AI authoring workflows must survive offline and degrade gracefully when AI/model services fail.

## Product center and architecture center of gravity

Product center
- primary center: DAW/plugin runtime

Architecture center of gravity
- plugin runtime first
- preserve explicit extraction path to a standalone native engine

Non-negotiable implication
- host-facing code may be the shipping shell, but it must not become the permanent owner of musical truth or runtime truth

## Canonical truths by domain

- musical intent: validated Intent IR
- live generated audio/music state: native engine runtime
- host persistence/restore state: plugin runtime state contract
- user session/project state: plugin/runtime project state layer
- generation result metadata: Python backend

Conflict resolution winners
- intent meaning: Intent IR
- project state: plugin/runtime project layer
- generation result metadata: Python backend
- live playback/runtime state: native engine

## Optimization priorities

In descending order:
1. real-time safety
2. correctness
3. plugin stability
4. iteration speed
5. UX fluidity
6. model experimentation

Never sacrifice for speed
- user customization
- user control

## Failure modes to avoid

- audio/runtime instability
- user losing control to AI automation
- model/backend dependency making the core app unusable

## Offline and graceful degradation requirements

Must work offline or local-only
- core editing
- plugin runtime
- playback
- saved project loading
- non-AI authoring workflows

May depend on cloud/model services
- generation assistance
- enrichment
- remote model inference
- optional collaboration/search

Must degrade gracefully when AI/model systems fail
- plugin loading
- manual editing
- transport/runtime control

Must degrade gracefully when external integrations fail
- model provider outage
- backend offline
- DAW transport quirks
- missing audio/MIDI devices
- cloud auth failure

## Human-owned vs agent-safe surfaces

Freer agent refactor surfaces
- non-ABI C++ implementation files
- Python orchestration internals

Strict-checklist-only surfaces
- JUCE ownership surfaces
- FFI bridge code
- project state serialization

Always human review
- exported ABI changes
- Intent semantic changes
- persistence model changes
- migration rules
- deletion or renaming of persisted fields
- restore-order changes
- downgrade/export compatibility rules
- module boundary redefinitions
- dependency policy changes
- canonical vs legacy/support/migration reclassification
- standalone-engine extraction seam changes

Max blast radius for one agent task
- one module

## Pass status

Completed and approved
- Pass A — global architecture principles
- Pass B — repository/module mapping
- Pass C — Intent IR / data contract architecture
- Pass D — native engine architecture
- Pass E — JUCE / plugin / real-time rules
- Pass F — FFI / ABI / cross-language ownership
- Pass G — project state / persistence / save-load architecture

## Repo-level translation of completed passes

Current canonical module map and relevant surfaces
- `shared_schemas/` and `engine/intent_ir/`: Intent contract layer
- `src/bridge/kelly_ffi.h`, `src/bridge/kelly_ffi.cpp`, `src/bridge/intent_ir_ffi.cpp`, `engine/intent_ir/src/ffi.rs`: FFI/bridge layer
- `src/plugin/`, `src/project/`, `src/ui/`: plugin-runtime shell, with `src/ui/` as presentation under plugin/runtime authority
- `src/engine/`, `src/dsp/`, `src/midi/`, `src/common/`, runtime-needed `src/ml/`: native engine core
- `music_brain/`: backend/orchestration
- `scripts/`, `python/mcp/`, `training/`: tooling/integration
- `docs/`, `AGENTS.md`: docs/governance

Support and classification notes
- `src_penta-core/`, `include/penta/`, `include/prrot/`, and `libs/daiw/` are native support/code-organization surfaces, not independent semantic or persistence authority layers
- build outputs, caches, and rescue artifacts are never authority
- older docs that assume Tauri as the desktop/runtime center are historical, not architectural authority

## Document map

The completed workbook passes are converted into these focused documents:
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`
- `docs/AGENT_ALLOWED_SURFACES.md`
- `docs/HUMAN_OWNED_SURFACES.md`

Use them as follows:
- repo/module ownership, dependency direction, and classification questions -> `REPO_MODULE_MAP.md`
- semantic and schema questions -> `INTENT_IR_AUTHORITY.md`
- engine lifetime, mutation, shutdown, and runtime truth -> `NATIVE_RUNTIME_OWNERSHIP.md`
- plugin processor/editor/threading/RT constraints -> `JUCE_RT_RULES.md`
- C ABI, allocators, handles, and cross-language lifetime -> `FFI_OWNERSHIP_AND_ABI.md`
- persistence, migration, autosave, and compatibility questions -> `PERSISTENCE_AND_MIGRATION.md`
- refactor permissions and review policy -> `AGENT_ALLOWED_SURFACES.md`, `HUMAN_OWNED_SURFACES.md`

## Cross-pass synthesis

These rules are now architectural baseline:

1. Plugin-first product, engine-separable internals
- Ship around plugin/runtime first.
- Do not let host-facing code become the permanent owner of musical truth.

2. Intent IR is canonical where correctness matters
- Engine-facing and persisted intent must converge into validated Intent IR.
- UI may be looser, but only before normalization and validation.

3. Native engine owns live runtime truth
- Plugin/runtime owns project/session/persistence truth.
- Python backend owns generation metadata, not live runtime truth.

4. Core usability must survive AI/backend failure
- plugin load
- playback
- core editing
- saved project loading
- non-AI authoring

5. Persisted project compatibility is a long-term promise
- stable-release saved projects should remain loadable with migration

6. Ownership discipline dominates
- unique ownership by default
- shared ownership only in narrow non-RT cases
- raw pointers are observer-only

7. RT discipline is absolute
- no locks
- no allocation
- no network/syscalls/model calls
- no uncaught exceptions
- no unsafe dynamic mutation

8. FFI is a deliberate bridge, not a shortcut
- narrow ABI
- explicit allocator ownership
- no ambiguous lifetime rules
- no new exported symbols without human approval

9. Persistence is project-layer authority, not runtime-layer authority
- project file is canonical persisted project/session truth
- persisted intent inside it is canonical validated Intent IR
- plugin/runtime owns save/load orchestration
- native engine reconstructs runtime state from validated project truth
- autosave is recovery-oriented and never becomes live engine authority

10. Repo/module boundaries are now explicit architecture
- `src/ui/` belongs to plugin/runtime authority as presentation, not truth ownership
- `src/project/` is canonical project/session/persistence authority under plugin/runtime
- `src/ml/` belongs on the native side only when runtime-needed; broader experimentation should live in backend/orchestration
- engine extraction seams must remain preservable rather than collapsing into host shell code
- support/code-organization surfaces must not become accidental authority layers

## What remains after the architecture program

Architecture passes A through G are now captured.
Remaining work is implementation and governance follow-through, for example:
- align drifting code to the approved module map and ownership rules
- add verification/checklists where architecture promises need mechanical enforcement
- identify and shrink legacy or migration surfaces over time without changing authority implicitly

## Authoring and review policy

When these docs conflict with implementation, do not silently trust implementation.
- If code violates the completed pass decisions, treat the code as drift.
- If a proposed change would alter these decisions, route it through human review as an architecture change.
- If an implementation detail is ambiguous, prefer the focused authority doc for that domain instead of inferring from convenience or historical layout.
