# KmiDi Repository Module Map

Status: authoritative for approved workbook Pass B
Last updated: 2026-06-08

Purpose
- Define the durable module ownership map for the canonical repository tree.
- Define allowed dependency directions between major repo areas.
- Identify the intended standalone-engine extraction seam.
- Classify current repo areas as canonical, support, migration, generated, historical, or artifact so future work does not guess.

This document is architectural authority for repository/module mapping.
If current implementation layout conflicts with this map, treat the implementation as drift unless a human approves an architecture change.

## Canonical top-level architectural modules

1. Plugin-runtime shell
Primary responsibility
- host-facing plugin lifecycle
- parameter and automation bridge
- project/session authority
- save/load orchestration
- UI command issuance and presentation plumbing
- runtime creation/ownership from the host shell side

Canonical repo surfaces
- `src/plugin/`
- `src/project/`
- `src/ui/`

Notes
- `src/ui/` is a distinct presentation submodule, but it remains under plugin/runtime authority rather than becoming its own product-truth layer.
- `src/project/` is the canonical persisted project/session authority module under the plugin/runtime layer.

2. Native engine core
Primary responsibility
- live runtime audio/music truth
- DSP/audio graph behavior
- timing-sensitive control application
- transport/timeline/runtime state
- RT-safe native support needed by runtime execution

Canonical repo surfaces
- `src/engine/`
- `src/dsp/`
- `src/midi/`
- `src/common/`
- native runtime support in `src/` and `include/` that serves runtime execution

Notes
- `src/ml/` belongs here only when it is runtime-needed and respects native/runtime constraints.
- Native engine core is separable from the plugin shell even though plugin/runtime is the primary shipping center.

3. Intent contract layer
Primary responsibility
- semantic authority for engine-facing and persisted intent
- schema/version truth for validated Intent IR
- validator and generated contract surfaces

Canonical repo surfaces
- `shared_schemas/`
- `engine/intent_ir/`
- generated mirrors driven by schema sync

4. FFI / bridge layer
Primary responsibility
- narrow ABI boundary into native runtime and Intent IR
- allocator/lifetime contract enforcement
- boundary adaptation between internal native code and external consumers

Canonical repo surfaces
- `src/bridge/kelly_ffi.h`
- `src/bridge/kelly_ffi.cpp`
- `src/bridge/intent_ir_ffi.cpp`
- `engine/intent_ir/src/ffi.rs`

5. Backend / orchestration
Primary responsibility
- Python API
- normalization/orchestration before canonical validation/application
- generation workflows and metadata
- model/backend experimentation outside RT authority

Canonical repo surfaces
- `music_brain/`

Notes
- Backend owns generation metadata, not live runtime truth, project persistence authority, or intent semantics.

6. Tooling / integration
Primary responsibility
- developer tooling
- automation scripts
- MCP/tooling surfaces
- training and model workflows
- support utilities that are not canonical runtime/product truth

Canonical repo surfaces
- `scripts/`
- `python/mcp/`
- `training/`
- repo build/config support files where appropriate

7. Docs / governance
Primary responsibility
- architecture authority
- development rules
- migration/governance guidance
- execution constraints for humans and agents

Canonical repo surfaces
- `docs/`
- `AGENTS.md`
- `CLAUDE.md` if present

## Submodule placement decisions

`src/ui/`
- classification: plugin-runtime presentation submodule
- ownership: plugin/runtime layer
- authority: transient UI/presentation state only
- must not own runtime-critical state, persistence authority, or engine truth

`src/project/`
- classification: canonical project/session/persistence authority submodule
- ownership: plugin/runtime layer
- authority: project/session state and save/load orchestration

`src/ml/`
- classification: conditional native support submodule
- current policy: belongs on the native-engine side only when runtime-needed and compatible with native/RT boundaries
- preferred future direction: model/provider experimentation should trend toward backend/orchestration when not required for local native runtime behavior

## Standalone-engine extraction seam

The intended extraction seam is:
- native engine core
- Intent contract layer
- narrow bridge/adapter seams as needed

This means the future standalone-engine path should be able to carry forward:
- `src/engine/`
- `src/dsp/`
- `src/midi/`
- `src/common/`
- runtime-needed portions of `src/ml/`
- `shared_schemas/`
- `engine/intent_ir/`
- narrow bridge adapters where external control or packaging requires them

The plugin/runtime shell remains a replaceable host-facing shell around that core.
Do not design new features such that the plugin shell becomes the permanent owner of musical truth or live runtime truth.

## Allowed dependency direction

Use this as the default dependency policy.

Docs / governance
- may describe any layer
- must not become a runtime dependency

Tooling / integration
- may depend on public contracts, public formats, and stable interfaces
- should avoid depending on internal RT guts unless explicitly tool-only and non-authoritative

Backend / orchestration
- may depend on Intent contract and stable bridge/API contracts
- must not depend on plugin/runtime UI internals
- must not become the owner of live runtime truth or persistence authority

Plugin-runtime shell
- may depend on native engine core
- may depend on Intent contract
- may depend on FFI/bridge where required by packaging/interop boundaries
- owns host integration, project/session, and persistence orchestration

Native engine core
- may depend on Intent contract
- may depend on internal native support libraries
- must not depend on plugin/UI/backend layers

FFI / bridge layer
- may depend on native engine core and Intent contract
- must not depend on plugin UI
- must not become a shortcut around ownership and validation rules

Intent contract layer
- should remain dependency-light and authority-oriented
- generated artifacts depend on it; it should not depend on plugin/backend semantics

## Forbidden dependency shortcuts

These are architectural violations unless explicitly re-approved by a human:
- backend depending on plugin or UI internals
- engine depending on plugin, UI, or backend layers
- persistence semantics being owned by the engine layer
- UI talking around validated boundaries to mutate engine truth directly
- FFI becoming a broad convenience mirror of internal object graphs
- tooling artifacts becoming accidental semantic or persistence authority

## Repo area classification

Canonical authority surfaces
- `shared_schemas/`
- `engine/intent_ir/`
- `src/plugin/`
- `src/project/`
- `src/ui/` as presentation under plugin/runtime authority
- `src/bridge/` ABI/bridge surfaces
- `music_brain/`
- `docs/` architecture handoff docs

Canonical native support surfaces
- `src/engine/`
- `src/dsp/`
- `src/midi/`
- `src/common/`
- runtime-needed parts of `src/ml/`
- `include/penta/` as native support/public header surface for penta-related code
- `src_penta-core/` as a canonical native support surface for `penta::` namespace code where current build rules designate it canonical
- `libs/daiw/` as canonical low-level native support/toolchain surface for internal RT-safe primitives
- `include/prrot/` as specialized native support/public header surface, not a separate architecture authority layer

Migration / legacy-compatible surfaces
- any older duplicate or partially overlapping native implementations that exist only for compatibility, staged consolidation, or transition
- historical architecture docs whose narratives conflict with the current handoff

Generated surfaces
- generated Intent IR mirrors and other generated artifacts derived from canonical sources of truth

Artifact / non-authority surfaces
- `build/`
- `build-rescue/`
- `.cache/`
- compiled objects, local logs, and other transient build outputs

Important classification note
- `src_penta-core/`, `include/penta/`, `include/prrot/`, and `libs/daiw/` are not independent semantic, persistence, or product-authority layers.
- They are support/code organization surfaces inside the broader native architecture and must not accumulate conflicting authority with the canonical docs and module rules.

## Canonical vs historical docs

Historical documents may remain in-tree for reference.
They are not authority when they conflict with:
- `docs/ARCHITECTURE.md`
- this file
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`

In particular, older Tauri-centered narratives are historical and not the current architecture center of gravity.

## Human-owned Pass B surfaces

Always require human review for:
- module boundary redefinitions
- dependency policy changes
- canonical vs legacy/support/migration reclassification
- standalone-engine extraction seam changes
- moving persistence authority between modules
- moving semantic authority between modules
- moving runtime truth ownership between modules

## Agent execution implications

Agents may safely use this map to:
- scope tasks to one module
- reject cross-boundary refactors that need decomposition
- distinguish canonical source-of-truth layers from support or artifact zones
- avoid treating historical or generated surfaces as authority

Agents should not autonomously:
- redefine module boundaries
- widen allowed dependency directions
- reclassify ambiguous native support surfaces into new authority layers
- collapse plugin/runtime, engine, backend, and persistence responsibilities into one refactor

## Companion documents

Use this document together with:
- `docs/ARCHITECTURE.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`
- `docs/AGENT_ALLOWED_SURFACES.md`
- `docs/HUMAN_OWNED_SURFACES.md`
- `AGENTS.md`
