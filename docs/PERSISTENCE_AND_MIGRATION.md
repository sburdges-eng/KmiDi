# Persistence and Migration

Status: authoritative for approved workbook Pass G
Last updated: 2026-06-08

Purpose
- Define canonical persisted project truth.
- Define save/load authority boundaries.
- Define what is persisted versus reconstructed.
- Define versioning, load behavior, autosave, and compatibility promises.

## Core persistence rule

The project file is the canonical persisted truth for user session/project state.

Within that persisted project truth:
- persisted intent is canonical as validated Intent IR
- engine runtime state is reconstructed, not persisted as authority

Short form
- save the project truth
- save canonical intent inside it
- rebuild live runtime machinery on load

## Save/load authority split

The plugin/runtime layer owns all save/load orchestration.
The native engine does not own persistence authority.

Plugin/runtime layer responsibilities:
- open/read project artifacts
- detect versions
- run migrations
- validate canonical persisted intent
- load project/session state
- decide what derived/runtime state must be rebuilt
- submit validated apply/swap inputs to the engine
- expose coherent loaded state to UI/editor surfaces

Native engine responsibilities:
- accept validated apply/swap inputs
- reconstruct live runtime state from validated project-layer truth
- reject invalid runtime-facing application requests
- avoid becoming persistence authority

## Persisted versus derived data

Persist as first-class canonical data:
- project/session metadata
- validated Intent IR
- arrangement/track/project structure
- user-editable control/parameter state
- references to external assets
- version/migration metadata

Derive or rebuild instead of treating as canonical persisted authority:
- live engine runtime objects
- transient caches
- backend-generated ephemeral metadata
- RT buffers/queues/snapshots

Implication
- if something can be deterministically or safely reconstructed from project truth, do not make it persisted authority by default

## File model

Chosen direction
- hybrid manifest plus optional sidecars for large/media/generated artifacts

Interpretation
- there is still one canonical project truth at the project level
- but that truth may reference additional artifacts when needed for scale or practicality
- large media and generated outputs do not need to force a monolithic single-file design

Guardrail
- sidecars must not create ambiguity about what the canonical project/session truth is
- if sidecars exist, the manifest/project root remains the authority entrypoint for loading

## Versioning scope

Required versioning layers:
- top-level project format version
- Intent IR version
- per-section/schema versions where needed

Interpretation
- top-level project format gates overall compatibility and migration entry
- Intent IR version protects semantic contract evolution
- per-section versions allow targeted migration without pretending the whole file evolved uniformly

## Load pipeline order

Canonical load order:
1. open manifest/project container
2. detect versions
3. migrate raw persisted structures forward
4. validate canonical Intent IR
5. load project/session state
6. reconstruct engine-facing runtime state
7. publish coherent session swap
8. expose UI/editor state

Why this order matters
- migration happens before canonical validation so older saves can be normalized into current meaning
- runtime is reconstructed only after project/session truth is established
- UI/editor should not observe half-loaded or pre-swap state

## Failure behavior on load

Chosen policy
- hard fail on canonical intent/project corruption
- soft-fail optional derived sections

This means:
- if canonical project truth or canonical persisted intent is corrupt or invalid, the load must fail explicitly
- if optional derived sections, caches, or non-authoritative generated artifacts fail, load may continue without promoting them to truth

Do not use best-effort loading to silently coerce broken canonical state into a loaded project.

## Autosave and crash recovery

Autosave policy:
- autosave stores project-layer state only
- autosave never becomes live engine authority
- autosave uses atomic replace or rotated snapshots
- on crash recovery, restore the latest valid autosave, then rebuild runtime

Chosen format direction
- lighter recovery snapshot format

Interpretation
- autosave is for fast, resilient recovery, not for redefining the canonical persistence model
- recovery snapshots may be slimmer than full save artifacts as long as they preserve the project-layer truth needed to recover user work safely

## Compatibility promise

Chosen policy
- forever compatibility for stable releases
- best effort for pre-release/dev saves

Implications
- released user work must remain loadable through migration
- development or unstable formats may change with weaker guarantees
- migration policy is part of the product promise, not a convenience feature

## Human-owned persistence surfaces

The following remain explicitly human-owned and require human review:
- migration rules
- deletion or renaming of persisted fields
- restore-order changes
- downgrade/export compatibility rules
- persistence model changes generally

These are architecture and product-compatibility decisions, not routine implementation details.

## Safe assumptions for future work

Future work may assume:
- project/session truth lives above live runtime truth
- persisted intent must converge to validated Intent IR
- save/load orchestration belongs to the plugin/runtime layer
- runtime is rebuilt from validated persisted truth
- autosave is recovery-oriented and not runtime authority
- stable-release compatibility is a long-term obligation

## Disallowed shortcuts

Do not:
- treat current runtime objects as the primary saved truth
- let sidecars become ambiguous competing sources of project truth
- expose UI/editor before coherent session swap is complete
- silently load corrupted canonical project state
- turn derived caches or ephemeral backend metadata into canonical persisted authority without explicit architectural review

## Repo anchors

Current implementation-related surfaces include:
- `src/plugin/PluginState.h`
- `src/plugin/PluginState.cpp`
- `src/project/ProjectFile.cpp`
- `src/project/ProjectManager.cpp`
- plugin save/open flows in `src/plugin/`

These files should evolve toward this architecture.
When implementation and this document disagree, treat the implementation as drift unless the architecture is explicitly changed through human review.
