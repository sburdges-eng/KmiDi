# Human-Owned Surfaces

Status: derived from completed workbook passes A, B, C, D, E, F, and G
Last updated: 2026-06-08

Purpose
- Make explicit which architecture surfaces require human ownership or human review.
- Prevent accidental delegation of semantic, ABI, persistence, or module-authority decisions.

## Permanently human-owned decision classes

These decisions remain human-owned even if implementation help is delegated:
- Intent semantic changes
- exported ABI changes
- persistence model changes
- module boundary redefinitions
- dependency policy changes
- canonical/support/legacy/migration reclassification
- standalone-engine extraction seam changes

This includes changes to:
- what canonical intent fields mean
- who owns which runtime or persistence truth
- what an external consumer may rely on at the ABI boundary
- how saved projects are versioned, migrated, and restored
- which repo area is authoritative for a given responsibility
- which seams preserve or destroy the future standalone-engine path

## Human review required before merge

Always require human review for:
- exported symbol additions/removals
- ABI break decisions and version-boundary changes
- schema meaning changes in `shared_schemas/`
- Rust validator behavior changes that alter semantic acceptance/rejection
- persistence schema/model changes
- migration policy changes
- save/load authority changes between plugin/runtime and native engine
- deletion or renaming of persisted fields
- restore-order changes
- downgrade/export compatibility rule changes
- thread-affinity changes across FFI contracts
- ownership changes at JUCE lifecycle roots
- module boundary redefinitions
- dependency policy changes
- canonical/support/legacy/migration reclassification
- standalone-engine extraction seam changes
- moving semantic, runtime, or persistence authority between modules

## Why these surfaces are human-owned

They carry long-lived product promises:
- correctness of musical intent meaning
- safety of cross-language memory and lifetime rules
- compatibility of saved user work over time
- stability expectations for hosts, bindings, and future apps
- architectural clarity about which repo area owns which responsibility
- preservation of the future standalone-engine extraction path

A mistaken implementation can be fixed locally.
A mistaken semantic, ABI, persistence, or module-authority decision spreads externally and becomes expensive to unwind.

## Human-owned authority map

Intent meaning authority
- owned through `shared_schemas/`, validator behavior, and architecture/product docs

ABI authority
- owned through `src/bridge/kelly_ffi.h`, `src/bridge/kelly_ffi.cpp`, `engine/intent_ir/src/ffi.rs`, and version policy

Persistence authority
- owned at the plugin/runtime project layer and save-load architecture level
- migration, persisted field lifecycle, restore ordering, and compatibility rules remain explicit human-owned surfaces

Runtime ownership authority
- owned at the architecture level: plugin/runtime owns lifecycle and persistence boundaries; native engine owns live runtime truth

Module authority
- owned through `docs/REPO_MODULE_MAP.md` and the companion architecture docs
- support/code-organization surfaces do not get to redefine product, persistence, semantic, or runtime truth by convenience

## What humans may delegate safely

Humans may delegate implementation help for:
- code generation and drift cleanup
- non-semantic refactors
- documentation drafting
- compatibility-first wrappers
- tests and verification scaffolding
- architecture summaries and gap inventories
- narrow intra-module implementation cleanup

But the human still owns the decision if the work changes:
- semantics
- authority boundaries
- ABI promises
- persistence promises
- module boundaries
- dependency direction
- extraction seams

## Review prompts for humans

When reviewing changes in these surfaces, ask:
- Did intent meaning change or only representation?
- Did any ownership rule change implicitly?
- Did any save/load promise narrow or widen?
- Did migration behavior change?
- Did persisted field lifecycle or restore order change?
- Did a borrow/copy/free rule become less explicit?
- Did a runtime truth move to the wrong layer?
- Did a repo/module boundary move implicitly?
- Did a support surface become accidental authority?
- Did a historical implementation accidentally become canonical policy?

## Companion documents

Use this document together with:
- `docs/ARCHITECTURE.md`
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`
- `AGENTS.md`
