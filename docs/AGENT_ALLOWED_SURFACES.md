# Agent Allowed Surfaces

Status: derived from completed workbook passes A, B, C, D, E, F, and G
Last updated: 2026-06-08

Purpose
- Define where agents may work with relative freedom.
- Define where agents may work only with strict checklists.
- Keep future autonomous work inside approved blast-radius limits.

This is a refactoring and execution policy, not a substitute for the architecture docs.

## Default blast-radius rule

Maximum blast radius for one agent task
- one module

Use `docs/REPO_MODULE_MAP.md` to decide what counts as one module.
If a task crosses multiple ownership domains, stop treating it as a single agent-safe change.
Split it, or route it to human review.

## Freer agent refactor surfaces

These are the most agent-safe areas, assuming normal repo verification still runs:
- non-ABI C++ implementation files
- Python orchestration internals
- non-semantic adapters that do not change canonical intent meaning
- documentation alignment that does not alter architectural authority

Examples
- internal C++ cleanup inside one subsystem without changing FFI signatures
- Python orchestration refactors that preserve behavior and ownership boundaries
- generated-file drift fixes paired with source-of-truth alignment

## Strict-checklist-only surfaces

Agents may work here only with explicit checklist discipline and narrow scope:
- JUCE ownership surfaces
- FFI bridge code
- project state serialization
- cross-module seam code where a change could blur plugin/runtime, engine, backend, or persistence boundaries

Checklist expectations for these surfaces
- verify ownership before editing
- preserve shutdown order and callback invalidation rules
- preserve RT-thread prohibitions
- preserve allocator/free pairing across language boundaries
- preserve module boundary directionality
- verify no semantic widening happened unintentionally
- require grounded file-level review before claiming success

## Surfaces that should default to human review even for small changes

- exported ABI changes
- Intent semantic changes
- persistence model changes
- version-boundary changes for persisted contracts
- changes that alter ownership authority between plugin/runtime, native engine, and backend
- module boundary redefinitions
- dependency policy changes
- canonical/support/legacy/migration reclassification
- standalone-engine extraction seam changes

## Strong cautions for agents

Agents should not lead structural rewrites of:
- plugin lifecycle roots
- RT callback core or processBlock-adjacent architecture
- exported ABI ownership surfaces
- persistence/serialization roots
- cross-language lifetime contracts
- transport/timeline mutation core
- repo/module seams that define the future standalone-engine boundary

Agents may inspect these areas, document them, and propose plans, but execution should remain constrained unless a human explicitly approves the exact change.

## Safe patterns for autonomous work

Preferred task shapes
- one file or one narrow subsystem
- compatibility-first wrappers
- additive docs/tests/checks
- generated-artifact sync after source-of-truth changes
- bug fixes that do not redefine ownership or semantics
- implementation cleanup contained entirely inside one approved module

Preferred execution style
- inspect affected files first
- restate the invariant being preserved
- make the smallest viable change
- run the relevant verification gates
- report unresolved architectural uncertainty instead of guessing

## Unsafe patterns for autonomous work

Avoid agent-led tasks that:
- redefine ownership implicitly through refactor churn
- mix semantic changes with formatting or modernization noise
- cross plugin, engine, persistence, and FFI boundaries at once
- introduce new exported symbols
- assume current persistence implementation equals final architecture
- silently turn historical docs into authority without review
- collapse support surfaces into new authority layers by accident

## Required companions

Use this document together with:
- `docs/ARCHITECTURE.md`
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`
- `AGENTS.md`
