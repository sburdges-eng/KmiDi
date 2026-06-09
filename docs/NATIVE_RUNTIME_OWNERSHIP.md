# Native Runtime Ownership

Status: authoritative for approved workbook Pass D
Last updated: 2026-06-08

Purpose
- Define what the native engine owns.
- Define who may mutate runtime state.
- Define lifetime, teardown, and error rules for runtime-critical native code.

## Native engine role

The native engine owns and executes low-latency musical runtime behavior safely and deterministically.

It includes:
- live audio state
- timing-sensitive transformation
- engine-facing adaptation of validated Intent IR

Native is primarily responsible for:
- DSP/audio runtime
- low-latency generation/control

## Native must not become authority for

Permanently excluded from native ownership/authority:
- cloud/network concerns
- UI/view concerns
- product/session persistence authority
- model-provider-specific orchestration
- semantic authority over intent definitions
- long-latency backend workflow control

## Long-lived native objects

Expected long-lived native objects include:
- EngineRuntime / EngineSession
- transport/timeline state
- AudioGraph or DSPCore
- parameter/control state store
- intent adaptation/application state
- resource managers
- RT-safe queue/snapshot bridge objects

## Ownership model

- plugin/runtime layer owns EngineRuntime lifetime
- EngineRuntime uniquely owns core runtime subsystems
- RT path borrows stable non-owning references only
- UI/backend never directly own runtime-critical native objects

## Single-owner-only surfaces

The following are single-owner by default:
- runtime/session root
- DSP core/audio graph root
- transport/timeline state owners
- resource managers
- worker thread objects
- lifetime-critical queues and buffers

## Shared ownership exception policy

Shared ownership is allowed only for:
- immutable shared assets
- expensive shared non-RT resources
- read-mostly resource blobs outside RT-critical paths

If a surface is mutable, lifetime-critical, or RT-adjacent, do not use shared ownership unless a narrowly documented exception exists.

## Raw pointer policy

Raw pointers are acceptable only as non-owning observers for:
- RT access to pre-owned stable objects
- temporary function parameters
- JUCE observer interop
- strictly invalidated non-owning caches

Raw pointers are not ownership declarations.

## Creation and teardown authority

Creation authority
- plugin/runtime control layer through explicit factory/init boundaries

Teardown authority
- owning plugin/runtime layer through a single ordered shutdown path

Engine runtime may outlive editor/UI state.
No UI-owned object may own or outlive runtime-critical engine state.

## Shutdown order

Objects that die first:
- non-owning observers
- UI callbacks
- async listeners
- external ingress points

Guaranteed shutdown order:
1. stop new ingress
2. detach UI/editor observers and async callbacks
3. stop worker/background threads and queued work
4. quiesce runtime mutation pathways
5. destroy engine-owned subsystems in reverse ownership order
6. destroy EngineRuntime root last

## Mutation authority

Who may directly mutate engine state
- only native runtime control surfaces

Who may not directly mutate engine state
- UI
- plugin shell outside explicit control interfaces
- Python backend
- arbitrary foreign consumers

Allowed mutation mechanisms
- commands, snapshots, or queues for non-RT -> RT handoff
- atomics for narrow scalar signal state
- locks only outside RT-critical paths

## RT-thread state rules

Mutable on audio thread only if all are true:
- pre-declared RT-safe state
- bounded mutation
- allocation-free mutation
- lock-free mutation
- mutation over pre-owned data only

Immutable once handed to RT:
- validated/applied intent snapshots
- graph/configuration snapshots
- consistent parameter/control bundles
- RT-used resource bindings/lookups
- playback/control snapshots used by the audio callback

## Error policy

Recoverable errors
- explicit status/result forms at control boundaries
- never silent failure
- never exceptions across critical runtime boundaries

Fatal contract violations
- fail fast in dev/test
- controlled failure boundaries in release where possible
- do not silently limp through poisoned states

Mechanisms
- assertions for programmer/invariant failures
- Result/status returns for recoverable failures
- no exceptions across FFI
- no exceptions on RT paths
- exceptions strongly discouraged elsewhere

## Modernization policy

Modernization strategy
- compatibility-first wrappers at dangerous boundaries
- aggressive RAII modernization only inside contained subsystems once ownership roots are explicit

Raw-pointer-heavy legacy areas
- wrap first
- then selectively rewrite based on risk and centrality
- quarantine low-value legacy islands

## Hazardous structural surfaces

Too risky for agent-led structural rewrites without explicit human direction:
- plugin lifecycle roots
- RT callback core / processBlock-adjacent architecture
- exported ABI ownership surfaces
- persistence/serialization roots
- cross-language lifetime contracts
- transport/timeline mutation core

## Repo anchors

Key repo areas this policy applies to:
- `src/plugin/`
- `src/engine/`
- `src/dsp/`
- `src/ml/`
- `src/common/`
- `src/bridge/`
- `engine/intent_ir/` when it affects runtime-facing application of validated intent

Implementation-focused verification references:
- `docs/NATIVE_SAFETY_AND_FFI.md`
- `AGENTS.md`
- `docs/HEADLESS_ENGINE.md`
