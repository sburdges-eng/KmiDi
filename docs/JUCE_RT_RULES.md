# JUCE and Real-Time Rules

Status: authoritative for approved workbook Pass E
Last updated: 2026-06-08

Purpose
- Define plugin shell responsibilities.
- Define editor vs processor vs engine responsibilities.
- Define thread rules, handoff rules, and destruction hazards.

## Canonical plugin/runtime architecture

- host-facing plugin shell owns lifecycle, persistence, and host integration
- separable native engine owns live runtime execution
- plugin is first-class product shell, not owner of musical truth

## Ownership split

PluginProcessor owns:
- host lifecycle integration
- parameter/automation bridge
- serialization boundary
- EngineRuntime creation/ownership
- RT callback entrypoint
- transport/host adapter
- minimal host -> engine control plumbing

PluginEditor owns:
- UI/presentation
- transient UI state
- visualization
- user command issuance
- safe observation/subscription
- never runtime-critical ownership

Engine / DSP core owns:
- live execution state
- DSP/audio graph behavior
- timing-sensitive control application
- validated runtime intent/config application
- RT-safe state transitions
- audio-result production

Background workers own:
- non-RT preprocessing
- expensive analysis/precompute
- backend/model interaction outside RT
- resource preparation
- command/snapshot preparation for runtime interfaces

## Relevant threads

- audio thread
- message/UI thread
- background worker thread(s)
- model/backend orchestration thread(s)
- I/O/resource loading thread(s)
- timer/async callback ingress surfaces

## Thread policy by thread type

Audio thread may run:
- RT-safe processing only

UI thread may run:
- presentation
- interaction
- command creation

Worker threads may run:
- non-RT compute and preparation

Model/backend threads may run:
- orchestration
- inference coordination

I/O threads may run:
- loading
- device work
- resource work

## Forbidden on the audio thread

- allocation
- locks or blocking waits
- file I/O
- network I/O
- model/backend calls
- unbounded logging
- uncertified FFI/Python calls
- exceptions
- unsafe dynamic graph mutation

Absolute RT rules
- no locks
- no allocations
- no syscalls/network/model inference

## Non-RT -> RT handoff

Allowed handoff pattern:
- explicit command/snapshot handoff through bounded preallocated RT-safe queues or buffers
- only prevalidated stable data crosses into RT consumption

Sanctioned communication primitives:
- snapshots or double buffers for compound state
- bounded lock-free queue/ring buffer for commands/events
- atomics only for narrow scalar state

Consistency guarantees:
- RT sees coherent fully formed updates only
- UI may see slightly stale snapshots
- no torn compound state visible anywhere

## RT -> UI feedback

Allowed feedback pattern:
- one-way snapshot publication
- or bounded telemetry/event queue

UI reads:
- stale-tolerant view models
- parameter snapshots
- bounded telemetry

UI must not read:
- live RT internals
- raw audio-thread-owned mutable structures

## Component ownership rules

Preferred ownership patterns:
- direct member ownership where practical
- otherwise `std::unique_ptr`

Forbidden in new architectural code:
- owning raw `Component*`

Child widget ownership:
- parent owns via direct members or `std::unique_ptr`
- JUCE child registration does not change conceptual ownership

Legacy compatibility zones:
- `OwnedArray` and `ScopedPointer` tolerated only in quarantined legacy areas
- forbidden for new architecture

## SafePointer and deferred UI work

`SafePointer` is mandatory for:
- async, deferred, timer, or message callbacks targeting Components or Editors
- worker completion callbacks into UI
- any deferred observation path where destruction may race with callback execution

Async cancellation and destruction policy:
- explicit disconnect, cancel, or invalidate before target destruction
- no passive teardown assumptions
- `SafePointer` or equivalent weak observer guards required for deferred UI-targeting work

## Destruction-order hazards

Primary hazards:
- post-close editor callbacks
- stale UI references into processor/runtime
- worker completion into dead UI
- processor teardown with active ingress
- runtime teardown with outstanding RT/mutation references
- persistence/shutdown collisions

Fragile relationships to treat as hazardous:
- Editor <-> Processor observation across separate lifetimes
- UI -> Processor -> EngineRuntime callback chains
- background work updating UI or processor surfaces
- host lifecycle vs editor lifetime mismatch
- transport/automation/restore during UI transitions

Before editor close:
- detach listeners/subscriptions
- stop timers/updaters/deferred callbacks
- cancel or guard worker callbacks
- sever observation paths that can fire after close
- keep runtime continuity intact

Before processor shutdown:
- stop new ingress
- detach editor-facing observers
- stop or cancel background activity
- quiesce runtime mutation pathways
- remove RT dependence on dying resources
- perform ordered runtime teardown

Before project/session unload:
- flush or reject pending commands to old state
- persist through defined boundary if needed
- detach UI/project observers
- transition runtime via validated coherent session swap
- forbid lingering old-session references

## Performance and determinism policy

Most important budgets:
- audio callback deadline safety
- bounded non-RT -> RT control latency
- stable plugin load/open/close behavior
- responsive UI without harming RT
- predictable load/restore timing

Unacceptable jitter:
- audio overruns/glitches
- unstable timing of runtime control application
- host automation/transport response instability
- backend/model delays leaking into runtime behavior

Determinism:
- required for engine-facing runtime interpretation
- required for persisted project replay
- exceptions only for explicitly marked stochastic behavior

Latency variance policy:
- RT/runtime: tightly bounded
- UI: mild variance acceptable
- backend/model: broad variance acceptable only if isolated from core runtime and control continuity

## Repo anchors

Primary code surfaces:
- `src/plugin/PluginProcessor.cpp`
- `src/plugin/PluginProcessor.h`
- `src/plugin/PluginEditor.cpp`
- `src/plugin/PluginEditor.h`
- `src/ml/`
- `src/common/`

Verification and companion docs:
- `docs/NATIVE_SAFETY_AND_FFI.md`
- `docs/HEADLESS_ENGINE.md`
- `docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md`
