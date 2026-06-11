# FFI Ownership and ABI

Status: authoritative for approved workbook Pass F
Last updated: 2026-06-08

Purpose
- Define what KellyFFI is for.
- Define allowed and banned exported patterns.
- Define allocator, lifetime, and thread rules across C++, Rust, and Python.

## KellyFFI role

KellyFFI is a narrow, explicit, memory-safe-by-contract control/data boundary into the native runtime and Intent IR.

It is not:
- a broad convenience mirror of internal objects
- a shortcut around ownership rules
- a permission slip to expose raw internals

## ABI posture

- narrow stable ABI
- expand only through deliberate, reviewed, versioned additions

Intended consumers:
- Python bindings/backend
- tests/benchmarks
- internal tooling
- future external apps only if ABI discipline is explicitly maintained

## Stability policy

ABI stability is required within a declared version line.
Breaks are allowed only when they are:
- explicit
- reviewed
- version-bounded
- migration-accounted-for

Current-phase rule
- ABI breaks are allowed only with explicit version boundary and human review

Required ABI break process:
1. version bump
2. header/contract update
3. consumer-impact review
4. binding and test updates
5. compatibility or migration notes
6. CI/build/test gates

Compatibility posture
- manual compatibility policy first
- explicit ABI versioning and header discipline now
- symbol versioning only later if needed

## Allowed ownership / return styles

Allowed exported patterns:
- opaque handles with create/destroy pairs
- caller-provided output buffers where practical
- immutable returned snapshots/structs with explicit free if callee allocates
- borrowed pointers only when lifetime is trivial and documented
- static immutable data only when clearly non-freeable

Dominant preferred return styles:
- opaque handles
- caller-owned output buffers where practical
- immutable callee-allocated snapshots with exact free pair when needed
- borrowed pointers only for tightly bounded read access

## Banned patterns

Never export:
- undocumented ownership transfer
- raw internal object pointers as public API
- guessed or free-form deallocation rules
- cross-allocator frees
- unsynchronized mutable shared cross-language structures
- exceptions or panics crossing FFI
- undocumented thread-affinity requirements

## Freeing and allocator rules

Freeing rule
- allocating side frees
- only the matching ABI free/destroy function may release allocated memory

Free thread policy
- non-RT threads by default
- stricter affinity must be documented explicitly
- never require free/destroy on audio thread

Invalidation triggers include:
- matching destroy/free
- owning runtime/session destruction
- documented reset/reload/unload operations
- documented borrow lifetime end on snapshot replacement

## Handle lifetime and thread policy

Cross-thread handle policy
- only when explicitly documented thread-safe
- default is no unrestricted cross-thread mutation or use

Outliving policy
- no runtime or structural handle may outlive its owning session/runtime
- detached immutable copied snapshots may outlive the runtime only if explicitly documented

## Rust / C++ ownership boundary

Core rule
- each side owns what it allocates

Rust rules
- Rust FFI allocations are freed by Rust free functions
- Rust-returned allocated buffers/strings/structs are freed by Rust-side free functions
- exception only for explicitly documented static immutable data

C++ rules
- C++ may copy or internalize Rust outputs into native-owned forms
- C++ may not retain undocumented long-lived references into Rust-owned allocations

## Panic and error boundary policy

Panic policy
- panics are fully contained
- Rust FFI entrypoints must not let panic behavior cross the boundary
- failures convert to ABI-safe error results

Validation failure policy
- return structured nonfatal error
- do not partially initialize runtime state
- do not silently coerce invalid intent into engine truth

Exceptions policy
- no exceptions across FFI

## Python binding policy

Python must not receive general raw ownership-sensitive pointers.
Preferred Python interaction patterns:
- opaque handles
- copied data
- serialized payloads
- managed bindings

Binding strategy
- binding technology may evolve
- stable C ABI contract remains primary
- all bindings must preserve ownership and error rules

Lifetime safety enforcement
- both layers matter
- C ABI is foundational
- bindings add ergonomic safety guards

## New exported symbols policy

Agents may not add new exported symbols without explicit human approval.

## Repo anchors

Primary files and dirs:
- `src/bridge/kelly_ffi.h`
- `src/bridge/kelly_ffi.cpp`
- `engine/intent_ir/src/ffi.rs`
- `engine/intent_ir/cbindgen.toml`
- `tests/cpp/`
- `bindings/` if present for consumers
- `docs/NATIVE_SAFETY_AND_FFI.md`

Boundary shape reminder
- KellyFFI is one combined dylib with a C++ `kelly_*` half and a Rust `IntentFrameBuilder_*` / `validate_intent_frame_ffi` half embedded into the same artifact

## Review requirements

Always require human review for:
- ABI surface changes
- exported symbol additions or removals
- ownership contract changes
- free/destroy contract changes
- thread-affinity changes
- persistence-visible ABI-version changes
