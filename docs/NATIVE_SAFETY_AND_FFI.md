# Native safety, FFI ownership, and verification

Human-readable reference for **memory safety across the KellyFFI boundary**, **duplicate JUCE / ODR**, **real-time audio constraints**, and **version drift** between layers.

**Canonical for automation and agents:** the same material lives in [`AGENTS.md`](../AGENTS.md) under **Native safety, FFI ownership, and verification map**. Update **`AGENTS.md` first** when paths or policy change, then align this file in the **same PR**.

**Related:** [`BUILD.md`](../BUILD.md), [`docs/FULL_STACK_BUILD.md`](FULL_STACK_BUILD.md), [`CLAUDE.md`](../CLAUDE.md) (ASan / CMake notes).

---

## FFI buffer ownership (who allocates / frees)

| What | Where | What to do |
|------|--------|------------|
The KellyFFI dylib exposes **one combined C ABI** with two halves:

- **`kelly_*` half** — implemented in C++ (`src/bridge/kelly_ffi.cpp`), declared in `src/bridge/kelly_ffi.h`. Owns `kelly_free_string` for caller-frees returns.
- **`IntentFrameBuilder_*` / `validate_intent_frame_ffi` half** — implemented in Rust (`engine/intent_ir/src/ffi.rs`) and linked **into** KellyFFI as a staticlib. Header is generated via cbindgen (`engine/intent_ir/cbindgen.toml`).

There is no separate Rust crate that consumes KellyFFI — Rust lives inside the dylib, not above it.

| What | Where | What to do |
|------|--------|------------|
| C++ ABI contract (which `char*` to free) | `src/bridge/kelly_ffi.h` — `kelly_free_string`, comments on static vs heap | Document every new `kelly_*` return: caller frees vs static (e.g. `kelly_get_error_message`). |
| C++ ABI implementation | `src/bridge/kelly_ffi.cpp` | Pair heap allocations returned across the boundary with `kelly_free_string` (or document static storage). |
| Rust ABI implementation | `engine/intent_ir/src/ffi.rs` | All `extern "C"` symbols use `Box::into_raw` / `Box::from_raw` for opaque handles, wrap bodies in `catch_unwind` (release `panic = "abort"` backstop), document out-pointer ownership. |
| Rust ABI header generation | `engine/intent_ir/cbindgen.toml` | When adding/changing an `extern "C"` symbol in `ffi.rs`, regenerate the C header so C++ callers see the same prototype. |
| External callers (C++ tests, Python bindings, plugin code) | `tests/cpp/`, `bindings/`, `src/plugin/` | Match the header: heap → callee documents the free fn; static/thread-local → **never** free. |
| Regression tests | `tests/cpp/test_kelly_ffi.cpp` (C++ side); `engine/intent_ir` `cargo test` (Rust side) | Extend when adding FFI; run with `BUILD_TESTS=ON` and C++ tests enabled. |

---

## Duplicate JUCE / ODR / allocator mismatch

| What | Where | What to do |
|------|--------|------------|
| CMake policy (PRIVATE JUCE on KellyFFI) | `CMakeLists.txt` (~KellyFFI: `target_link_libraries(KellyFFI ... PRIVATE juce::...)`, `JUCE_DISABLE_JUCE_VERSION_PRINTING`) | Do not add a second JUCE link to targets that only consume KellyFFI incorrectly — see CMake comments near `KellyFFIBenchmark`. |
| Benchmark pattern | `CMakeLists.txt` — `KellyFFIBenchmark` links **only** `KellyFFI` | Avoid `juce::` on small harness exes that already link KellyFFI. |
| Manual verification | Linker / `nm` / `otool -L` (macOS) | On double-free or JUCE init crashes: confirm one JUCE inside `libKellyFFI`, not a second copy in the host. |

---

## RT allocations, locks, audio thread

| What | Where | What to do |
|------|--------|------------|
| Lock-free RT snapshot type | `include/penta/common/RTState.h` | Atomics only; `static_assert` lock-freedom. No heap on RTState hot paths. |
| Plugin callback | `src/plugin/PluginProcessor.cpp`, `PluginProcessor.h` | Scrutinize `processBlock` for heap alloc, blocking locks, unbounded work. |
| Headless harness | `rt_harness/` (`BUILD_RT_HARNESS` in root `CMakeLists.txt`) | Use for callback regression when RT behavior changes. |
| Sanitizers | Debug + `KMIDI_ENABLE_ASAN=ON` (see `CLAUDE.md`) | Run `ctest` after native changes that might introduce UB or lifetime bugs. |

---

## Version and contract drift

| What | Where | What to do |
|------|--------|------------|
| KellyFFI ABI / dylib | `CMakeLists.txt` — `KellyFFI` `VERSION` / `SOVERSION`; `engine/intent_ir/Cargo.toml`; cbindgen-generated header | Breaking the C ABI (either half) requires a version bump and a coordinated update of all C++/Python consumers. |
| TS / Rust / Python intent | `shared_schemas/`, `scripts/sync_entities.py`, `src/types/Intent.ts`, `engine/intent_ir/src/generated/` | After schema edits: run sync, commit generated files, run Python schema tests. |
| HTTP API only | `music_brain/api_schemas/` | REST evolution; does **not** substitute for native memory safety. |

*(HTTP `intent_schema_version` is API contract versioning, not C++ toolchain versioning.)*

---

## Commands (when native or FFI changes)

- Configure/build: `KellyFFI`, `KellyCore`, plugins — see `BUILD.md` and root `CMakeLists.txt`.
- C++ tests: `BUILD_TESTS=ON`, then `ctest --test-dir build --output-on-failure` (when enabled).
- Sanitizer: `KMIDI_ENABLE_ASAN=ON` Debug + `ctest` per `CLAUDE.md`.
- Rust: `cd engine/intent_ir && cargo test`.
- Python (if API/schemas touched): `flake8 music_brain/`, `pytest tests/`.

---

## Integration gate (native / FFI) — excerpt

Full checklist (Python, schema, canonical tree, etc.) is in [`AGENTS.md`](../AGENTS.md) **Integration gate (merge checklist)**. Native-heavy PRs must include at least:

- [ ] Clean build of affected targets (`KellyCore`, `KellyFFI`, plugins, tests); no warnings promoted to errors without waiver.
- [ ] Sanitizer clean: `KMIDI_ENABLE_ASAN=ON` Debug, tests pass, zero ASan/UBSan (or documented ticket).
- [ ] No new heap allocations or locks on RT paths (`processBlock`, RT harness).
- [ ] No duplicate JUCE linkage / ODR violations; KellyFFI keeps JUCE **PRIVATE**.
- [ ] FFI ownership: new `extern "C"` pointers documented in `kelly_ffi.h` and mirrored in `kelly_ffi.rs`.

---

## See also

| Doc | Role |
|-----|------|
| [`AGENTS.md`](../AGENTS.md) | Canonical agent context + same map (keep in sync) |
| [`docs/FULL_STACK_BUILD.md`](FULL_STACK_BUILD.md) | React ↔ Music Brain API ↔ KellyFFI ↔ KellyCore build order |
| [`docs/DEVELOPMENT.md`](DEVELOPMENT.md) | Dev workflows and debugging |
