---
name: audit-kmidi
description: KMiDi-specific overlay for the security-audit skill. Adds RT-thread safety, FFI ownership, JUCE/Qt PRIVATE-link rules, ODR/allocator mismatch, GIL, ASan/TSan exclusivity, MIDI-CI/Core ML, dataset path leakage, schema-sync drift on top of the global audit skill.
---

# KMiDi Audit Overlay

Use **in addition to** the global `audit` skill. These are project-specific categories the global pass alone won't catch. Anchor docs: `AGENTS.md` (Native safety, FFI ownership, and verification map) and `~/Dev/totali/Docs/CXX_AGENTIC_RULES.md`.

## KMiDi-specific categories

### A. RT / audio-thread safety (binding)

| Pattern | Severity | Where it bites |
|---|---|---|
| Heap allocation on audio callback (`new`, `malloc`, `std::vector::push_back` that may realloc, `std::string` ctor with content) | critical | `processBlock`, any `noexcept` audio callback |
| Lock acquisition on audio thread (`std::mutex`, `std::lock_guard`, `std::scoped_lock`) | critical | RT path; use lock-free queues / `std::atomic` |
| Blocking I/O on audio thread (`std::cout`, file I/O, network, `printf`) | critical | RT path |
| `throw` / `catch` on a `noexcept` audio callback | critical | Causes `std::terminate` |
| Unbounded loop in audio callback | high | DSP block deadlines |
| `std::pmr` arena not used for per-block scratch | medium | Pattern violation |

### B. FFI / KellyFFI ownership

| Pattern | Severity |
|---|---|
| Executable links JUCE *and* links KellyFFI | critical (allocator mismatch / static-init crash) |
| Symbols re-exported across the FFI boundary (ODR-violating) | critical |
| Raw pointer crossing FFI boundary without documented ownership (`*kelly_ffi_*` returns) | high |
| FFI free function not paired with constructor (memory leak) or paired with wrong allocator (UB) | critical |
| C ABI function declared `noexcept(false)` (UB if it throws across the boundary) | critical |
| `kelly_ffi.rs` calls into KellyFFI without matching pairing comment ("// caller frees / callee frees") | medium |

Read **`AGENTS.md` → Native safety, FFI ownership, and verification map** for the canonical rule set; flag any divergence.

### C. JUCE / Qt link discipline

| Pattern | Severity |
|---|---|
| JUCE linked PUBLIC instead of PRIVATE on `KellyFFI` target | critical |
| Qt linked PUBLIC instead of PRIVATE on `KellyFFI` target | critical |
| `target_link_libraries(... PUBLIC juce::*)` on a target consumed by KellyFFI consumers | high |
| Two JUCE copies in the build graph (root + `KmiDi_FINAL`) | critical (ODR + static init) |
| Mixing root CMake `BUILD_PLUGINS` with legacy `DAIW_BUILD_VST3` / `DAIW_BUILD_AU` | high |

### D. ODR / static-init / allocator

| Pattern | Severity |
|---|---|
| `inline` global with non-trivial constructor in a header included from both sides of FFI | critical |
| `static` translation-unit-local with non-trivial dtor in a `.cpp` linked into both KellyFFI and a JUCE-linking executable | critical |
| `std::vector<T, AllocA>` returned from one DLL and freed in another | critical |
| Custom allocator referenced across translation units without ODR-safe declaration | high |

### E. Sanitizer / build correctness

| Pattern | Severity |
|---|---|
| `KMIDI_ENABLE_ASAN=ON` and `KMIDI_ENABLE_TSAN=ON` set together (mutually exclusive) | high |
| Release-only code path (`#ifdef NDEBUG`) hides a UB-triggering branch | medium |
| Sanitizer runtime mismatch between linked libraries | high |

### F. Python / FastAPI / GIL

| Pattern | Severity |
|---|---|
| `pybind11` releases GIL on a function that touches Python objects | critical |
| `pybind11` holds GIL on a function that performs long-running C++ work (blocks event loop / starves audio) | high |
| FastAPI route reads/writes a non-thread-safe global from multiple workers | high |
| `--timeout` passed to pytest (pytest-timeout not installed — convention violation) | low |

### G. Schema sync / contract drift

| Pattern | Severity |
|---|---|
| Edit to `shared_schemas/CompleteSongIntentRequest.json` without `python3 scripts/sync_entities.py` run | critical |
| Edit to `src/types/Intent.ts` or `engine/intent_ir/src/generated/intent.rs` (these are generated) | high |
| `/generate` API caller sends `instruments: ["piano"]` (must be `[{"instrument": "piano"}]`) | high |
| `structure` name not matching `^(intro\|verse\|chorus\|bridge\|outro\|build\|drop)$` | high |

### H. Dataset / weight path leakage

| Pattern | Severity |
|---|---|
| Hardcoded `/Users/<name>/...` path in source | critical (per data-governance rule) |
| Hardcoded `/Volumes/...` path in source | critical |
| Audio / MIDI / `.pt` / `.pth` / `.ckpt` / `.safetensors` / `.onnx` staged for commit | critical |
| Training run launched without `run_manifest.yaml` | high |
| Dataset path bypassing `KELLY_AUDIO_DATA_ROOT` env var | high |

### I. MIDI-CI / Core ML / Apple-silicon

| Pattern | Severity |
|---|---|
| Sub-4-bit quantization shipped as production (research-grade per watchlist) | medium |
| `coremltools` version unpinned in a code path expected to run on a user machine | medium |
| Stateful Core ML / KV-cache loop without parity test | medium |
| ANE/Metal fallback assumed without explicit validation | medium |

### J. App entrypoint drift

| Pattern | Severity |
|---|---|
| New code imports `App.tsx` (legacy) instead of `AppConsole` (canonical entrypoint in `main.tsx`) | high |
| New feature work landing in `KmiDi_FINAL/`, `KmiDi_PROJECT/`, or `.worktrees/integration-finalize/KmiDi_FINAL/` (legacy paths per CLAUDE.md) | high |

## Output

Same severity table as the global skill. Tag findings with `(kmidi)` so the user can sort. For RT-safety findings, also note whether the offending code is on a path reachable from `processBlock` (audio thread) — if not, downgrade severity by one level.

## Verification commands the audit may invoke (read-only)

- `git diff` and `git log --stat` (no `git apply`)
- `nm`, `otool -L`, `readelf -d` to verify link discipline (read-only inspection)
- `bash /Users/seanburdges/Dev/KmiDi/ci_listening_guardrails.sh --dry-run` if that flag exists; else **do not** run guardrails — they may write artifacts
- `cmake --build build --target ... -n` (dry-run mode) to inspect what would build

Never invoke a target build, sanitizer run, or guardrail script that writes artifacts. Audit is read-only.
