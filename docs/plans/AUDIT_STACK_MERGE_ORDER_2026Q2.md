# Audit stack merge order — 2026-04-25

> The 2026 Q2 audit work is on a 13-PR linear stack rooted at
> `audit/rt-ffi-safety-2026q2` → `main`. This doc captures the
> **prerequisite-ordered merge sequence**, including the seven sub-PRs
> opened on 2026-04-25:
>
> - **Six bot-fix PRs** (#162–#167) addressing Bugbot + Codex findings.
> - **One build-system fix PR** (#169) unblocking the `AGENTS.md`
>   integration gate (sanitizer linkage, ctest registration, and a
>   real shipping blocker introduced when PR #159 deleted source files
>   that `libs/daiw/CMakeLists.txt` still referenced).

## Stack at a glance

```
main
└── #149  audit/rt-ffi-safety-2026q2          (T0–T7 RT/FFI safety integration)
    ├── #162  cursor/rt-ffi-fixes-149-893f          (4 bot-finding fixes for #149)
    ├── #150  follow-up/wound-odr-consolidation     (Wound tri-defined → canonical)
    │   └── #163  cursor/wound-types-h-compat-893f  (unbreak repo-root common/Types.h consumers)
    ├── #151  follow-up/state-bridge-wire-real-caller (StateBridge first real caller)
    │   └── #164  cursor/kellybrain-json-escape-893f (JSON-escape intent.key/mode)
    └── #152  audit/cxx-debt-pack-2                  (BUILD_TESTS repair + WAV bound-check)

#149 ── (linear chain through dependent bases) ──
└── #153  audit/cxx-debt-pack-3                       (CMake ODR fix + 3 noexcept lies)
    ├── #165  cursor/spectral-vector-include-893f     (<vector> include in SpectralAnalyzer.h)
    └── #154  audit/ui-lifetime-fixes-2026q2          (8 JUCE Component lifetime findings)
        ├── #166  cursor/tooltip-singleton-fix-893f   (TooltipComponent UAF fix)
        └── #155  audit/harmony-groove-osc-survey     (src/ vs src_penta-core/ ODR survey)
            └── #156  audit/odr-dedup-2026q2          (delete 4 duplicate .cpp files)
                └── #157  audit/biometric-guard-fix   (HealthKit destructor leak)
                    └── #158  audit/dead-code-sweep   (24-candidate audit doc)
                        └── #159  audit/dead-code-delete       (-2528 lines)
                            └── #160  audit/dead-code-delete-rest (-3426 lines)
                                └── #161  audit/regression-guards (basename collision scanner)
                                    └── #167  cursor/scanner-fix-893f (parser + allowlist fixes)
```

## Recommended merge order (prerequisite → leaf)

| # | PR | Base ← Head | Net | Class |
|---|---|---|---|---|
| 1 | **#162** | `audit/rt-ffi-safety-2026q2` ← `cursor/rt-ffi-fixes-149-893f` | 4 files, +43/-22 | bot fixes (HIGH/MEDIUM) |
| 2 | **#163** | `follow-up/wound-odr-consolidation` ← `cursor/wound-types-h-compat-893f` | 4 files, +26/-4 | bot fix (HIGH) |
| 3 | **#150** | `audit/rt-ffi-safety-2026q2` ← `follow-up/wound-odr-consolidation` | 3 files, +11/-19 | ODR fix |
| 4 | **#164** | `follow-up/state-bridge-wire-real-caller` ← `cursor/kellybrain-json-escape-893f` | 1 file, +47/-2 | bot fix (LOW) |
| 5 | **#151** | `audit/rt-ffi-safety-2026q2` ← `follow-up/state-bridge-wire-real-caller` | StateBridge wiring | feature |
| 6 | **#152** | `audit/rt-ffi-safety-2026q2` ← `audit/cxx-debt-pack-2` | BUILD_TESTS + WAV bound | bug fixes |
| 7 | **#149** | `main` ← `audit/rt-ffi-safety-2026q2` | 49 files, +6301/-547 | foundation |
| 8 | **#165** | `audit/cxx-debt-pack-3` ← `cursor/spectral-vector-include-893f` | 1 file, +1/-0 | bot fix (P1) |
| 9 | **#153** | `audit/rt-ffi-safety-2026q2` ← `audit/cxx-debt-pack-3` | CMake ODR + noexcept | mixed |
| 10 | **#166** | `audit/ui-lifetime-fixes-2026q2` ← `cursor/tooltip-singleton-fix-893f` | 2 files, +42/-23 | bot fix (MEDIUM) |
| 11 | **#154** | `audit/cxx-debt-pack-3` ← `audit/ui-lifetime-fixes-2026q2` | UI lifetime | bug fixes |
| 12 | **#155** | `audit/ui-lifetime-fixes-2026q2` ← `audit/harmony-groove-osc-survey` | survey doc | docs |
| 13 | **#156** | `audit/harmony-groove-osc-survey` ← `audit/odr-dedup-2026q2` | -608 lines | cleanup |
| 14 | **#157** | `audit/odr-dedup-2026q2` ← `audit/biometric-guard-fix` | HealthKit dtor | bug fix |
| 15 | **#158** | `audit/biometric-guard-fix` ← `audit/dead-code-sweep` | audit doc | docs |
| 16 | **#159** | `audit/dead-code-sweep` ← `audit/dead-code-delete` | -2528 lines | cleanup |
| 17 | **#160** | `audit/dead-code-delete` ← `audit/dead-code-delete-rest` | -3426 lines | cleanup |
| 18 | **#161** | `audit/dead-code-delete-rest` ← `audit/regression-guards` | scanner | tooling |
| 19 | **#167** | `audit/regression-guards` ← `cursor/scanner-fix-893f` | 1 file, +54/-24 | bot fix (MEDIUM) |
| 20 | **#169** | `audit/regression-guards` ← `cursor/sanitizer-and-ctest-wiring-893f` | 2 files, +42/-39 | build-system fix (CMake) |

> **Practical cadence**: merge child fixes (#162/#163/#164/#165/#166/#167)
> **into their respective parent stack PRs** before merging the parent
> into `main`. GitHub will auto-update each downstream PR's `head` after
> the parent merges, so the order can also be: merge each fix into its
> parent, then merge each parent into its grandparent, all the way
> down to `main`.

## Alternative: collapse-and-merge

Once all sub-PRs (#162–#167) are merged into their parent branches,
the entire chain can be merged into `main` as a single fast-forward
of `audit/regression-guards`. That's 13 PRs × 6 fix PRs collapsed
into one merge commit + a clean linear history.

## Pre-merge gates

Per `AGENTS.md` § Integration gate, every PR touching native code must
satisfy these gates **before merge**. As of 2026-04-25, gates 1–3 below
have all been run locally on macOS arm64 (Apple clang 21, cmake 4.3.1,
ninja 1.13.2) against the integration of `audit/rt-ffi-safety-2026q2`
(#149) + `cursor/rt-ffi-fixes-149-893f` (#162) + the build-system fix
(#169). Results recorded inline.

### ✅ Gate 1 — `cargo test` on the Rust FFI crate

```
$ cargo test --manifest-path engine/intent_ir/Cargo.toml
running 9 tests
test validator::tests::test_builder_invalid_values ... ok
test validator::tests::test_invalid_version ... ok
test validator::tests::test_builder_validation ... ok
test validator::tests::test_clamp_arousal ... ok
test validator::tests::test_clamp_tempo_bias ... ok
test validator::tests::test_time_scope_validation ... ok
test validator::tests::test_valid_intent_frame ... ok
test validator::tests::test_clamp_valence ... ok
test validator::tests::test_version_supported ... ok

test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured
```

### ✅ Gate 2 — Release build of `KellyCore` + `KellyFFI`

```
$ cmake -S . -B build-gate -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON
$ cmake --build build-gate --target KellyCore KellyFFI -j8
[223/224] Linking CXX static library libKellyCore.a
[224/224] Linking CXX shared library libKellyFFI.1.0.0.dylib
```

`otool -L build-gate/libKellyFFI.dylib` shows no JUCE rpath entry —
JUCE is statically linked into the dylib (PRIVATE linkage, per the
`AGENTS.md` integration-gate "no duplicate JUCE" rule).

### ✅ Gate 3 — ASan + UBSan, full `KellyCore` + `KellyFFI` build + `ctest`

```
$ cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON \
        -DBUILD_TESTS=ON -DKMIDI_ENABLE_ASAN=ON
-- KellyCore: ASan + UBSan enabled
-- KellyFFI:  ASan + UBSan enabled

$ cmake --build build-asan --target KellyCore KellyFFI -j8
[244/245] Linking CXX shared library libKellyFFI.1.0.0.dylib

$ ctest --test-dir build-asan --output-on-failure
1/2 Test #1: RTStateSeqlockTest .............. Passed   0.26 sec
2/2 Test #2: SIMDKernelsTest ................. Passed   0.00 sec
100% tests passed, 0 tests failed out of 2
```

`RTStateSeqlockTest` runs 1,000,000 writer/reader iterations with
**zero tearings** under ASan + UBSan with `halt_on_error=1`.

### ✅ Gate 4 — TSan, RT seqlock torture

```
$ cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_KELLY_CORE=ON -DBUILD_TESTS=ON -DKMIDI_ENABLE_TSAN=ON
$ TSAN_OPTIONS=halt_on_error=1 ./build-tsan/RTStateSeqlockTest
iterations=1000000 tearings=0 snapshot_failures=755412 pre_publish=0
```

Zero data-race reports across 1M iterations; the `s_pyStartupOnce`
collapsed `Py_Initialize`/`PyEval_SaveThread` change from #162 doesn't
trip TSan.

### ✅ Gate 5 — Live FFI smoke test (Release dylib + new error-message branch)

```
$ ./ffi_smoke   # links only libKellyFFI.dylib via the C ABI
KELLY_SUCCESS:        Success
KELLY_ERROR_AGAIN:    Transient seqlock contention; retry the call
KELLY_ERROR_NULL_PTR: Null pointer error
```

Confirms the new `KELLY_ERROR_AGAIN` error-message branch from #162
fix #4 is reachable from a real consumer linking only the dylib (not
KellyCore). Same output under ASan + UBSan with `halt_on_error=1`.

### ✅ Gate 6 — Python unit suite

```
$ python3 -m pytest tests/unit/ -q --no-header --tb=no
270 passed, 35 skipped, 4 failed in 1.30s
```

The 4 failures are **pre-existing** (test_spectocloud_rejects_*,
test_emotion_rust_exists_after_sync, test_intent_frame_rust_exists_after_sync
— all present on `main` already, not regressions caused by the audit
stack).

### What still requires a real merge box

- **Plugin smoke test in a DAW** — load `KellyPlugin_VST3` in a host,
  exercise `processBlock`, observe no allocation / glitches under
  `KMIDI_BUILD_JUCE_UI=ON -DBUILD_PLUGINS=ON`. Apple clang 21 + cmake
  4.3.1 produce the `.vst3` bundle, but a human ear is the test.
- **Concurrent Python bridge stress** — exercise `StateBridge` +
  `PreferenceBridge` on multiple threads with the collapsed
  `s_pyStartupOnce` from #162 fix #3.
- **GitHub Actions CI runners** — every workflow run since 2026-04-21
  still fails in <10s with no runner assigned. PR #161's body documents
  this. Unblocking is owner-only and orthogonal to the merge.

### What's no longer a blocker

- ❌ ~~Cargo `edition2024`~~ — local cargo 1.94.1 handles it cleanly.
- ❌ ~~`external/JUCE/` missing in cloud agent~~ — local checkout has the
  full submodule, build verified end-to-end.
- ❌ ~~Build-system blocker after PR #159 deletes daiw_core sources~~ — fixed
  by PR #169 in this stack.

## Follow-ups (out of scope for this stack)

See `docs/plans/RT_FFI_FOLLOWUPS_2026Q2.md`:

- **Follow-up A** — full `kelly::Wound` consumer sweep (~20 files,
  reconcile `intensity` vs `urgency` semantic).
- **Follow-up B** — RT-safe arena-backed `StateBridge::emitStateUpdateRT`
  for audio-thread emission.

Both can ship after this stack lands; neither blocks merge.
