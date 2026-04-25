# Audit stack merge order — 2026-04-25

> The 2026 Q2 audit work is on a 13-PR linear stack rooted at
> `audit/rt-ffi-safety-2026q2` → `main`. This doc captures the
> **prerequisite-ordered merge sequence**, including the six bot-fix
> sub-PRs opened on 2026-04-25 to address Bugbot + Codex review
> findings.

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
satisfy these gates **before merge**. They have not all been runnable
in the current cloud environment because:

- **CI is broken at the account level** (every workflow run since
  2026-04-21 fails in <10s with no runner assigned). PR #161's body
  documents this: *"all 100 recent runs fail in <10s with no runner
  assigned"*. Unblocking CI is a prerequisite to merging.
- **Cargo `edition2024`**: workspace cargo (1.83) doesn't yet stabilize
  the feature required by `getrandom 0.4.2` pulled transitively. Use
  cargo ≥1.85 (or set `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=false`).

What **was** verified locally on each fix PR's branch:

```
✅ python3 scripts/audit/cross_tree_basename_scan.py        (post #167)
✅ python3 -m pytest tests/unit/                            (306 passed,
                                                             4 pre-existing
                                                             failures present
                                                             on `main`)
✅ python3 -m flake8 music_brain/ --max-line-length 100     (240 informational,
                                                             0 critical via
                                                             E9/F63/F7/F82)
✅ g++ -std=c++17 -fsyntax-only on each modified .cpp/.h
```

## What still has to happen at merge time

1. **Unblock CI runners** at the account / GitHub Actions level.
2. **Run the full sanitizer + native gate**:
   ```
   cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
     -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON \
     -DBUILD_TESTS=ON -DKMIDI_ENABLE_ASAN=ON
   cmake --build build-asan -j8
   ctest --test-dir build-asan --output-on-failure
   ```
3. **Verify the new RT seqlock test** introduced in #149 passes after the
   #162 sequence-coherence fix:
   `ctest --test-dir build-tsan -R rt_state_seqlock`.
4. **Plugin smoke test in a DAW** (load KellyPlugin VST3, exercise
   `processBlock`, observe no allocation / glitches).
5. **Concurrent Python bridge stress** (`StateBridge` + `PreferenceBridge`
   on multiple threads with #162's `s_pyStartupOnce` collapsed
   `Py_Initialize`/`PyEval_SaveThread`).

Items 2–5 require `external/JUCE/` which isn't bootstrapped in the
cloud agent environment. They run on the merge box.

## Follow-ups (out of scope for this stack)

See `docs/plans/RT_FFI_FOLLOWUPS_2026Q2.md`:

- **Follow-up A** — full `kelly::Wound` consumer sweep (~20 files,
  reconcile `intensity` vs `urgency` semantic).
- **Follow-up B** — RT-safe arena-backed `StateBridge::emitStateUpdateRT`
  for audio-thread emission.

Both can ship after this stack lands; neither blocks merge.
