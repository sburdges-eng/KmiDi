# Orphan branch triage — 2026-04-25

> 17 `cursor/*` and `feature/*` branches are ahead of `main` but not
> attached to any open PR. This doc classifies each and recommends a
> disposition (close-as-superseded vs. salvage vs. keep) so the repo
> can be tidied once the audit stack lands.

The audit/cleanup stack lives at PRs #149–#161 (head: `audit/regression-guards`).
Most of these orphan branches are alternative Cursor-generated takes
on the same fixes; many are **byte-equivalent in intent** to PRs
already in the stack.

## Disposition summary

| Disposition | Count | Action |
|---|---|---|
| Superseded by PR #149–#167, #169 | 12 | Close branch + delete after parent PR merges. |
| Likely subsumed (verify diff) | 3 | Diff against post-stack `main`, then close. |
| Independent — needs decision | 2 | Keep open, scope to a real PR or close. |

## Detail

| Branch | Last activity | Commits ahead | Disposition | Rationale |
|---|---|---|---|---|
| `cursor/healthkit-bridge-memory-leak-e7cc` | 2026-04-21 | 21 | **Superseded** by [PR #157](https://github.com/sburdges-eng/KmiDi/pull/157) | Same HealthKit destructor-guard fix landed in the audit stack. |
| `cursor/dangling-tooltip-singleton-pointer-df13` | 2026-04-21 | 25 | **Superseded** by [PR #166](https://github.com/sburdges-eng/KmiDi/pull/166) | Same `TooltipComponent` UAF fix; #166 stacks under #154. |
| `cursor/wound-struct-aggregate-initialization-3f3c` | 2026-04-21 | 17 | **Superseded** by [PR #163](https://github.com/sburdges-eng/KmiDi/pull/163) | Branch only fixes one of four `Wound`-aggregate-init sites and forgets `urgency`. PR #163 covers all four sites and mirrors `intensity` → `urgency` per #150's documented "alias" semantic. |
| `cursor/json-string-escaping-47c9` | 2026-04-21 | 17 | **Superseded** by [PR #164](https://github.com/sburdges-eng/KmiDi/pull/164) | Same `KellyBrain::generateMidi` JSON-escape fix; #164 stacks under #151. |
| `cursor/namespace-parsing-and-allowlist-2a35` | 2026-04-21 | 37 | **Superseded** by [PR #167](https://github.com/sburdges-eng/KmiDi/pull/167) | Same cross-tree scanner namespace-parse + allowlist-keying fix; #167 stacks under #161. |
| `cursor/runtime-synchronization-bugs-c15b` | 2026-04-21 | 16 | **Superseded** by [PR #149](https://github.com/sburdges-eng/KmiDi/pull/149) (T3, T4 of the RT/FFI audit) | Three sync bug fixes — verify each commit hash matches a T3/T4 fix in the audit stack before closing. |
| `cursor/intent-ir-ffi-bridge-issues-4f03` | 2026-03-30 | 1 | **Subsumed** by [PR #149](https://github.com/sburdges-eng/KmiDi/pull/149) (T2 + T1) | Single-commit branch fixing 3 bugs in `intent_ir_ffi.cpp`; T2 (Rust FFI hardening) covers the same surface. Diff before closing. |
| `cursor/uint8-t-pitch-overflow-60a2` | 2026-03-30 | 1 | **Salvageable** | Single-file fix touching `src/harmony/VoiceLeading.cpp` + 4 duplicates in legacy trees + `tests/cpp/test_pitch_overflow.cpp` (275 lines new test). Likely overlaps T5 of #149. Diff against post-stack `main`; if uniquely-different, salvage the test, close the rest. |
| `cursor/rust-test-env-var-race-2d5c` | 2026-03-17 | 1 | **Independent — needs decision** | Adds `serial_test` to `src-tauri/Cargo.toml`. Doesn't conflict with the audit stack. Either land as a single commit on `main` after the stack, or close as not-needed if the test it serializes was deleted. |
| `cursor/multiple-code-issues-f237` | 2026-04-14 | 20 | **Subsumed (verify)** | "Fix 4 bugs: legacy fallback method call, spectocloud indentation, VolumeRamp linear shape, duplicate helpers". The duplicate-helpers fix overlaps #156; spectocloud indentation might still be unique. Diff per-file before closing. |
| `cursor/processor-array-and-cleanup-2c82` | 2026-03-22 | 6 | **Independent — needs decision** | "Fix `models_` array OOB access and remove accidentally committed restaurant project" (84 files). The cleanup half (rm restaurant project) is independent of any audit work; the OOB fix needs comparison to `src/ml/AudioEmotionRunner.cpp`. |
| `cursor/repository-configuration-and-build-bba6` | 2026-04-07 | 5 | **Superseded** by `feat/workflow-gates-sanitizers` (already merged via #146) | "Fix 5 bugs: portable paths, sanitizer scope/guards, remove scratch dir". Same theme as the merged sanitizer-config PR. Verify before closing. |
| `cursor/tsan-preset-frame-pointer-6074` | 2026-04-07 | 5 | **Subsumed** by PR #146 | `-fno-omit-frame-pointer` for TSan; T3 of #149 already pulls in the seqlock TSan harness with frame pointers via `KMIDI_ENABLE_TSAN`. |
| `cursor/workflow-gates-sanitizers` | 2026-04-07 | 4 | **Already-merged ancestor** | This is the ancestor branch of merged PR #146. Just delete the remote ref. |
| `cursor/daiw-core-cmake-sources-465f` | 2026-04-21 | 36 | **Superseded** by [PR #169](https://github.com/sburdges-eng/KmiDi/pull/169) | PR #169 fixes the same `libs/daiw/CMakeLists.txt`-references-deleted-sources issue (the only real bug in this orphan branch's 36-commit churn) by converting `daiw_core` to an INTERFACE library, plus also fixes two adjacent CMake bugs (sanitizer wiring, ctest registration). Verified locally on macOS arm64 with full ASan + UBSan + TSan + ctest run. |
| `feature/ai-integration-and-readiness` | 2026-03-19 | 4 | **Independent — needs decision** | Stale (5 weeks old). Last commit is a `main` merge; no real diff. Either rebase + scope a PR, or close as superseded by the audit work. |
| `feature/universal-music-input` | 2026-03-22 | 5 | **Independent — needs decision** | Substantive feature ("probabilistic interpretation engine") in 130 files. Either rebase against the post-stack `main`, scope a PR, or close if the work has moved elsewhere. |

## Recommended bulk-close commands (after audit stack lands)

After PRs #149–#167 + #169 are merged, the following branches can be removed:

```bash
# Definitively superseded (9 branches):
for b in \
  cursor/healthkit-bridge-memory-leak-e7cc \
  cursor/dangling-tooltip-singleton-pointer-df13 \
  cursor/wound-struct-aggregate-initialization-3f3c \
  cursor/json-string-escaping-47c9 \
  cursor/namespace-parsing-and-allowlist-2a35 \
  cursor/runtime-synchronization-bugs-c15b \
  cursor/workflow-gates-sanitizers \
  cursor/tsan-preset-frame-pointer-6074 \
  cursor/daiw-core-cmake-sources-465f \
; do
  git push origin --delete "$b"
done
```

For the "verify before closing" group (`cursor/intent-ir-ffi-bridge-issues-4f03`,
`cursor/multiple-code-issues-f237`, `cursor/repository-configuration-and-build-bba6`,
`cursor/uint8-t-pitch-overflow-60a2`):

```bash
git fetch origin <branch>
git log --oneline origin/main..origin/<branch>
git diff origin/main...origin/<branch> -- <suspected-overlap-files>
# If diff is empty / equivalent → push --delete; else cherry-pick the unique
# hunk to a fresh branch, open a small PR, then delete the orphan.
```

For the "needs decision" group (`cursor/rust-test-env-var-race-2d5c`,
`cursor/processor-array-and-cleanup-2c82`, `cursor/daiw-core-cmake-sources-465f`,
`feature/ai-integration-and-readiness`, `feature/universal-music-input`):
human review needed; they touch areas not covered by the audit stack.
