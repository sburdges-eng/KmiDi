---
description: Security + RT/FFI audit of the current branch vs main. Uses global audit skill plus the KMiDi audit overlay (RT safety, FFI ownership, JUCE/Qt link discipline, ODR, GIL, schema drift, dataset path leakage).
argument-hint: "[base-branch]"
---

Run a security + safety audit of the current branch.

**Base:** `$1` if provided, else `main` (fall back to `master`).

Invoke the **global `audit` skill** (path traversal, SSRF, secrets, SQLi, XSS, command injection, crypto misuse) and **the KMiDi `audit-kmidi` overlay** (RT-thread allocations, FFI ownership, JUCE/Qt PRIVATE link, ODR / allocator mismatch, GIL, sanitizer exclusivity, schema sync, dataset path leakage, app entrypoint drift, legacy-path encroachment) in a single pass.

## Procedure

1. Compute the diff: `git diff $(git merge-base HEAD main)..HEAD` plus working tree + staged.
2. Walk the diff. For each hunk, run both rule sets (global + KMiDi).
3. For C++ changes touching RT or FFI, optionally use `nm` / `otool -L` / `readelf -d` (read-only) to verify link discipline:
   - `nm build/libKellyFFI.dylib | grep -i juce` (JUCE symbols should be local, not exported)
   - Check that no executable in the build graph links both KellyFFI and JUCE directly.
4. For schema changes (`shared_schemas/` touched), check that `src/types/Intent.ts`, `engine/intent_ir/src/generated/intent.rs`, and Python validation are regenerated in the same diff.
5. Print findings as `| Severity | File:Line | Issue | Suggested fix |`. Tag KMiDi-specific findings with `(kmidi)`.

## Hard rules

- **Read-only.** No `git apply`, `clang-format -i`, no commits.
- **Don't run guardrail scripts** (they write artifacts).
- **Don't run sanitizer builds** as part of the audit — those belong in `/verify rt`.
- **Don't bump JUCE or any submodule.**
- If no findings: state explicitly `**No exploitable issues found in <N> changed files.**`. Don't pad.
- Hand the report to the user. Do **not** apply fixes.
