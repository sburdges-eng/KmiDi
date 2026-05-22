---
description: Run KMiDi's verification gates across the 4 layers (TS / Python / C++ / Rust). Auto-detects scope from changed files. Reports per-stack status table.
argument-hint: "[scope: changed | full | ts | python | cpp | rust | rt]"
---

Run KMiDi's verification gates. Scope from `$1`:

- `changed` (default): infer stacks from `git diff --name-only` and run gates only for touched stacks.
- `full`: every gate below — slow (10-20 min on a clean build).
- `ts` | `python` | `cpp` | `rust`: scope to that stack.
- `rt`: ASan/TSan run for RT-touched code (audio path).

## Per-stack gate definitions

```bash
# TS / Frontend
npx tsc --noEmit
npm run build                       # Vite build catches lazy-import / SSR issues

# Python
python3 -m flake8 music_brain/ --max-line-length 100
python3 -m pytest tests/unit -x -q
python3 -m pytest tests/integration -x -q  # only if scope=full

# C++ (release)
cmake --build build -j8
ctest --test-dir build --output-on-failure

# C++ (ASan — required for RT/FFI changes)
cmake --build build-asan -j8
ctest --test-dir build-asan --output-on-failure

# C++ (TSan — separate, mutually exclusive with ASan)
cmake --build build-tsan -j8
ctest --test-dir build-tsan --output-on-failure

# Rust
cd engine/intent_ir && cargo test

# Schema sync (when shared_schemas/ touched)
python3 scripts/sync_entities.py --check
```

## Procedure

1. **Detect stacks** from `git diff --name-only HEAD` (and uncommitted). Map: `*.ts/.tsx` → ts; `music_brain/**`, `tests/**.py`, `python/**` → python; `*.cpp/.h/.hpp`, `CMakeLists.txt`, `engine/**` → cpp; `engine/intent_ir/**`, `*.rs` → rust; `shared_schemas/**` → schema-sync.

2. **Bail early if `external/JUCE/` changed** — that's a vendored submodule, requires human review.

3. **Run gates in order: lint → typecheck → tests → build.** Stop if an earlier gate goes red — don't waste minutes on a build after red tests.

4. **For C++ RT/FFI paths** (any file under `engine/`, `src_penta-core/`, `include/penta/`, `libs/daiw/` that's reachable from a `processBlock` or FFI symbol), include the ASan run.

5. **Schema gate** runs first if `shared_schemas/` touched: `sync_entities.py --check` must pass before any other layer test (otherwise TS/Rust will fail to compile against stale generated code).

## Report

```
| Layer       | Gate          | Status | Time | Notes |
|-------------|---------------|--------|-----:|-------|
| schema      | sync --check  | green  | 1s   | — |
| ts          | tsc --noEmit  | green  | 8s   | — |
| ts          | vite build    | green  | 14s  | — |
| python      | flake8        | green  | 2s   | — |
| python      | pytest unit   | red    | 12s  | test_prrot_bindings: AssertionError @ tests/unit/test_prrot_bindings.py:42 |
| cpp         | cmake build   | skipped | —   | Python red — fix first |
| cpp (asan)  | ctest         | skipped | —   | — |
| rust        | cargo test    | skipped | —   | — |
```

## Hard rules

- **Run from `/Users/seanburdges/Dev/KmiDi`.** If pwd is elsewhere, `cd` and announce.
- **Never auto-fix.** No `clang-format -i`, no `ruff --fix`, no `npm test -- -u`.
- **Read-only.** No commits, no stash, no file modifications.
- **ASan + TSan are mutually exclusive** — run them in separate build dirs (`build-asan`, `build-tsan`). Don't try to enable both in the same configure.
- **Don't run guardrail scripts** (`ci_listening_guardrails.sh`) from `/verify` — those write artifacts. The user runs them explicitly.
- **Long-running gates** (build-asan from clean, full pytest with `tests/integration`): warn before starting, offer `changed` scope first.
- **If `external/JUCE/` is missing** (`-d external/JUCE` fails), report it explicitly and skip C++ gates with a `setup needed: git submodule update --init external/JUCE`.
