# Project Error Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring all build/test/lint layers to green — fix the TypeScript build blocker, pytest configuration conflict, missing test dependencies, ML test collection guards, and C++ warnings.

**Architecture:** Five independent fixes across TS, Python config, Python deps, Python test guards, and C++ headers. No cross-task dependencies — each task produces a green delta on its own.

**Tech Stack:** TypeScript/Vite, Python/pytest/pyproject.toml, C++20

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/vite-env.d.ts` | Vite `ImportMeta.env` type augmentation |
| Delete | `pytest.ini` | Remove duplicate config; consolidate into pyproject.toml |
| Modify | `pyproject.toml:31-40` | Add `httpx` to dev deps, merge pytest.ini settings |
| Modify | `tests/unit/test_emotion_probe.py:1-5` | Add `importorskip("torch")` guard |
| Modify | `tests/unit/test_jepa_models.py:1-5` | Add `importorskip("torch")` guard |
| Modify | `tests/unit/test_export_audio_jepa.py:1-10` | Add `importorskip("torch")` + `importorskip("onnx")` guards |
| Modify | `include/penta/groove/GrooveEngine.h:50-54,87` | Fix deleted move ops, suppress `lastAnalysisPosition_` warning |
| Modify | `include/penta/groove/OnsetDetector.h:73-75` | Suppress unused-field warnings from src_penta-core divergence |
| Modify | `include/penta/mixer/MixerEngine.h:277` | Suppress unused `sampleRate_` warning |

---

### Task 1: Fix TypeScript build — add Vite env type declaration

The frontend build (`npm run build`) and `npx tsc --noEmit` both fail because `import.meta.env` has no type definition. Vite projects need a `vite-env.d.ts` reference file.

**Files:**
- Create: `src/vite-env.d.ts`

- [ ] **Step 1: Create the type declaration file**

```typescript
/// <reference types="vite/client" />
```

- [ ] **Step 2: Verify TypeScript type-check passes**

Run: `cd /Users/seanburdges/Dev/KMiDi && npx tsc --noEmit`
Expected: exit 0, no errors

- [ ] **Step 3: Verify frontend build passes**

Run: `cd /Users/seanburdges/Dev/KMiDi && npm run build`
Expected: exit 0, produces `dist/` output

- [ ] **Step 4: Commit**

```bash
git add src/vite-env.d.ts
git commit -m "fix(ts): add vite-env.d.ts to resolve import.meta.env types

Vite augments ImportMeta with env property but TypeScript needs the
reference type declaration to see it. Without this, tsc and the
production build both fail on useMusicBrain.ts.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix pytest configuration conflict — consolidate into pyproject.toml

pytest.ini and pyproject.toml both define pytest settings. pytest.ini wins, silently ignoring `asyncio_mode = "auto"` from pyproject.toml. Merge the pytest.ini content into pyproject.toml and delete pytest.ini.

**Files:**
- Delete: `pytest.ini`
- Modify: `pyproject.toml:65-67`

- [ ] **Step 1: Update pyproject.toml with merged pytest config**

Replace the current `[tool.pytest.ini_options]` section (lines 65-67) with the full merged config:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--disable-warnings",
    "--color=yes",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
    "cpp: Tests requiring C++ bindings",
]
log_cli = false
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"
minversion = "3.8"
```

- [ ] **Step 2: Delete pytest.ini**

```bash
rm pytest.ini
```

- [ ] **Step 3: Verify pytest discovers tests and asyncio_mode is active**

Run: `.venv/bin/python -m pytest tests/ -q --co 2>&1 | head -5`
Expected: shows collected tests, no "WARNING: ignoring pytest config in pyproject.toml"

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/unit/test_emotion_probe.py --ignore=tests/unit/test_export_audio_jepa.py --ignore=tests/unit/test_jepa_models.py -x 2>&1 | tail -5`
Expected: passes (same as before, minus collection errors)

- [ ] **Step 4: Commit**

```bash
git rm pytest.ini
git add pyproject.toml
git commit -m "fix(test): consolidate pytest config into pyproject.toml

pytest.ini was silently overriding pyproject.toml, which meant
asyncio_mode = 'auto' was never active. Merge all settings into
the single pyproject.toml source and delete the duplicate file.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add httpx to dev dependencies

`test_api_audit_fixes.py` uses `starlette.testclient.TestClient` which requires `httpx`. All 11 tests error at fixture setup.

**Files:**
- Modify: `pyproject.toml:31-40`

- [ ] **Step 1: Add httpx to dev dependencies**

In `pyproject.toml` under `[project.optional-dependencies]`, add `httpx` to the `dev` list:

```toml
dev = [
    "pybind11>=2.10",
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "flake8>=6.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "httpx>=0.24",
]
```

- [ ] **Step 2: Install updated dev deps in venv**

Run: `.venv/bin/pip install -e ".[dev]"`
Expected: httpx installed successfully

- [ ] **Step 3: Verify test_api_audit_fixes passes**

Run: `.venv/bin/python -m pytest tests/unit/test_api_audit_fixes.py -q`
Expected: 11 passed, 0 errors

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "fix(deps): add httpx to dev dependencies for TestClient

starlette.testclient requires httpx at runtime. The 11 tests in
test_api_audit_fixes.py all errored because httpx was missing from
the dev extras.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add importorskip guards to ML-dependent tests

Three test files import `torch` at module level, causing collection errors when torch is not installed (CI, lightweight envs). One also needs `onnx`/`onnxruntime` guards.

**Files:**
- Modify: `tests/unit/test_emotion_probe.py:1-5`
- Modify: `tests/unit/test_jepa_models.py:1-5`
- Modify: `tests/unit/test_export_audio_jepa.py:1-10`

- [ ] **Step 1: Guard test_emotion_probe.py**

Replace the top of the file:

```python
"""Tests for EmotionProbe model."""

import pytest

torch = pytest.importorskip("torch")

from music_brain.jepa.emotion_probe import EmotionProbe
```

(Remove the bare `import torch` line; the `pytest.importorskip` both imports and skips if missing.)

- [ ] **Step 2: Guard test_jepa_models.py**

Replace the top imports — change `import torch` to:

```python
import pytest

torch = pytest.importorskip("torch")
```

Keep all other imports after this guard.

- [ ] **Step 3: Guard test_export_audio_jepa.py**

Replace the top imports — change `import torch` to the importorskip form. Also add guards for onnx tests. At the module level:

```python
import pytest

torch = pytest.importorskip("torch")
```

For the three onnx-specific test methods, add a skip decorator or inline guard:

```python
onnx = pytest.importorskip("onnx")
```

at the top of each onnx test method, or use `@pytest.mark.skipif` with an import check. The simplest approach: add module-level guards for both:

```python
torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
```

This skips the entire file when either is missing, which is acceptable since the torch tests and onnx tests are tightly coupled in this file.

- [ ] **Step 4: Verify clean collection without torch**

Run: `python3 -m pytest tests/ --co -q 2>&1 | tail -5`
Expected: no collection errors (the torch/onnx files are skipped, not errored)

- [ ] **Step 5: Verify tests still pass with torch (in venv if torch is available)**

Run: `.venv/bin/python -m pytest tests/unit/test_emotion_probe.py tests/unit/test_jepa_models.py -q 2>&1`
Expected: either passes or skips cleanly (depends on torch availability in venv)

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_emotion_probe.py tests/unit/test_jepa_models.py tests/unit/test_export_audio_jepa.py
git commit -m "fix(test): add importorskip guards for torch/onnx tests

These three test files imported torch at module level, breaking
pytest collection in environments without ML dependencies. Use
pytest.importorskip so they skip cleanly instead of erroring.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Fix C++ compiler warnings

Eight warnings from three headers. Root cause: `src_penta-core/` implementations diverged from `src/` (known consolidation debt — see `docs/CODEAUDIT_CXX_DEEP_2026Q2.md`). The penta-core .cpp files don't use all fields declared in the shared headers.

**Approach:** Suppress with `[[maybe_unused]]` where the field is intentionally unused by one implementation but used by the other. Fix the deleted-move-ops warning by removing the `= default` declarations (the class is non-movable due to `std::atomic<bool>` + `std::unique_ptr` members).

**Files:**
- Modify: `include/penta/groove/GrooveEngine.h:50-54,87`
- Modify: `include/penta/groove/OnsetDetector.h:73-75`
- Modify: `include/penta/mixer/MixerEngine.h:277`

- [ ] **Step 1: Fix GrooveEngine.h — remove broken default move ops**

Replace lines 50-54:

```cpp
    // Non-copyable, non-movable (std::atomic is not movable)
    GrooveEngine(const GrooveEngine&) = delete;
    GrooveEngine& operator=(const GrooveEngine&) = delete;
    GrooveEngine(GrooveEngine&&) = delete;
    GrooveEngine& operator=(GrooveEngine&&) = delete;
```

- [ ] **Step 2: Fix GrooveEngine.h — suppress lastAnalysisPosition_ warning**

Change line 87:

```cpp
    [[maybe_unused]] uint64_t lastAnalysisPosition_;  // Used by src/ impl; unused by src_penta-core impl
```

- [ ] **Step 3: Fix OnsetDetector.h — suppress diverged fields**

Change lines 73-75:

```cpp
        [[maybe_unused]] size_t fluxHistoryIndex_ = 0;   // Used by src/ impl (ring-buffer); unused by src_penta-core (rotate)
        [[maybe_unused]] size_t fluxHistoryCount_ = 0;   // Used by src/ impl
        [[maybe_unused]] float lastFlux_ = 0.0f;         // Used by src/ impl
```

- [ ] **Step 4: Fix MixerEngine.h — suppress unused sampleRate_**

Change line 277:

```cpp
    [[maybe_unused]] double sampleRate_;
```

- [ ] **Step 5: Verify clean C++ build**

Run: `cd /Users/seanburdges/Dev/KMiDi && cmake --build build -j8 2>&1 | grep -E "warning|error" | grep -v "ranlib:"`
Expected: no warnings (ranlib warnings about empty .o files are harmless and unrelated)

- [ ] **Step 6: Commit**

```bash
git add include/penta/groove/GrooveEngine.h include/penta/groove/OnsetDetector.h include/penta/mixer/MixerEngine.h
git commit -m "fix(cpp): resolve 8 compiler warnings in penta headers

- GrooveEngine: delete move ops instead of defaulting (std::atomic
  is not movable). Suppress lastAnalysisPosition_ unused warning
  (field used by src/ impl, not src_penta-core — consolidation debt).
- OnsetDetector: suppress 3 unused-field warnings from src_penta-core
  divergence (ring-buffer fields used by src/ impl only).
- MixerEngine: suppress unused sampleRate_ (stored at construction,
  never read — reserved for future sample-rate-dependent processing).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Post-fix verification

After all five tasks, run the full suite:

```bash
# TypeScript
npx tsc --noEmit && npm run build

# Python (in venv)
.venv/bin/python -m pytest tests/ -q

# C++
cmake --build build -j8 2>&1 | grep -E "warning|error" | grep -v "ranlib:"
```

Expected: all green. The only remaining test failures should be the 3 `test_export_audio_jepa` onnx tests (skipped if onnx not installed) — these are optional ML deps, not project errors.
