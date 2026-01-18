```markdown
# Build & Smoke Plan (All Components)

## Scope
- Workspace: /Users/seanburdges/.cursor/worktrees/KmiDi-1/tmb
- Targets: Python (mcp_workstation, music_brain, penta_core), C++ (iDAW_Core/penta_core JUCE build), Training pipelines (KmiDi_TRAINING).

## Plan
1) **Env prep (once)**
   - Ensure Homebrew deps on macOS: `brew install cmake ninja pkg-config python@3.11` (plus `lcov` if coverage needed).
   - Create venv `python3 -m venv .venv && source .venv/bin/activate`.
   - Upgrade pip/setuptools/wheel; install repo root in editable mode with dev extras: `pip install -e ".[dev]"`.
   - Optional: install training extras if required (`pip install -e ".[train]"` or training/requirements.txt if present).

2) **Python sanity (mcp_workstation / music_brain / penta_core)**
   - Run import smoke: `python - <<'PY'` to import key packages (`mcp_workstation`, `music_brain`, `penta_core`).
   - Run targeted tests if available: `pytest tests -q` and `pytest tests_music-brain -q` (adjust paths if different).

3) **C++ build (penta_core / JUCE components)**
   - From workspace root: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPENTA_BUILD_TESTS=ON -DPENTA_BUILD_JUCE_PLUGIN=OFF` (toggle plugin ON if needed).
   - Build: `cmake --build build --target penta_core penta_tests -j`.
   - Test: `ctest --output-on-failure` (in `build`).

4) **Training pipelines (KmiDi_TRAINING)**
   - Install training deps if separate (check `training/requirements*.txt`); ensure CUDA not required for CPU smoke.
   - Run quick smoke: `python training/training/cuda_session/train_midi_generator.py --help` (or a dry-run flag if available).
   - Optional: run a minimal CPU-only unit/integration test if provided.

5) **Report & next steps**
   - Capture any failures (Python imports/tests, CMake configure/build, ctest, training smoke).
   - If blockers occur (missing imports, build config), document fixes before rerunning.
```
