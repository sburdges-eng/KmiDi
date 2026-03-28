# KmiDi - Project Context

**Emotion-Driven Music Intelligence & Audio Workstation**
A multi-language platform (Python, C++, Rust, React/Tauri) for generative audio/MIDI.

## Parallel Workflow — Gemini + Claude

Gemini CLI and Claude Code work in parallel on this project.

**Gemini's role:** Planning, code review, proposing changes, exploration, research.
**Claude's role:** Heavy coding implementation, all approvals.

---

## Gemini Tools & Permissions

### Full Tool Access List

You have access to the following tools. Use them freely.

| Tool | Name in Policy | What It Does |
|------|---------------|--------------|
| Read files | `read_file` | Read any file in the project |
| List directories | `list_directory` | List contents of any directory |
| Search content | `grep_search` | Search file contents by regex pattern |
| Find files | `glob` | Find files by name/path pattern |
| Run commands | `run_shell_command` | Execute shell commands (see prefixes below) |
| Write files | `write_file` | Create/overwrite files (plans & docs only) |
| Edit files | `replace` | Edit file contents (plans & docs only) |
| Save memory | `save_memory` | Persist context across sessions |
| Ask user | `ask_user` | Ask clarifying questions |
| Track tasks | `write_todos` | Create and manage todo items |
| Web search | `google_web_search` | Search the web for references |

### Shell Command Prefixes (All Allowed)

Every command below is pre-approved in the policy. Use without hesitation.

**Navigation & Inspection:**
`ls`, `cat`, `head`, `tail`, `find`, `pwd`, `cd`, `which`, `command`, `file`, `nm`, `df`, `du`

**Search & Text Processing:**
`grep`, `rg`, `wc`, `diff`, `sort`, `uniq`, `echo`

**Git (all subcommands):**
`git log`, `git status`, `git diff`, `git show`, `git branch`, `git blame`, `git stash list`
Note: `git commit`, `git push`, `git reset`, `git checkout` are allowed by the policy but **should go through Claude** per workflow rules.

**Build & Test:**
`python3`, `/Users/seanburdges/Dev/KmiDi/venv/bin/python`, `cmake`, `npm`, `pip`

**System:**
`bash`, `source`, `export`, `chmod`, `tmux`, `ps`, `kill`, `sleep`, `true`

### Access Scope

- **Full read access** to everything inside `/Users/seanburdges/Dev/KmiDi/`
- **Write access** for plan/doc files in `docs/superpowers/plans/`
- **All plan files** are at `docs/superpowers/plans/` (NOT in `~/.cursor/plans/`)

### What You Should Do Autonomously

1. Read and explore the entire codebase — any file, any directory
2. Run any shell command from the prefixes above
3. Run tests: `python3 -m pytest`, `npm test`, `cmake --build`
4. Run git read commands: `git log`, `git status`, `git diff`, `git show`
5. Search with `grep_search`, `glob`, `rg`, `find`
6. Write plans to `docs/superpowers/plans/`
7. Review code and flag issues
8. Propose changes as diffs/suggestions with file paths
9. Save memories about what you learn
10. Search the web for technical references

### What Requires Claude Approval

- `git commit`, `git push`, `git reset`, `git checkout` (branch-changing ops)
- Creating or modifying source files (`.py`, `.cpp`, `.h`, `.ts`, `.rs`, `.cmake`)
- Deleting files or directories
- Installing packages (`pip install X`, `npm install X`)
- Any destructive or irreversible operation

**Workflow:** Describe proposed changes clearly with file paths and diffs. Claude will review, implement, and commit.

---

## Master Plan

**Authoritative plan:** `docs/superpowers/plans/claud_completion_master_plan.md`

### Completed
- Phase 1a: mcp_workstation importlib facade
- Phase 1b: penta_core importlib facade
- Phase 1c: ComprehensiveIntegrationManager
- Phase 2: C++ build fix (227/227 targets)
- Phase 4a: PRROT Python bridge

### Remaining — Phases & Tasks

**Phase 3 — Real-Time Engine & behavior_lab Integration**
- [ ] C++ FFI Extension: Add `RTState` polling/callback endpoints to `KellyFFI.cpp` (BPM, position, emotion, track params)
- [ ] Python Bridge: Extend `ComprehensiveIntegrationManager` to consume `RTState` at ~10-50Hz
- [ ] behavior_lab Wiring: Connect `behavior_lab.runner.run_scenario` to bridge; listeners for "New Bar" / "Emotion Change"; wire `ClosedLoopController` to adjust `penta_core` params
- [ ] RT Safety Audit: No allocations/locks in audio thread; `readerwriterqueue` for all cross-thread comms

**Phase 4b — DAW & Voice Bridge Wiring**
- [ ] `send_to_daw(daw_name, midi_data)` in `ComprehensiveIntegrationManager` (Ableton via OSC, Logic via `LogicOSCBridge`)
- [ ] PRROT Voice: `pybind11` bindings for `src/prrot/`; wire into `AIOrchestrator` + `IntegrationBridge`; fix broken import path

**Phase 5 — Training Infrastructure Consolidation**
- [ ] JEPA: Replace stubs in `training/` with actual logic from `KmiDi_CANON.training.train_jepa`
- [ ] Emotion Model: Implement `train_emotion_optimized.py` using `~/Datasets/kmidi_emotion`
- [ ] Model Registry: Wire checkpoints (`~/Models/checkpoints/`) to `penta_core.ml.inference`

**Technical Debt — Types & Imports**
- [ ] Deploy `KellyTypes.h` to `src/common/` + `TypeAdapter.h`; update `KellyBrain`/`MLBridge`
- [ ] Top-level re-export packages for `mcp_workstation/` and `penta_core/`
- [ ] Add `KellyBrain.cpp`/`MLBridge.cpp` to `CMakeLists.txt`; link JUCE + pybind11

**Low-Latency UI — FFI/Tauri**
- [ ] Direct C FFI: `kelly_ffi.h/cpp` exposing `KellyBrain` to Rust
- [ ] Tauri Commands: Update `src-tauri/src/commands.rs` to call C++ via FFI (bypass Python HTTP)

### Risk Mitigations
| Risk | Mitigation |
|------|------------|
| Python/C++ Latency | Shared memory or high-speed OSC/FFI. No JSON in hot path. |
| Circular Dependencies | `KellyTypes.h` pattern + forward declarations |
| Import Path Fragility | `mcp_workstation` facades at root |
| RT Safety Violations | `thread_local` storage, pre-allocated buffers, `rt_harness` benchmarks |

### Plan Files (`docs/superpowers/plans/`)
- **`claud_completion_master_plan.md`** — master synthesis (this section mirrors it)
- `phase3-realtime-bridge.md` — Phase 3 implementation detail
- `phase_3_todo.md` — Phase 3 task checklist
- `kmidi_integration_gaps.plan.md` — original gap analysis
- `kellybrain_ml_integration_30ad0a80.plan.md` — ML integration plan
- `kmidi_multi-technology_integration_eab3ba5f.plan.md` — multi-tech integration
- `logic_kmidi_bidirectional_integration_66e5b0ee.plan.md` — Logic Pro bridge
- `kmidi_workspace_integration_f95796ef.plan.md` — workspace integration
- `kmidi_roadmap_execution_plan_de9fe8a4.plan.md` — roadmap execution
- `gemini-onboarding-plan.md` — Gemini tool access & workflow guide

---

## Architecture

- **`music_brain/`**: Python ML core. 50+ modules (emotion analysis, groove generation, harmony, arrangement). FastAPI-based API.
- **`KellyCore`**: C++ static library (JUCE-based). Pure DSP and audio processing logic.
- **`KellyFFI`**: Shared C-ABI bridge linking the C++ engine to the Tauri host.
- **`src-tauri/`**: Rust-based Tauri desktop host.
- **`apps/kmidi/`**: Python app (config, data, experiments, tests, roadmap).
- **`frontend/`**: React-based UI for the desktop application.
- **`plugin/`**: JUCE-based VST3/CLAP plugins.
- **`engine/`**: Core C ABI for RT harness and FFI.
- **`rt_harness/`**: Headless real-time callback runner for performance profiling (P50/P90/P99 stats).
- **`libs/jepa/`**: JEPA real-time AI — proposed for C++ core integration (low-latency inference).
- **`libs/ai_core/`**: Shared AI utilities and abstractions.

## Building & Running

### Python ML Core (`music_brain`)
- **Setup**: `pip install -e ".[dev,audio]"`
- **Run API**: `./start-api.sh` or `uvicorn music_brain.api:app --reload`

### Frontend & Desktop Host (Tauri)
- **Install**: `npm install`
- **Dev**: `npm run dev`
- **Build**: `./scripts/build-full-stack.sh` (Pipeline B)

### C++ Core & Plugins (CMake)
- **Configure**: `cmake -B build -G Ninja -DBUILD_PLUGINS=ON -DBUILD_DESKTOP=ON`
- **Build**: `cmake --build build`
- **Options**: `ENABLE_RTNEURAL`, `ENABLE_ONNX_RUNTIME` for ML inference in C++.

## Testing

- **Python**: `pytest` (Tests located in `tests/`)
- **Frontend**: `npm test`
- **C++**: `ctest` (requires `-DBUILD_TESTS=ON`) or `build/KellyTests`
- **Performance**: `build/KellyFFIBenchmark`

## Development Conventions

1. **Safety**: Always verify the C++ bridge (KellyFFI) when changing the engine ABI.
2. **ML Inference**: Python is the source of truth for models; export to ONNX for C++ usage.
3. **Real-Time Safety**: C++ code (especially in `engine/` and `src_penta-core/`) must be RT-safe (no allocations, no locks). Use `readerwriterqueue`.
4. **Emotions**: The emotion engine is the primary driver for all generative tasks (groove, harmony, etc.).
5. **std::optional pattern**: MIDI layers use `std::optional<vector<MidiNote>>`. Check with `hasMidiLayer()`, unwrap with `midiLayerOrEmpty()` or `.value_or({})`.

## Key Directories
- `music_brain/`: The "Intelligence" (NLP, Emotion, Harmony).
- `src/`: C++ source for plugins and engine.
- `docs/`: Full documentation, including [ADR](docs/adr/) and [ARCHITECTURE.md](docs/ARCHITECTURE.md).
- `docs/plans/`: Feature development plans (consolidated from Claude plans).
- `docs/integration/`: Integration summaries and task artifacts.
- `scripts/`: Build and automation scripts.
- `bindings/`: Python bindings for the C++ penta-core.
- `libs/jepa/`: JEPA RT AI (proposed for C++ core).
- `libs/ai_core/`: Shared AI utilities.
- `apps/kmidi/`: Python app package (config, data, experiments, tests).
