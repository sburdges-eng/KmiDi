# KmiDi Unified Repository Structure

**Status**: Planning phase for repository consolidation
**Date**: 2025-12-29
**Objective**: Create unified, organized monorepo from miDiKompanion, kelly-project, and brain-python

---

## Target Directory Structure

```
KmiDi/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Main CI/CD pipeline
│   │   ├── tests.yml              # Unit/integration tests
│   │   ├── performance.yml        # Performance regression
│   │   └── release.yml            # Release automation
│
├── music_brain/                    # Music Intelligence (Python)
│   ├── tier1/                      # Pretrained models
│   │   ├── midi_generator.py
│   │   ├── audio_generator.py
│   │   └── voice_generator.py
│   ├── tier2/                      # LoRA fine-tuning
│   │   └── lora_finetuner.py
│   ├── mac_optimization.py
│   └── examples/
│
├── penta_core/                     # C++ Real-time Engines
│   ├── include/penta/
│   ├── src/
│   ├── python/penta_core/
│   └── tests/
│
├── iDAW_Core/                      # JUCE Plugin Suite
│   ├── plugins/
│   └── shaders/
│
├── mcp_workstation/                # MCP Multi-AI Orchestration
├── mcp_todo/                       # MCP Task Management
│
├── scripts/                        # Command-line tools
│   ├── quickstart_tier1.py
│   └── train_tier2_lora.py
│
├── tests/                          # Testing infrastructure
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── docs/                           # Documentation
│   ├── TIER123_MAC_IMPLEMENTATION.md
│   ├── iDAW_IMPLEMENTATION_GUIDE.md
│   ├── QUICKSTART_TIER123.md
│   └── ARCHITECTURE.md
│
├── config/                         # Hardware-specific configs
│   ├── build-dev-mac.yaml
│   ├── build-train-nvidia.yaml
│   └── build-prod-aws.yaml
│
├── workspaces/                     # VSCode workspaces
│
├── Data_Files/                     # JSON/YAML data
├── Production_Workflows/           # Production guides
├── Songwriting_Guides/             # Songwriting methodology
├── Theory_Reference/               # Music theory
├── vault/                          # Obsidian Knowledge Base
│
├── IMPLEMENTATION_PLAN.md
├── IMPLEMENTATION_ALTERNATIVES.md
├── BUILD_VARIANTS.md
├── README.md
├── Makefile
├── CMakeLists.txt
├── pyproject.toml
├── .gitignore
└── CLAUDE.md
```

---

## Next Steps

1. Create directory structure
2. Organize files into correct locations
3. Create testing infrastructure
4. Set up CI/CD workflows
5. Merge branches cleanly
6. Final commit and push

---

## Execution Roadmap (canonical template for production)

- Scope: This staging repo mirrors the production layout; use it to rehearse workflows before pushing to the canonical monorepo.
- Governance: Follow the structure above; do not add new top-level folders without updating this plan.

### Phase 0 — Environment & Data Readiness (Weeks -2 → 0)
- Provision per BUILD_VARIANTS: dev-mac (MPS), train-nvidia (CUDA), prod-aws (CUDA multi-GPU).
- Verify Tier 1 checkpoints load on MPS and CUDA; log baselines to `output/BASELINE_PERFORMANCE.json`.
- Validate datasets via `scripts/validate_datasets.py`; stash inventories in `data/` and `output/`.
- CI smoke: `pytest -q` for Python, `ctest --output-on-failure` for C++ (skip heavy jobs here; run fully in canonical repo).

### Phase 1 — MVP API + Demo (Weeks 1-4)
- Stand up API in `api/` (FastAPI) wrapping Tier 1; containerize alongside `music_brain/` models.
- Ship demo UI: Streamlit in `app/` or Tauri/web via [src-tauri/](src-tauri/) + [web/](web/); target localhost:8000.
- Collect feedback artifacts in `output/beta/` (sessions, MOS, bug notes); keep PII out of repo.
- Gates: API reachable, demo usable, 10+ therapist sessions logged, MOS ≥ 3.5.

### Phase 2 — Fine-tuning + Validation (Weeks 5-12)
- Run LoRA fine-tuning with `scripts/train_tier2_lora.py`; configs in [config/](config/) per hardware target.
- Store checkpoints under `models/` (ONNX/PT) and report results in `output/reports/TIER2_VALIDATION.md`.
- Expand emotion/genre maps in `data/` and `music_brain/emotion/` with cross-cultural presets; update tests in `tests/`.
- Gates: +0.3 MOS uplift, 100+ sessions, cross-cultural presets reviewed.

### Phase 3 — Clinical/RCT + Launch Prep (Weeks 13-24)
- Document protocol and results in `docs/` (`RCT_PROTOCOL.md`, `RCT_RESULTS.md`); keep de-identified data in `output/clinical/`.
- Package deployable artifacts: Docker (API), binaries (JUCE plugins in [iDAW_Core/](iDAW_Core/)), and wheels (music_brain) under `deployment/`.
- Commercial/readiness docs in `Production_Workflows/` and `docs/`; track partner integrations in `mcp_workstation/` tasks.
- Gates: RCT analysis ready, deployable images built, partner pilot checklists signed.

### Operating Rules
- Staging only: treat changes here as rehearsals; mirror approved changes into the canonical monorepo via documented sync steps.
- Line length ≤100; format with `black`, lint with `flake8`/`mypy`, C++ via clang-format if configured.
- No new data with PII; use synthetic or redacted samples under `output/`.
- CI / QA hooks:
	- VS Code tasks: `verify` runs `mvn -B verify`; `test` runs `mvn -B test` (use Run Task in VS Code).
	- Python: `pytest -v` (root and music_brain/tests); C++: `ctest --output-on-failure` from build dir.
	- Frontend/Tauri: `npm test` or `npm run tauri dev` for smoke; keep API target at 127.0.0.1:8000.

