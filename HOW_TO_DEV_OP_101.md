# How to Dev/Op 101 (Mac dev staging)

Purpose: fast setup on Apple Silicon for staging KmiDi work (not the canonical monorepo). Mirrors the KMIDI structure; use to rehearse before syncing upstream.

## 0) Prereqs
- macOS 13+ on Apple Silicon (M-series)
- Xcode CLT: `xcode-select --install`
- Homebrew: https://brew.sh
- Python 3.11, Node 18, CMake + Ninja: `brew install python@3.11 node cmake ninja`
- Claude Code extension: install VS Code extension `anthropic.claude-dev` (recommended list is preloaded).

## 1) Clone and env
```bash
git clone <repo-url> kmdi
cd kmdi
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"  # optional lint/test extras
```

## 2) Mac dev config (Apple Silicon)
- Config file: [config/build-dev-mac.yaml](config/build-dev-mac.yaml)
  - `device: mps`, `inference_only: true`, perf target ~150ms/bar
  - Paths honor DATA_ROOT/MODELS_ROOT/OUTPUT_ROOT (defaults to ./data, ./models, ./output)
- Keep training off on Mac (set `training.enabled: false`); use `train-nvidia` build for real fine-tuning.

## 3) Open the preloaded workspace
- Open [KmiDi-dev-mac.code-workspace](KmiDi-dev-mac.code-workspace) in VS Code.
- Settings preloaded: Black on save, pytest (tests + music_brain/tests + ML Kelly Training), flake8, Ninja/CMake, env vars for data/models/output.
- Recommended extensions include Claude Code.

## 4) Data/model layout
```text
./data        # datasets (no PII; use synthetic/redacted)
./models      # checkpoints (Tier 1/2); default checkpoints under models/checkpoints
./output      # logs, reports, generated assets

External SSD (recommended for large sets):
- Mount your drive (e.g., /Volumes/Extreme SSD) and export paths before running:
  - `export DATA_ROOT="/Volumes/Extreme SSD/data"`
  - `export MODELS_ROOT="/Volumes/Extreme SSD/models"`
  - `export OUTPUT_ROOT="/Volumes/Extreme SSD/output"`
- Or place these in `.env` (VS Code picks it up via python.envFile and your shell).
```

## 5) Quick checks
```bash
# Python tests
pytest -v tests || true
pytest -v music_brain/tests || true

# C++ configure/build/test
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
cd build && ctest --output-on-failure || true
```

## 6) Run API/UI locally
```bash
# Start API (example fastapi app if present)
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Web/Tauri smoke
npm install --ignore-scripts --prefix web
npm test -- --watch=false --prefix web || true
npm run tauri dev
```

## 7) Training stubs (Mac)
- Use `scripts/train.py` only for smoke; MPS is fine for tiny batches. For real training, switch to the `train-nvidia` config.
- Set env for paths: `export DATA_ROOT=./data MODELS_ROOT=./models OUTPUT_ROOT=./output`

## 8) CI reference (staging)
- Workflow: [.github/workflows/dev-base-template.yml](.github/workflows/dev-base-template.yml) runs pytest, ctest, and web smoke on push/PR to main/staging/develop.

## 9) Sync discipline
- This repo is staging; mirror approved changes into the canonical monorepo following KMIDI_STRUCTURE_PLAN.md.
- Keep line length ≤100, format with `black`, lint with `flake8`/`mypy` as needed.

## 10) Base ML models (retain stubs, version clearly)
- Registry: [models/registry.json](models/registry.json) (backup: ML Kelly Training/backup/models/registry.json).
- Core IDs (all RTNeural stubs, replace via training outputs):
  - emotionrecognizer (emotion_embedding, 128→64)
  - melodytransformer (melody_generation, 64→128)
  - harmonypredictor (harmony_prediction, 128→64)
  - dynamicsengine (dynamics_mapping, 32→16)
  - groovepredictor (groove_prediction, 64→32)
  - instrumentrecognizer (dual_instrument_recognition, 128→160, dual heads)
  - emotionnodeclassifier (emotion_node_classification, 128→258, multi-head)
- Keep stub JSONs intact; write new exports alongside with run-tagged filenames and update registry entries instead of overwriting blindly.
- For concurrent runs, set distinct `output_dir`/run IDs and point `MODELS_ROOT` to SSD to avoid collisions.

## 11) Training data on SSD (and what if it’s insufficient)
- Point paths to external SSD for datasets/checkpoints:
  - `export DATA_ROOT="/Volumes/Extreme SSD/data"`
  - `export MODELS_ROOT="/Volumes/Extreme SSD/models"`
  - `export OUTPUT_ROOT="/Volumes/Extreme SSD/output"`
- If SSD space is tight:
  - Prune old `output/` artifacts and interim checkpoints; keep `best.pt` and final exports.
  - Use stratified subsets for smoke (e.g., 1–5% of data) and stream from HDD/NAS for full runs.
  - Compress cold checkpoints (xz) and move to archival storage; keep live models under `MODELS_ROOT` only as needed.
  - Prefer `train-mac-smoke` config for MPS, reserve large runs for CUDA hosts.
