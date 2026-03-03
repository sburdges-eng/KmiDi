#!/usr/bin/env bash
# One-command dev environment setup for KmiDi v1.
# Run from repo root. After this, use: npm run dev:all (or npm run dev / npm run tauri dev).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=> KmiDi v1 dev setup (bootstrap + npm + pip)..."

# 1) Bootstrap (JUCE submodule, CMake/Node checks, pybind11 hint)
bash ./bootstrap.sh

# 2) Node deps (Tauri + React)
if command -v npm >/dev/null 2>&1; then
  echo "-> npm install..."
  npm install
else
  echo "WARNING: npm not found. Install Node 20+ and run: npm install"
fi

# 3) Python env (music_brain API, sync_entities, tests)
if command -v python3 >/dev/null 2>&1; then
  echo "-> pip install -e . (music_brain)..."
  python3 -m pip install -e . --quiet
  python3 -m pip install pydantic uvicorn --quiet 2>/dev/null || true
else
  echo "WARNING: python3 not found. Install Python 3.11+ and run: pip install -e ."
fi

echo "=> Dev setup complete."
echo "   Start full stack: npm run dev:all"
echo "   Or: npm run dev (React) | npm run tauri dev (Tauri) | npm run dev:python (API on :8000)"
