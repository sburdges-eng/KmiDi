#!/usr/bin/env bash
# One-command dev environment setup for KmiDi.
# Run from repo root. After this, use: npm run dev:all (or npm run dev / npm run dev:python).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=> KmiDi v1 dev setup (bootstrap + npm + pip)..."

# 1) Bootstrap (JUCE submodule, CMake/Node checks, pybind11 hint)
"$SCRIPT_DIR/bootstrap.sh"

# 2) Node deps (React frontend tooling)
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
echo "   React + API: npm run dev:all"
echo "   Or: npm run dev (React) | npm run dev:python (API on :8000)"
echo "   Note: package.json does not currently define npm run dev:tauri."
