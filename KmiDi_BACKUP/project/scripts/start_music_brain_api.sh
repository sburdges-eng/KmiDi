#!/usr/bin/env bash
# Start the Music Brain API server used by the Tauri desktop app.
# Tries api_server first, then falls back to `python -m music_brain.api`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
echo "[music_brain] Using Python: ${PYTHON_BIN}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[music_brain] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("music_brain.api_server") else 1)
PY
then
  CMD=("$PYTHON_BIN" "-m" "music_brain.api_server")
  echo "[music_brain] Starting Music Brain API via music_brain.api_server"
else
  CMD=("$PYTHON_BIN" "-m" "music_brain.api")
  echo "[music_brain] Starting Music Brain API via music_brain.api"
fi

echo "[music_brain] Working directory: ${ROOT}"
exec "${CMD[@]}"

