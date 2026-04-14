#!/usr/bin/env bash
# Per-edit verify for KMiDi. Called by ~/.claude/hooks/verify-after-edit.sh.
# $1 = edited file. Must be fast — runs after every Write|Edit.
# Strategy: incremental compile only (ninja is a no-op if nothing changed).
# Full ctest is too slow for per-edit; run it manually via bench scripts.

set -u
FILE="${1:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

case "$FILE" in
  *.cpp|*.cc|*.cxx|*.h|*.hpp|*.inl|*CMakeLists.txt|*.cmake)
    ;;
  *)
    exit 0
    ;;
esac

case "$FILE" in
  */build/*|*/build-*/*|*/_archive/*|*/external/*|*/_deps/*)
    exit 0
    ;;
esac

[ -d "$REPO/build" ] || { echo "no build/ dir — run cmake configure first"; exit 0; }

cd "$REPO" || exit 0
cmake --build build -j 2>&1 | tail -60
exit "${PIPESTATUS[0]}"
