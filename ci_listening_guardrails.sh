#!/usr/bin/env bash
# ci_listening_guardrails.sh
#
# Scans vetted source directories for hardcoded /Users/ paths that indicate
# developer-local absolute paths accidentally committed to the repo.
#
# Deliberately excludes archival/generated/vendored trees that already contain
# such strings (e.g. KmiDi_FINAL/, KmiDi/external/JUCE) so that legitimate
# source additions are the only thing that can trigger a failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Directories to scan (canonical source paths only)
# ---------------------------------------------------------------------------
SOURCE_DIRS=(
  music_brain
  src_penta-core
  penta_core
  iDAW_Core
  mcp_workstation
  mcp_todo
  "src-tauri/src"
  "web/src"
  scripts
  tests
  include
  bindings
)

# ---------------------------------------------------------------------------
# File extensions to inspect
# ---------------------------------------------------------------------------
EXTENSIONS=(
  "*.cpp" "*.h" "*.hpp"
  "*.py"
  "*.rs"
  "*.ts" "*.tsx" "*.js" "*.jsx"
  "*.sh"
)

# Build ripgrep glob flags
GLOB_FLAGS=()
for ext in "${EXTENSIONS[@]}"; do
  GLOB_FLAGS+=(--glob "$ext")
done

# ---------------------------------------------------------------------------
# Collect only the source dirs that actually exist in this checkout
# ---------------------------------------------------------------------------
EXISTING_DIRS=()
for d in "${SOURCE_DIRS[@]}"; do
  [[ -d "$d" ]] && EXISTING_DIRS+=("$d")
done

if [[ ${#EXISTING_DIRS[@]} -eq 0 ]]; then
  echo "WARNING: No source directories found; skipping forbidden-path scan."
  exit 0
fi

echo "==> Scanning for hardcoded /Users/ paths in: ${EXISTING_DIRS[*]}"

# Run ripgrep; collect matches (exit 0 = found, exit 1 = no match, exit 2 = error)
# Exclude check_external_dependencies.py because it is a detection script that
# intentionally contains the patterns it scans for.
MATCHES=$(rg --no-heading --line-number "/Users/" \
  "${GLOB_FLAGS[@]}" \
  --glob '!check_external_dependencies.py' \
  "${EXISTING_DIRS[@]}" 2>/dev/null || true)

if [[ -n "$MATCHES" ]]; then
  echo ""
  echo "ERROR: Hardcoded /Users/ path(s) found in source files:"
  echo "$MATCHES"
  echo ""
  echo "Remove or replace these absolute Mac paths before merging."
  exit 1
fi

echo "OK: No hardcoded /Users/ paths found in source files."
