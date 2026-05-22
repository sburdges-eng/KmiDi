#!/usr/bin/env bash
# Find references to a symbol across KmiDi repo (for "what depends on X" discovery).
# Usage: ./scripts/docs/find_deps.sh <symbol> [scope]
#   scope: music_brain | src | engine/intent_ir | cpp | all (default: all)
# Requires: rg (ripgrep) on PATH.
# Output: file:line matches, then a markdown table row template (Name | What it does | What calls it | What it calls) for 10_Dependency_Map.md.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SYMBOL="${1:?Usage: $0 <symbol> [scope]}"
SCOPE="${2:-all}"

# Escape for basic regex (literal match); allow word-boundary style.
escape_regex() {
  printf '%s' "$1" | sed 's/[.[\*^$()+?{|]/\\&/g'
}
PATTERN="$(escape_regex "$SYMBOL")"
# Prefer word-boundary match when symbol looks like an identifier.
if printf '%s' "$SYMBOL" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_]*$'; then
  PATTERN="\\b${PATTERN}\\b"
fi

run_rg() {
  rg --line-number --no-heading "$PATTERN" "$@" 2>/dev/null || true
}

echo "=== References to: $SYMBOL (scope: $SCOPE) ==="
echo ""

case "$SCOPE" in
  music_brain)
    run_rg music_brain/
    ;;
  src)
    run_rg src/
    ;;
  engine/intent_ir)
    run_rg engine/intent_ir/
    ;;
  cpp)
    run_rg src_penta-core/ include/ src/bridge/
    ;;
  all)
    run_rg music_brain/ src/ engine/intent_ir/ src_penta-core/ include/ src/bridge/
    ;;
  *)
    echo "Unknown scope: $SCOPE" >&2
    echo "Use: music_brain | src | engine/intent_ir | cpp | all" >&2
    exit 1
    ;;
esac

echo ""
echo "=== Table row template for docs/KmiDi_101/10_Dependency_Map.md ==="
echo "| **<name>** | <What it does> | <What calls it> | <What it calls> |"
