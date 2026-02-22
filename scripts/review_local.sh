#!/usr/bin/env bash
# review_local.sh — Run lint, format, and test checks locally before pushing.
# Usage:  ./scripts/review_local.sh [--fix]
#
# With --fix, auto-format Python files instead of only checking.
set -uo pipefail

FIX=false
if [[ "${1:-}" == "--fix" ]]; then
  FIX=true
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

PASS=0
FAIL=0
WARN=0

report() {
  local status="$1" label="$2"
  if [[ "$status" -eq 0 ]]; then
    echo "  ✅  $label"
    PASS=$((PASS + 1))
  else
    echo "  ❌  $label"
    FAIL=$((FAIL + 1))
  fi
}

warn() {
  echo "  ⚠️   $1"
  WARN=$((WARN + 1))
}

echo "═══════════════════════════════════════"
echo " KmiDi Local Review"
echo "═══════════════════════════════════════"
echo ""

# ── Python formatting ──────────────────────────────────────────
echo "── Python Formatting ──"
if command -v black &>/dev/null; then
  if $FIX; then
    black --line-length=100 music_brain/ tests/ scripts/*.py 2>/dev/null
    report 0 "black (auto-fixed)"
  else
    black --check --diff --line-length=100 music_brain/ tests/ scripts/*.py 2>/dev/null
    report $? "black --check"
  fi
else
  warn "black not installed — skipping formatting check"
fi

# ── Python linting ─────────────────────────────────────────────
echo "── Python Linting ──"
if command -v flake8 &>/dev/null; then
  flake8 music_brain/ tests/ scripts/*.py --max-line-length=100 --extend-ignore=E203,W503 2>/dev/null
  report $? "flake8"
else
  warn "flake8 not installed — skipping lint check"
fi

# ── Python type check ──────────────────────────────────────────
echo "── Python Type Check ──"
if command -v mypy &>/dev/null; then
  mypy music_brain/ --ignore-missing-imports 2>/dev/null || true
  report 0 "mypy (informational)"
else
  warn "mypy not installed — skipping type check"
fi

# ── Python unit tests ──────────────────────────────────────────
echo "── Python Unit Tests ──"
if command -v pytest &>/dev/null; then
  pytest tests/unit/ -v --tb=short -q 2>/dev/null
  report $? "pytest tests/unit/"
else
  warn "pytest not installed — skipping unit tests"
fi

# ── Git checks ─────────────────────────────────────────────────
echo "── Git Checks ──"
if git diff --cached --name-only | grep -qE '\.env$|\.env\.'; then
  echo "  ❌  Staged .env file detected — do not commit secrets"
  FAIL=$((FAIL + 1))
else
  report 0 "No .env files staged"
fi

LARGE_FILES=$(git diff --cached --numstat | awk '$1 > 1000 || $2 > 1000 {print $3}')
if [[ -n "$LARGE_FILES" ]]; then
  warn "Large staged diffs detected: $LARGE_FILES"
else
  report 0 "No oversized staged diffs"
fi

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════"
echo " Results: ✅ $PASS passed  ❌ $FAIL failed  ⚠️  $WARN warnings"
echo "═══════════════════════════════════════"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
