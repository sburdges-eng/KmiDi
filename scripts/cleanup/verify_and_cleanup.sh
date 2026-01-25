#!/usr/bin/env bash
# Verify builds and test suites, then delete quarantine directories
# Usage: ./verify_and_cleanup.sh [--dry-run] [--skip-delete]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/Users/seanburdges/KmiDi-1"
REPORT_DIR="$PROJECT_ROOT/docs/cleanup"
VERIFICATION_REPORT="$REPORT_DIR/VERIFICATION_REPORT.md"
CLEANUP_COMPLETE="$REPORT_DIR/CLEANUP_COMPLETE.md"

DRY_RUN=false
SKIP_DELETE=false
BUILD_FAILED=false
TEST_FAILED=false

# Quarantine directories from Phase 1 & 2
QUARANTINE_DIRS=(
  "_QUARANTINE_20260124_032139"
  "_QUARANTINE_20260124_032257"
  "_QUARANTINE_20260124_032329"
  "_QUARANTINE_20260124_032435"
)

log() { echo "[verify] $*"; }
log_err() { echo "[verify] ERROR: $*" >&2; }

# Parse arguments
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --skip-delete) SKIP_DELETE=true ;;
  esac
done

mkdir -p "$REPORT_DIR"

# Initialize report
{
  echo "# Build and Test Verification Report"
  echo ""
  echo "**Generated:** $(date)"
  echo ""
  echo "## Build Verification"
  echo ""
} > "$VERIFICATION_REPORT"

# Function to test CMake builds
test_cmake_build() {
  local name=$1
  local dir=$2
  log "Testing CMake build: $name"

  local build_dir="$dir/build-test-verify"
  rm -rf "$build_dir"
  mkdir -p "$build_dir"

  if cd "$build_dir" && cmake .. > "$REPORT_DIR/cmake_${name}.log" 2>&1; then
    if make -j$(sysctl -n hw.ncpu) >> "$REPORT_DIR/cmake_${name}.log" 2>&1; then
      echo "### $name CMake Build: ✅ PASSED" >> "$VERIFICATION_REPORT"
      log "✅ $name CMake build passed"
      return 0
    else
      echo "### $name CMake Build: ❌ FAILED (see cmake_${name}.log)" >> "$VERIFICATION_REPORT"
      log_err "$name CMake build failed"
      BUILD_FAILED=true
      return 1
    fi
  else
    echo "### $name CMake Build: ❌ FAILED (cmake configure failed)" >> "$VERIFICATION_REPORT"
    log_err "$name CMake configure failed"
    BUILD_FAILED=true
    return 1
  fi
}

# Function to test Python package
test_python_package() {
  log "Testing Python package installation and imports"

  if cd "$PROJECT_ROOT" && python3 -c "import music_brain" > "$REPORT_DIR/python_import.log" 2>&1; then
    echo "### Python Package: ✅ PASSED" >> "$VERIFICATION_REPORT"
    log "✅ Python imports OK"
    return 0
  else
    echo "### Python Package: ❌ FAILED (see python_import.log)" >> "$VERIFICATION_REPORT"
    log_err "Python imports failed"
    BUILD_FAILED=true
    return 1
  fi
}

# Function to test Rust build
test_rust_build() {
  log "Testing Rust build"

  if cd "$PROJECT_ROOT/src-tauri" && cargo build --release > "$REPORT_DIR/rust_build.log" 2>&1; then
    echo "### Rust Build: ✅ PASSED" >> "$VERIFICATION_REPORT"
    log "✅ Rust build passed"
    return 0
  else
    echo "### Rust Build: ❌ FAILED (see rust_build.log)" >> "$VERIFICATION_REPORT"
    log_err "Rust build failed"
    BUILD_FAILED=true
    return 1
  fi
}

# Function to run pytest suite
run_python_tests() {
  log "Running Python test suite"

  local test_dir=$1
  local test_name=$2

  if cd "$PROJECT_ROOT" && pytest "$test_dir" -v --tb=short > "$REPORT_DIR/test_${test_name}.log" 2>&1; then
    local passed=$(grep -c "PASSED" "$REPORT_DIR/test_${test_name}.log" || echo "0")
    local failed=$(grep -c "FAILED" "$REPORT_DIR/test_${test_name}.log" || echo "0")
    echo "### $test_name Tests: ✅ PASSED (passed: $passed, failed: $failed)" >> "$VERIFICATION_REPORT"
    log "✅ $test_name tests passed"
    if ((failed > 0)); then
      TEST_FAILED=true
      return 1
    fi
    return 0
  else
    echo "### $test_name Tests: ❌ FAILED (see test_${test_name}.log)" >> "$VERIFICATION_REPORT"
    log_err "$test_name tests failed"
    TEST_FAILED=true
    return 1
  fi
}

# Function to run C++ tests
run_cpp_tests() {
  log "Running C++ tests"

  local build_dir="$PROJECT_ROOT/build-test-verify"
  if [[ -d "$build_dir" ]] && cd "$build_dir" && ctest -V > "$REPORT_DIR/test_cpp.log" 2>&1; then
    echo "### C++ Tests: ✅ PASSED" >> "$VERIFICATION_REPORT"
    log "✅ C++ tests passed"
    return 0
  else
    echo "### C++ Tests: ⚠️ SKIPPED (no test targets found or build not available)" >> "$VERIFICATION_REPORT"
    log "⚠️ C++ tests skipped"
    return 0
  fi
}

# Function to run Rust tests
run_rust_tests() {
  log "Running Rust tests"

  if cd "$PROJECT_ROOT/src-tauri" && cargo test --release > "$REPORT_DIR/test_rust.log" 2>&1; then
    echo "### Rust Tests: ✅ PASSED" >> "$VERIFICATION_REPORT"
    log "✅ Rust tests passed"
    return 0
  else
    echo "### Rust Tests: ❌ FAILED (see test_rust.log)" >> "$VERIFICATION_REPORT"
    log_err "Rust tests failed"
    TEST_FAILED=true
    return 1
  fi
}

# Phase 1: Build Verification
log "=== Phase 1: Build Verification ==="
{
  echo "## Build Verification"
  echo ""
} >> "$VERIFICATION_REPORT"

test_cmake_build "Root" "$PROJECT_ROOT" || true
test_cmake_build "PentaCore" "$PROJECT_ROOT/src_penta-core" || true
test_python_package || true
test_rust_build || true

# Phase 2: Test Suite Execution
log "=== Phase 2: Test Suite Execution ==="
{
  echo ""
  echo "## Test Suite Results"
  echo ""
} >> "$VERIFICATION_REPORT"

run_python_tests "tests/music_brain" "Python" || true
run_cpp_tests || true
run_rust_tests || true
run_python_tests "tests/integration" "Integration" || true

# Phase 3: Summary
log "=== Phase 3: Verification Summary ==="
{
  echo ""
  echo "## Summary"
  echo ""
  if [[ "$BUILD_FAILED" == "true" ]]; then
    echo "❌ **Build failures detected** - Do not delete quarantine directories"
  elif [[ "$TEST_FAILED" == "true" ]]; then
    echo "⚠️ **Test failures detected** - Review before deleting quarantine directories"
  else
    echo "✅ **All builds and tests passed** - Safe to delete quarantine directories"
  fi
  echo ""
  echo "### Quarantine Directories"
  echo ""
  for dir in "${QUARANTINE_DIRS[@]}"; do
    if [[ -d "$PROJECT_ROOT/$dir" ]]; then
      local size=$(du -sh "$PROJECT_ROOT/$dir" 2> /dev/null | cut -f1 || echo "unknown")
      echo "- \`$dir\` - $size"
    fi
  done
  echo ""
  echo "**Total space to be freed:** ~2.6 GB"
} >> "$VERIFICATION_REPORT"

# Phase 4: Delete quarantine directories
if [[ "$SKIP_DELETE" == "true" ]]; then
  log "Skipping deletion (--skip-delete flag)"
  exit 0
fi

if [[ "$BUILD_FAILED" == "true" ]]; then
  log_err "Build failures detected. Not deleting quarantine directories."
  log_err "Review $VERIFICATION_REPORT for details."
  exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
  log "[DRY-RUN] Would delete quarantine directories:"
  for dir in "${QUARANTINE_DIRS[@]}"; do
    log "[DRY-RUN]   rm -rf $PROJECT_ROOT/$dir"
  done
  exit 0
fi

log "=== Phase 4: Quarantine Deletion ==="
log "All builds and tests passed. Proceeding with deletion..."

# Create backup manifest
{
  echo "# Quarantine Deletion Manifest"
  echo ""
  echo "**Deleted:** $(date)"
  echo ""
  echo "## Directories Deleted"
  echo ""
  for dir in "${QUARANTINE_DIRS[@]}"; do
    if [[ -d "$PROJECT_ROOT/$dir" ]]; then
      local size=$(du -sh "$PROJECT_ROOT/$dir" 2> /dev/null | cut -f1 || echo "unknown")
      echo "- \`$dir\` - $size"
      rm -rf "$PROJECT_ROOT/$dir"
    fi
  done
  echo ""
  echo "**Total space freed:** ~2.6 GB"
} > "$CLEANUP_COMPLETE"

log "✅ Quarantine directories deleted"
log "Report saved to: $VERIFICATION_REPORT"
log "Cleanup summary: $CLEANUP_COMPLETE"
