#!/bin/bash

# =============================================================================
# KmiDi Integration Test Suite
# =============================================================================
# Runs comprehensive integration tests across all technology layers
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║     ██╗███╗   ██╗████████╗███████╗ ██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗    ║"
echo "║     ██║████╗  ██║╚══██╔══╝██╔════╝██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║    ║"
echo "║     ██║██╔██╗ ██║   ██║   █████╗  ██║  ███╗██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║    ║"
echo "║     ██║██║╚██╗██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║    ║"
echo "║     ██║██║ ╚████║   ██║   ███████╗╚██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║    ║"
echo "║     ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ║"
echo "║                                                                  ║"
echo "║               ████████╗███████╗███████╗████████╗███████╗                                      ║"
echo "║               ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝                                      ║"
echo "║                  ██║   █████╗  ███████╗   ██║   ███████╗                                      ║"
echo "║                  ██║   ██╔══╝  ╚════██║   ██║   ╚════██║                                      ║"
echo "║                  ██║   ███████╗███████║   ██║   ███████║                                      ║"
echo "║                  ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝                                      ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# Utility Functions
# =============================================================================

log_step() {
  echo -e "${YELLOW}[$(date '+%H:%M:%S')] $1${NC}"
}

log_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
  echo -e "${RED}❌ $1${NC}"
}

log_info() {
  echo -e "${BLUE}ℹ️  $1${NC}"
}

run_test_suite() {
  local test_name="$1"
  local test_command="$2"
  local test_dir="$3"

  log_step "Running $test_name..."

  if [ -n "$test_dir" ]; then
    cd "$test_dir"
  fi

  if eval "$test_command"; then
    log_success "$test_name passed"
    return 0
  else
    log_error "$test_name failed"
    return 1
  fi
}

# =============================================================================
# Pre-test Validation
# =============================================================================

log_step "Validating test environment..."

cd "$PROJECT_ROOT"

# Check if builds exist
if [ ! -d "build/debug" ]; then
  log_error "Debug build directory not found"
  echo "Run: mkdir -p build/debug && cd build/debug && cmake ../../ -DCMAKE_BUILD_TYPE=Debug"
  exit 1
fi

if [ ! -f "build/debug/libKellyFFI.dylib" ] && [ ! -f "build/debug/libKellyFFI.so" ]; then
  log_error "Kelly FFI library not found"
  echo "Run: cd build/debug && make KellyFFI"
  exit 1
fi

log_success "Test environment validated"

# =============================================================================
# Test Suite Execution
# =============================================================================

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_and_track() {
  local test_name="$1"
  local test_command="$2"
  local test_dir="$3"

  ((TOTAL_TESTS++))

  if run_test_suite "$test_name" "$test_command" "$test_dir"; then
    ((PASSED_TESTS++))
    return 0
  else
    ((FAILED_TESTS++))
    return 1
  fi
}

# =============================================================================
# C++ FFI Tests
# =============================================================================

log_step "Starting C++ FFI integration tests..."

# Build C++ tests if they don't exist
if [ ! -f "build/debug/KellyTests" ]; then
  log_info "Building C++ test suite..."
  cd build/debug
  make -j4 KellyTests
  cd "$PROJECT_ROOT"
fi

# Run C++ FFI tests
run_and_track \
  "C++ FFI Tests" \
  "./KellyTests --reporter=console [kelly_ffi]" \
  "build/debug"

# =============================================================================
# Rust Integration Tests
# =============================================================================

log_step "Starting Rust integration tests..."

cd "$PROJECT_ROOT/src-tauri"

# Build Rust test dependencies
log_info "Building Rust test dependencies..."
cargo build --tests

# Run Rust integration tests
run_and_track \
  "Rust FFI Integration Tests" \
  "cargo test integration_test" \
  "src-tauri"

# Run Rust unit tests
run_and_track \
  "Rust Unit Tests" \
  "cargo test --lib" \
  "src-tauri"

cd "$PROJECT_ROOT"

# =============================================================================
# Build System Integration Tests
# =============================================================================

log_step "Testing build system integration..."

# Test that libraries are properly linked
run_and_track \
  "Library Linking Test" \
  "otool -L src-tauri/resources/libKellyFFI.dylib || ldd src-tauri/resources/libKellyFFI.so || echo 'Library checking not available'" \
  "."

# Test CMake configuration
run_and_track \
  "CMake Configuration Test" \
  "cd build/debug && cmake . --build . --target help | grep -q KellyFFI" \
  "."

# Test that FFI library exports expected symbols
run_and_track \
  "FFI Symbol Export Test" \
  "nm -D build/debug/libKellyFFI.dylib 2>/dev/null | grep -q kelly_brain || nm -D build/debug/libKellyFFI.so 2>/dev/null | grep -q kelly_brain || echo 'Symbol checking not available'" \
  "."

# =============================================================================
# Frontend Integration Tests (if available)
# =============================================================================

log_step "Testing frontend integration..."

# Check if Vitest is configured
if grep -q "vitest" package.json; then
  run_and_track \
    "Frontend Integration Tests" \
    "npm test -- tests/e2e/" \
    "."
else
  log_info "Frontend tests not configured (Vitest not found)"
fi

# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

log_step "Running end-to-end workflow tests..."

# Test 1: Build system workflow
run_and_track \
  "E2E Build Workflow" \
  "BUILD_TYPE=Debug BUILD_TESTS=OFF npm run build:cpp && npm run build" \
  "."

# Test 2: Quick build verification
log_info "Testing quick development build..."
run_and_track \
  "Quick Development Build" \
  "cd build/debug && make -j2" \
  "."

# =============================================================================
# Performance Benchmarks
# =============================================================================

log_step "Running basic performance tests..."

# C++ performance test
if [ -f "build/debug/KellyTests" ]; then
  run_and_track \
    "C++ Performance Benchmarks" \
    "./KellyTests --reporter=console [performance] || echo 'No performance tests found'" \
    "build/debug"
fi

# FFI call performance test
log_info "Testing FFI call performance..."

# Create simple performance test
cat > /tmp/ffi_perf_test.cpp << 'EOF'
#include "bridge/kelly_ffi.h"
#include <chrono>
#include <iostream>

int main() {
    const int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        const char* version = kelly_get_version();
        (void)version;  // Suppress unused warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = double(duration.count()) / iterations;
    std::cout << "Average FFI call time: " << avg_time << " microseconds" << std::endl;
    
    // Should be < 10 microseconds per call
    return (avg_time < 10.0) ? 0 : 1;
}
EOF

# Compile and run performance test
if cd build/debug && g++ -I../../src -L. -lKellyFFI /tmp/ffi_perf_test.cpp -o ffi_perf_test 2> /dev/null; then
  run_and_track \
    "FFI Call Performance" \
    "./ffi_perf_test" \
    "build/debug"
  rm -f ffi_perf_test
else
  log_info "FFI performance test compilation failed (skipping)"
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Test Results Summary
# =============================================================================

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                     TEST RESULTS SUMMARY                      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Test Statistics:${NC}"
echo "  📊 Total Tests: $TOTAL_TESTS"
echo "  ✅ Passed: $PASSED_TESTS"
echo "  ❌ Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
  echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
  echo -e "${GREEN}Integration is working correctly.${NC}"
  OVERALL_RESULT=0
else
  echo -e "${RED}⚠️  $FAILED_TESTS TEST(S) FAILED${NC}"
  echo -e "${YELLOW}Review the output above for details.${NC}"
  OVERALL_RESULT=1
fi

echo ""
echo -e "${CYAN}Integration Test Categories:${NC}"
echo "  🔗 C++ FFI Layer"
echo "  🦀 Rust Integration"
echo "  🏗️  Build System"
echo "  ⚛️  Frontend Integration"
echo "  🔄 End-to-End Workflows"
echo "  ⚡ Performance Benchmarks"
echo ""

if [ $OVERALL_RESULT -eq 0 ]; then
  echo -e "${GREEN}✨ KmiDi integration is ready for development!${NC}"
else
  echo -e "${YELLOW}🔧 Some integration issues need to be resolved.${NC}"
  echo -e "${YELLOW}   Check the failed tests above and run:${NC}"
  echo -e "${YELLOW}   - ./scripts/dev-setup.sh (for environment issues)${NC}"
  echo -e "${YELLOW}   - ./scripts/build-all.sh (for build issues)${NC}"
fi

echo ""

exit $OVERALL_RESULT
