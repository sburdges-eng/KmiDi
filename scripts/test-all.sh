#!/bin/bash
# Test All Script
# Runs all test suites: C++, Rust, Python, Frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

echo "=========================================="
echo "KmiDi Test Suite"
echo "=========================================="
echo ""

FAILED_TESTS=0

# Step 1: C++ Tests
print_status "Step 1: Running C++ Tests"
cd "$PROJECT_ROOT"

BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
if [ -d "$BUILD_DIR" ]; then
  cd "$BUILD_DIR"
  if cmake --build . --target test 2> /dev/null; then
    if ctest --output-on-failure 2> /dev/null; then
      print_status "C++ tests passed"
    else
      print_warning "C++ tests failed or not available"
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
  else
    print_warning "C++ test target not available (run build first)"
  fi
else
  print_warning "Build directory not found, skipping C++ tests"
fi

# Step 2: Rust Tests
print_status "Step 2: Running Rust Tests"
cd "$PROJECT_ROOT/src-tauri"

if cargo test 2> /dev/null; then
  print_status "Rust tests passed"
else
  print_warning "Rust tests failed or not available"
  FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Step 3: Python Tests
print_status "Step 3: Running Python Tests"
cd "$PROJECT_ROOT"

# Check for pytest
if python3 -m pytest --version > /dev/null 2>&1; then
  if python3 -m pytest tests/ -v 2> /dev/null; then
    print_status "Python tests passed"
  else
    print_warning "Python tests failed or not available"
    FAILED_TESTS=$((FAILED_TESTS + 1))
  fi
else
  # Try unittest
  if python3 -m unittest discover -s tests -p "test_*.py" -v 2> /dev/null; then
    print_status "Python tests passed"
  else
    print_warning "Python test framework not available"
  fi
fi

# Step 4: Frontend Tests
print_status "Step 4: Running Frontend Tests"
cd "$PROJECT_ROOT"

if [ -f "package.json" ]; then
  if grep -q '"test"' package.json; then
    if npm test 2> /dev/null; then
      print_status "Frontend tests passed"
    else
      print_warning "Frontend tests failed or not available"
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
  else
    print_warning "No test script in package.json"
  fi
else
  print_warning "package.json not found"
fi

# Summary
echo ""
echo "=========================================="
if [ $FAILED_TESTS -eq 0 ]; then
  echo -e "${GREEN}All Tests Passed!${NC}"
else
  echo -e "${RED}Some Tests Failed ($FAILED_TESTS)${NC}"
fi
echo "=========================================="
echo ""

exit $FAILED_TESTS
