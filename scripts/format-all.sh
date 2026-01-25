#!/bin/bash
# =============================================================================
# Format All Code - Unified Formatting Script
# =============================================================================
# Formats all code in the KmiDi project using the appropriate formatters:
# - Python: ruff format + ruff check
# - TypeScript/TSX: prettier
# - Rust: cargo fmt
# - C++: clang-format (if available)
# - Shell scripts: shfmt (if available)
# =============================================================================

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Formatting All Code in KmiDi Project${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Track what was formatted
FORMATTED=()

# =============================================================================
# Python Formatting (ruff format + ruff check)
# =============================================================================
echo -e "${YELLOW}[1/5] Formatting Python code with ruff...${NC}"

if command -v ruff &> /dev/null; then
  # Python directories to format
  PYTHON_DIRS=(
    "music_brain"
    "scripts"
    "training"
    "penta_core"
    "kelly"
    "tests/python"
    "examples"
    "daiw_mcp"
    "mcp_penta_swarm"
    "mcp_todo"
    "mcp_workstation"
    "python"
    "tools"
  )

  # Format Python code
  echo "  → Running ruff format..."
  if ruff format "${PYTHON_DIRS[@]}" 2> /dev/null; then
    FORMATTED+=("Python (ruff format)")
    echo -e "  ${GREEN}✓${NC} Python formatting complete"
  else
    echo -e "  ${RED}✗${NC} Python formatting failed (ruff format)"
  fi

  # Check Python code (linting)
  echo "  → Running ruff check..."
  if ruff check "${PYTHON_DIRS[@]}" --fix 2> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Python linting complete"
  else
    echo -e "  ${YELLOW}⚠${NC} Python linting found issues (check output above)"
  fi
else
  echo -e "  ${RED}✗${NC} ruff not found. Install with: pip install ruff"
fi

echo ""

# =============================================================================
# TypeScript/React Formatting (prettier)
# =============================================================================
echo -e "${YELLOW}[2/5] Formatting TypeScript/React code with prettier...${NC}"

if command -v prettier &> /dev/null || [ -f "node_modules/.bin/prettier" ]; then
  PRETTIER_CMD="prettier"
  if [ -f "node_modules/.bin/prettier" ]; then
    PRETTIER_CMD="node_modules/.bin/prettier"
  fi

  echo "  → Formatting TypeScript/TSX files..."
  if $PRETTIER_CMD --write 'src/**/*.{ts,tsx}' 'tests/**/*.{ts,tsx}' 2> /dev/null; then
    FORMATTED+=("TypeScript/React (prettier)")
    echo -e "  ${GREEN}✓${NC} TypeScript/React formatting complete"
  else
    echo -e "  ${YELLOW}⚠${NC} TypeScript/React formatting completed with warnings"
  fi
else
  echo -e "  ${RED}✗${NC} prettier not found. Install with: npm install"
fi

echo ""

# =============================================================================
# Rust Formatting (cargo fmt)
# =============================================================================
echo -e "${YELLOW}[3/5] Formatting Rust code with cargo fmt...${NC}"

if command -v cargo &> /dev/null && [ -d "src-tauri" ]; then
  echo "  → Formatting Rust files..."
  if (cd src-tauri && cargo fmt 2> /dev/null); then
    FORMATTED+=("Rust (cargo fmt)")
    echo -e "  ${GREEN}✓${NC} Rust formatting complete"
  else
    echo -e "  ${YELLOW}⚠${NC} Rust formatting completed with warnings"
  fi
else
  if ! command -v cargo &> /dev/null; then
    echo -e "  ${RED}✗${NC} cargo not found. Install Rust toolchain first"
  else
    echo -e "  ${YELLOW}⚠${NC} src-tauri directory not found, skipping Rust formatting"
  fi
fi

echo ""

# =============================================================================
# C++ Formatting (clang-format) - Optional
# =============================================================================
echo -e "${YELLOW}[4/5] Formatting C++ code with clang-format (optional)...${NC}"

if command -v clang-format &> /dev/null; then
  echo "  → Finding C++ files..."

  # C++ directories to format
  CPP_DIRS=(
    "src"
    "src_penta-core"
    "bindings"
    "benchmarks"
    "cpp_music_brain"
    "iDAW_Core"
  )

  # Find all C++ files
  CPP_FILES=$(find "${CPP_DIRS[@]}" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) 2> /dev/null | head -100)

  if [ -n "$CPP_FILES" ]; then
    echo "  → Formatting C++ files..."
    echo "$CPP_FILES" | while read -r file; do
      if [ -f "$file" ]; then
        clang-format -i "$file" 2> /dev/null || true
      fi
    done
    FORMATTED+=("C++ (clang-format)")
    echo -e "  ${GREEN}✓${NC} C++ formatting complete"
  else
    echo -e "  ${YELLOW}⚠${NC} No C++ files found to format"
  fi
else
  echo -e "  ${YELLOW}⚠${NC} clang-format not found. Install with: brew install clang-format (macOS) or apt-get install clang-format (Linux)"
  echo "     Skipping C++ formatting..."
fi

echo ""

# =============================================================================
# Shell Script Formatting (shfmt) - Optional
# =============================================================================
echo -e "${YELLOW}[5/5] Formatting shell scripts with shfmt (optional)...${NC}"

if command -v shfmt &> /dev/null; then
  echo "  → Finding shell script files..."

  # Shell script directories to format
  SHELL_DIRS=(
    "scripts"
    ".github"
  )

  # Find all shell script files
  SHELL_FILES=$(find "${SHELL_DIRS[@]}" -type f \( -name "*.sh" -o -name "*.bash" \) 2> /dev/null | head -100)

  if [ -n "$SHELL_FILES" ]; then
    echo "  → Formatting shell script files..."
    echo "$SHELL_FILES" | while read -r file; do
      if [ -f "$file" ]; then
        shfmt -w -i 2 -ci -sr "$file" 2> /dev/null || true
      fi
    done
    FORMATTED+=("Shell scripts (shfmt)")
    echo -e "  ${GREEN}✓${NC} Shell script formatting complete"
  else
    echo -e "  ${YELLOW}⚠${NC} No shell script files found to format"
  fi
else
  echo -e "  ${YELLOW}⚠${NC} shfmt not found. Install with: brew install shfmt (macOS) or go install mvdan.cc/sh/v3/cmd/shfmt@latest"
  echo "     Skipping shell script formatting..."
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Summary
if [ ${#FORMATTED[@]} -gt 0 ]; then
  echo -e "${GREEN}✓ Formatting complete!${NC}"
  echo ""
  echo "Formatted languages:"
  for lang in "${FORMATTED[@]}"; do
    echo -e "  ${GREEN}•${NC} $lang"
  done
else
  echo -e "${RED}✗ No formatters were successfully run.${NC}"
  echo ""
  echo "Please ensure the following are installed:"
  echo "  • ruff: pip install ruff"
  echo "  • prettier: npm install"
  echo "  • cargo: Install Rust toolchain"
  echo "  • clang-format: brew install clang-format (optional)"
  echo "  • shfmt: brew install shfmt (optional)"
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
