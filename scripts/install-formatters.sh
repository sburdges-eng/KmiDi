#!/bin/bash
# =============================================================================
# Install All Formatters - Setup Script
# =============================================================================
# Installs all code formatters required for the KmiDi project:
# - Python: ruff
# - TypeScript/TSX: prettier (via npm)
# - Rust: cargo fmt (via rustup)
# - C++: clang-format (optional)
# - Shell scripts: shfmt (optional)
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
echo -e "${BLUE}  Installing All Code Formatters for KmiDi${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

INSTALLED=()
SKIPPED=()
FAILED=()

# =============================================================================
# Python: ruff
# =============================================================================
echo -e "${YELLOW}[1/5] Installing ruff (Python formatter)...${NC}"

if command -v ruff &> /dev/null; then
  RUFF_VERSION=$(ruff --version 2> /dev/null || echo "unknown")
  echo -e "  ${GREEN}✓${NC} ruff already installed: $RUFF_VERSION"
  SKIPPED+=("ruff")
else
  if command -v pip &> /dev/null; then
    echo "  → Installing ruff via pip..."
    if pip install --upgrade ruff 2> /dev/null; then
      INSTALLED+=("ruff")
      echo -e "  ${GREEN}✓${NC} ruff installed successfully"
    else
      FAILED+=("ruff")
      echo -e "  ${RED}✗${NC} Failed to install ruff"
    fi
  elif command -v pip3 &> /dev/null; then
    echo "  → Installing ruff via pip3..."
    if pip3 install --upgrade ruff 2> /dev/null; then
      INSTALLED+=("ruff")
      echo -e "  ${GREEN}✓${NC} ruff installed successfully"
    else
      FAILED+=("ruff")
      echo -e "  ${RED}✗${NC} Failed to install ruff"
    fi
  else
    FAILED+=("ruff")
    echo -e "  ${RED}✗${NC} pip/pip3 not found. Install Python/pip first"
  fi
fi

echo ""

# =============================================================================
# TypeScript/React: prettier (via npm)
# =============================================================================
echo -e "${YELLOW}[2/5] Installing prettier (TypeScript/React formatter)...${NC}"

if command -v prettier &> /dev/null || [ -f "node_modules/.bin/prettier" ]; then
  if command -v prettier &> /dev/null; then
    PRETTIER_VERSION=$(prettier --version 2> /dev/null || echo "unknown")
    echo -e "  ${GREEN}✓${NC} prettier already installed globally: $PRETTIER_VERSION"
  else
    echo -e "  ${GREEN}✓${NC} prettier already installed locally (node_modules)"
  fi
  SKIPPED+=("prettier")
else
  if command -v npm &> /dev/null; then
    echo "  → Installing prettier via npm..."
    if npm install --save-dev prettier 2> /dev/null; then
      INSTALLED+=("prettier")
      echo -e "  ${GREEN}✓${NC} prettier installed successfully"
    else
      FAILED+=("prettier")
      echo -e "  ${RED}✗${NC} Failed to install prettier"
    fi
  else
    FAILED+=("prettier")
    echo -e "  ${RED}✗${NC} npm not found. Install Node.js/npm first"
    echo "     Install Node.js from: https://nodejs.org/"
  fi
fi

echo ""

# =============================================================================
# Rust: cargo fmt (via rustup)
# =============================================================================
echo -e "${YELLOW}[3/5] Installing Rust toolchain (cargo fmt)...${NC}"

if command -v cargo &> /dev/null; then
  RUST_VERSION=$(rustc --version 2> /dev/null || echo "unknown")
  echo -e "  ${GREEN}✓${NC} Rust toolchain already installed: $RUST_VERSION"

  # Check if rustfmt is installed
  if command -v rustfmt &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} rustfmt component already installed"
  else
    echo "  → Installing rustfmt component..."
    if rustup component add rustfmt 2> /dev/null; then
      INSTALLED+=("rustfmt")
      echo -e "  ${GREEN}✓${NC} rustfmt component installed"
    else
      echo -e "  ${YELLOW}⚠${NC} Could not install rustfmt component (may already be installed)"
    fi
  fi
  SKIPPED+=("rust")
else
  echo "  → Rust toolchain not found. Installing via rustup..."

  if command -v rustup &> /dev/null; then
    echo "  → Installing Rust via rustup..."
    if rustup install stable 2> /dev/null && rustup default stable 2> /dev/null; then
      INSTALLED+=("rust")
      if rustup component add rustfmt 2> /dev/null; then
        INSTALLED+=("rustfmt")
      fi
      echo -e "  ${GREEN}✓${NC} Rust toolchain installed successfully"
    else
      FAILED+=("rust")
      echo -e "  ${RED}✗${NC} Failed to install Rust toolchain"
    fi
  else
    echo "  → rustup not found. Installing rustup..."
    echo "  → Run this command to install rustup:"
    echo -e "     ${BLUE}curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
    echo ""
    read -p "  → Would you like to install rustup now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; then
        source "$HOME/.cargo/env" 2> /dev/null || true
        if rustup component add rustfmt 2> /dev/null; then
          INSTALLED+=("rust")
          INSTALLED+=("rustfmt")
          echo -e "  ${GREEN}✓${NC} Rust toolchain installed successfully"
        fi
      else
        FAILED+=("rust")
        echo -e "  ${RED}✗${NC} Failed to install rustup"
      fi
    else
      FAILED+=("rust")
      echo -e "  ${YELLOW}⚠${NC} Skipping Rust installation"
    fi
  fi
fi

echo ""

# =============================================================================
# C++: clang-format (optional)
# =============================================================================
echo -e "${YELLOW}[4/5] Installing clang-format (C++ formatter - optional)...${NC}"

if command -v clang-format &> /dev/null; then
  CLANG_VERSION=$(clang-format --version 2> /dev/null | head -n1 || echo "unknown")
  echo -e "  ${GREEN}✓${NC} clang-format already installed: $CLANG_VERSION"
  SKIPPED+=("clang-format")
else
  echo "  → clang-format not found. Installing..."

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
      echo "  → Installing clang-format via Homebrew..."
      if brew install clang-format 2> /dev/null; then
        INSTALLED+=("clang-format")
        echo -e "  ${GREEN}✓${NC} clang-format installed successfully"
      else
        FAILED+=("clang-format")
        echo -e "  ${RED}✗${NC} Failed to install clang-format"
      fi
    else
      FAILED+=("clang-format")
      echo -e "  ${RED}✗${NC} Homebrew not found. Install Homebrew first:"
      echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    fi
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
      echo "  → Installing clang-format via apt-get..."
      if sudo apt-get update && sudo apt-get install -y clang-format 2> /dev/null; then
        INSTALLED+=("clang-format")
        echo -e "  ${GREEN}✓${NC} clang-format installed successfully"
      else
        FAILED+=("clang-format")
        echo -e "  ${RED}✗${NC} Failed to install clang-format"
      fi
    elif command -v yum &> /dev/null; then
      echo "  → Installing clang-format via yum..."
      if sudo yum install -y clang-tools-extra 2> /dev/null; then
        INSTALLED+=("clang-format")
        echo -e "  ${GREEN}✓${NC} clang-format installed successfully"
      else
        FAILED+=("clang-format")
        echo -e "  ${RED}✗${NC} Failed to install clang-format"
      fi
    else
      FAILED+=("clang-format")
      echo -e "  ${RED}✗${NC} Package manager not found (apt-get/yum)"
    fi
  else
    FAILED+=("clang-format")
    echo -e "  ${YELLOW}⚠${NC} Unsupported OS. Please install clang-format manually"
  fi
fi

echo ""

# =============================================================================
# Shell Scripts: shfmt (optional)
# =============================================================================
echo -e "${YELLOW}[5/5] Installing shfmt (shell script formatter - optional)...${NC}"

if command -v shfmt &> /dev/null; then
  SHFMT_VERSION=$(shfmt --version 2> /dev/null || echo "unknown")
  echo -e "  ${GREEN}✓${NC} shfmt already installed: $SHFMT_VERSION"
  SKIPPED+=("shfmt")
else
  echo "  → shfmt not found. Installing..."

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
      echo "  → Installing shfmt via Homebrew..."
      if brew install shfmt 2> /dev/null; then
        INSTALLED+=("shfmt")
        echo -e "  ${GREEN}✓${NC} shfmt installed successfully"
      else
        FAILED+=("shfmt")
        echo -e "  ${RED}✗${NC} Failed to install shfmt"
      fi
    else
      FAILED+=("shfmt")
      echo -e "  ${RED}✗${NC} Homebrew not found. Install Homebrew first:"
      echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
      echo ""
      echo "  Or install via Go: go install mvdan.cc/sh/v3/cmd/shfmt@latest"
    fi
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - try Go install first, then package manager
    if command -v go &> /dev/null; then
      echo "  → Installing shfmt via Go..."
      if go install mvdan.cc/sh/v3/cmd/shfmt@latest 2> /dev/null; then
        INSTALLED+=("shfmt")
        echo -e "  ${GREEN}✓${NC} shfmt installed successfully"
      else
        FAILED+=("shfmt")
        echo -e "  ${RED}✗${NC} Failed to install shfmt via Go"
      fi
    elif command -v apt-get &> /dev/null; then
      echo "  → Installing shfmt via apt-get..."
      if sudo apt-get update && sudo apt-get install -y shfmt 2> /dev/null; then
        INSTALLED+=("shfmt")
        echo -e "  ${GREEN}✓${NC} shfmt installed successfully"
      else
        FAILED+=("shfmt")
        echo -e "  ${RED}✗${NC} Failed to install shfmt (may not be in repos)"
        echo "     Try: go install mvdan.cc/sh/v3/cmd/shfmt@latest"
      fi
    else
      FAILED+=("shfmt")
      echo -e "  ${RED}✗${NC} Package manager not found. Install Go and run:"
      echo "     go install mvdan.cc/sh/v3/cmd/shfmt@latest"
    fi
  else
    FAILED+=("shfmt")
    echo -e "  ${YELLOW}⚠${NC} Unsupported OS. Install via Go:"
    echo "     go install mvdan.cc/sh/v3/cmd/shfmt@latest"
  fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Summary
echo -e "${BLUE}Installation Summary:${NC}"
echo ""

if [ ${#INSTALLED[@]} -gt 0 ]; then
  echo -e "${GREEN}✓ Installed:${NC}"
  for item in "${INSTALLED[@]}"; do
    echo -e "  ${GREEN}•${NC} $item"
  done
  echo ""
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo -e "${YELLOW}⚠ Already installed:${NC}"
  for item in "${SKIPPED[@]}"; do
    echo -e "  ${YELLOW}•${NC} $item"
  done
  echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo -e "${RED}✗ Failed to install:${NC}"
  for item in "${FAILED[@]}"; do
    echo -e "  ${RED}•${NC} $item"
  done
  echo ""
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Next steps
if [ ${#INSTALLED[@]} -gt 0 ] || [ ${#SKIPPED[@]} -gt 0 ]; then
  echo -e "${GREEN}Next steps:${NC}"
  echo "  Run formatters with: ${BLUE}npm run format${NC} or ${BLUE}./scripts/format-all.sh${NC}"
  echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo -e "${YELLOW}Note:${NC} Some formatters failed to install. You can still use the ones that succeeded."
  echo ""
fi
