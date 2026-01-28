#!/bin/bash
# =============================================================================
# Environment Variable Validator
# =============================================================================
# Validates that required environment variables are set
# Checks file paths exist, validates API key formats, etc.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment first
if [ -f "$SCRIPT_DIR/load-env.sh" ]; then
    source "$SCRIPT_DIR/load-env.sh"
fi

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Environment Validation"
echo "=========================================="
echo ""

# Function to check if variable is set
check_var() {
    local var_name="$1"
    local required="${2:-false}"
    local value="${!var_name}"
    
    if [ -z "$value" ]; then
        if [ "$required" = "true" ]; then
            echo -e "${RED}✗${NC} $var_name: NOT SET (REQUIRED)"
            ((ERRORS++))
        else
            echo -e "${YELLOW}⚠${NC} $var_name: NOT SET (optional)"
            ((WARNINGS++))
        fi
    else
        echo -e "${GREEN}✓${NC} $var_name: $value"
    fi
}

# Function to check if path exists
check_path() {
    local var_name="$1"
    local path="${!var_name}"
    local required="${2:-false}"
    
    if [ -z "$path" ]; then
        if [ "$required" = "true" ]; then
            echo -e "${RED}✗${NC} $var_name: NOT SET (REQUIRED)"
            ((ERRORS++))
        fi
        return
    fi
    
    # Expand relative paths
    if [[ "$path" != /* ]]; then
        path="$PROJECT_ROOT/$path"
    fi
    
    if [ -d "$path" ] || [ -f "$path" ]; then
        echo -e "${GREEN}✓${NC} $var_name: $path (exists)"
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}✗${NC} $var_name: $path (NOT FOUND - REQUIRED)"
            ((ERRORS++))
        else
            echo -e "${YELLOW}⚠${NC} $var_name: $path (not found)"
            ((WARNINGS++))
        fi
    fi
}

# Function to validate API key format (basic check)
validate_api_key() {
    local var_name="$1"
    local value="${!var_name}"
    
    if [ -z "$value" ]; then
        return
    fi
    
    # Basic validation - not empty and not placeholder
    if [[ "$value" == *"your-"* ]] || [[ "$value" == *"here"* ]]; then
        echo -e "${YELLOW}⚠${NC} $var_name: Appears to be placeholder value"
        ((WARNINGS++))
    elif [ ${#value} -lt 10 ]; then
        echo -e "${YELLOW}⚠${NC} $var_name: Suspiciously short (may be invalid)"
        ((WARNINGS++))
    fi
}

echo "=== Required Variables ==="
check_var "KELLY_MODELS_PATH" true
check_path "KELLY_MODELS_PATH" false

echo ""
echo "=== API Keys ==="
check_var "OPENAI_API_KEY" false
validate_api_key "OPENAI_API_KEY"
check_var "ANTHROPIC_API_KEY" false
validate_api_key "ANTHROPIC_API_KEY"
check_var "GOOGLE_API_KEY" false
validate_api_key "GOOGLE_API_KEY"
check_var "XAI_API_KEY" false
validate_api_key "XAI_API_KEY"
check_var "GITHUB_TOKEN" false
validate_api_key "GITHUB_TOKEN"

echo ""
echo "=== Service Configuration ==="
check_var "TAURI_DEV_HOST" false
check_var "KMIDI_API_URL" false
check_var "ML_INFERENCE_URL" false

echo ""
echo "=== Logging Configuration ==="
check_var "RUST_LOG" false
check_var "RUST_BACKTRACE" false
check_var "CXX_LOG_LEVEL" false
check_var "PYTHON_LOG_LEVEL" false

echo ""
echo "=== Training Configuration ==="
check_var "LOCAL_RANK" false
check_var "WORLD_SIZE" false
check_var "CUDA_VISIBLE_DEVICES" false

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Validation complete with $WARNINGS warning(s)${NC}"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    exit 1
fi
