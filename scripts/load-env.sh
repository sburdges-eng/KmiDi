#!/bin/bash
# =============================================================================
# Environment Variable Loader
# =============================================================================
# Sources all relevant .env files in priority order:
# 1. Base .env (if exists)
# 2. Feature-specific files (if exists and feature enabled)
# 3. .env.local (highest priority, git-ignored)
#
# Usage: source scripts/load-env.sh [feature1] [feature2] ...
# Features: tauri, ml, training, mcp
# =============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Features to load (passed as arguments or default to all)
FEATURES="${@:-tauri ml training mcp}"

# Function to load .env file if it exists
load_env_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Loading: $file" >&2
        # Export variables, handling comments and empty lines
        # Use a safer method that works in all shells
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            
            # Export the variable
            export "$line" 2>/dev/null || true
        done < "$file"
    fi
}

# Load base .env first (lowest priority)
if [ -f "$PROJECT_ROOT/.env" ]; then
    load_env_file "$PROJECT_ROOT/.env"
fi

# Determine environment (development, production, or default)
ENV_MODE="${NODE_ENV:-${KMIDI_ENV:-development}}"

# Load environment-specific file (.env.development or .env.production)
if [ "$ENV_MODE" = "production" ] && [ -f "$PROJECT_ROOT/.env.production" ]; then
    load_env_file "$PROJECT_ROOT/.env.production"
elif [ "$ENV_MODE" = "development" ] && [ -f "$PROJECT_ROOT/.env.development" ]; then
    load_env_file "$PROJECT_ROOT/.env.development"
fi

# Load feature-specific files
for feature in $FEATURES; do
    case "$feature" in
        tauri)
            load_env_file "$PROJECT_ROOT/env/.env.tauri"
            ;;
        ml)
            load_env_file "$PROJECT_ROOT/env/.env.ml"
            ;;
        training)
            load_env_file "$PROJECT_ROOT/env/.env.training"
            ;;
        mcp)
            load_env_file "$PROJECT_ROOT/env/.env.mcp"
            ;;
    esac
done

# Load .env.local last (highest priority, user overrides)
load_env_file "$PROJECT_ROOT/.env.local"

echo "Environment loaded successfully" >&2
