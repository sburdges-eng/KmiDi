#!/bin/bash
# =============================================================================
# LOCAL TRAINING SCRIPT - Updated: Files moved from external SSD (2025-01-09)
# =============================================================================
# Usage:
#   ./scripts/local_train.sh                    # Train all models
#   ./scripts/local_train.sh emotion_recognizer # Train specific model
#   ./scripts/local_train.sh --list             # List available models
# =============================================================================

set -e

# Data root (configurable)
if [ -n "$KMI_DI_AUDIO_DATA_ROOT" ]; then
    DATA="$KMI_DI_AUDIO_DATA_ROOT"
elif [ -n "$KELLY_AUDIO_DATA_ROOT" ]; then
    DATA="$KELLY_AUDIO_DATA_ROOT"
else
    DATA="$(cd "$(dirname "$0")/.." && pwd)/data/audio"
fi
VENV="${VIRTUAL_ENV:-$(dirname "$(which python3)")/../venv}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_prereqs() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    # Updated: Files moved from external SSD to local storage (2025-01-09)
    if [ ! -d "$DATA" ]; then
        echo -e "${YELLOW}WARNING: Data directory not found at $DATA${NC}"
        echo -e "${YELLOW}Creating data directory...${NC}"
        mkdir -p "$DATA/raw" "$DATA/processed" "$DATA/downloads" "$DATA/cache"
    fi
    if [ ! -d "$VENV" ] && [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${YELLOW}WARNING: Virtual environment not found. Using system Python.${NC}"
        VENV=""
    fi
    echo -e "${GREEN}Prerequisites checked${NC}"
}

show_data_status() {
    echo ""
    echo "=== Dataset Status ==="
    du -sh "$DATA"/*/ 2>/dev/null || echo "No datasets found"
    echo ""
}

activate_env() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    if [ -n "$VENV" ] && [ -d "$VENV" ]; then
        source "$VENV/bin/activate"
        echo -e "${GREEN}Virtual environment activated${NC}"
    elif [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${GREEN}Using existing virtual environment: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}No virtual environment - using system Python${NC}"
    fi
    export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
    export HF_HOME="$DATA/hf_cache"
    export HF_DATASETS_CACHE="$DATA/hf_cache"
    export KELLY_AUDIO_DATA_ROOT="$DATA"
    echo -e "${GREEN}Environment ready${NC}"
}

train_model() {
    local model=$1
    echo -e "${YELLOW}=== Training: $model ===${NC}"
    cd "$REPO_DIR"
    python scripts/train.py --model "$model" --data "$DATA" --epochs 50 --export-onnx --export-coreml
    echo -e "${GREEN}=== $model training complete ===${NC}"
}

list_models() {
    echo "Available models:"
    echo "  emotion_recognizer  - Audio emotion classification"
    echo "  melody_transformer  - Melodic sequence generation"
    echo "  harmony_predictor   - Chord progression prediction"
    echo "  dynamics_engine     - Expression parameter mapping"
    echo "  groove_predictor    - Groove/timing patterns"
}

main() {
    echo "=============================================="
    echo "  KmiDi Local Training Script"
    echo "  Data: $DATA"
    echo "=============================================="

    [ "$1" == "--list" ] || [ "$1" == "-l" ] && list_models && exit 0

    check_prereqs
    show_data_status
    activate_env

    if [ -n "$1" ]; then
        train_model "$1"
    else
        for m in emotion_recognizer harmony_predictor melody_transformer dynamics_engine groove_predictor; do
            train_model "$m"
        done
    fi

    echo -e "${GREEN}=== Training complete ===${NC}"
    echo "Push models: git add models/ && git commit -m 'Trained models' && git push"
}

main "$@"
