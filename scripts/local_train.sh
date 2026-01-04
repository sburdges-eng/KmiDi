#!/bin/bash
# =============================================================================
# LOCAL TRAINING SCRIPT - Run on Mac with /Volumes/sbdrive
# =============================================================================
# Usage:
#   ./scripts/local_train.sh                    # Train all models
#   ./scripts/local_train.sh emotion_recognizer # Train specific model
#   ./scripts/local_train.sh --list             # List available models
# =============================================================================

set -e

VOL="/Volumes/sbdrive"
VENV="$VOL/venv"
DATA="$VOL/audio/datasets"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_prereqs() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    [ ! -d "$VOL" ] && echo -e "${RED}ERROR: $VOL not mounted${NC}" && exit 1
    [ ! -d "$VENV" ] && echo -e "${RED}ERROR: venv not found at $VENV${NC}" && exit 1
    [ ! -d "$DATA" ] && echo -e "${RED}ERROR: datasets not found at $DATA${NC}" && exit 1
    echo -e "${GREEN}All prerequisites met${NC}"
}

show_data_status() {
    echo ""
    echo "=== Dataset Status ==="
    du -sh "$DATA"/*/ 2>/dev/null || echo "No datasets found"
    echo ""
}

activate_env() {
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source "$VENV/bin/activate"
    export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
    export HF_HOME="$DATA/hf_cache"
    export HF_DATASETS_CACHE="$DATA/hf_cache"
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
