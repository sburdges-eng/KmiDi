#!/bin/bash
# =============================================================================
# KmiDi SpectoCloud CUDA Training Script v2
# =============================================================================
# Deployment: Ubuntu 22.04 Deep Learning AMI (NVIDIA GPU Cloud)
# Target: 4x GPU distributed training with PyTorch DDP
# Budget: $150 (12-18 hours on RTX 4090/A100, up to 30 hours on spot)
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# -----------------------------------------------------------------------------
# Color Output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
AUDIO_ROOT=""
MIDI_ROOT=""
NUM_GPUS=4
CONFIG="training/cuda_session/spectocloud_training_config.yaml"
RESUME=""
WANDB_KEY=""
DRY_RUN=false

usage() {
    cat << EOF
Usage: $0 --audio-root PATH --midi-root PATH [OPTIONS]

Required:
  --audio-root PATH     Path to audio dataset root
  --midi-root PATH      Path to MIDI dataset root

Optional:
  --num-gpus N          Number of GPUs to use (default: 4)
  --config PATH         Training config file (default: $CONFIG)
  --resume PATH         Resume from checkpoint
  --wandb-key KEY       Weights & Biases API key for logging
  --dry-run             Check setup without training
  -h, --help            Show this help message

Example:
  $0 --audio-root /data/audio --midi-root /data/midi --num-gpus 4

EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio-root)
            AUDIO_ROOT="$2"
            shift 2
            ;;
        --midi-root)
            MIDI_ROOT="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --wandb-key)
            WANDB_KEY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$AUDIO_ROOT" ]] || [[ -z "$MIDI_ROOT" ]]; then
    log_error "Both --audio-root and --midi-root are required"
    usage
fi

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
log_info "KmiDi SpectoCloud Training - Environment Check"
echo "=================================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

log_success "CUDA detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1)
log_info "Python: $PYTHON_VERSION"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    log_error "PyTorch not installed"
    exit 1
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
CUDA_AVAILABLE=$(python3 -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')")
log_info "PyTorch: $TORCH_VERSION (CUDA: $CUDA_AVAILABLE)"

if [[ "$CUDA_AVAILABLE" != "Yes" ]]; then
    log_error "PyTorch cannot access CUDA. Check installation."
    exit 1
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
log_info "Available GPUs: $GPU_COUNT"

if [[ "$GPU_COUNT" -lt "$NUM_GPUS" ]]; then
    log_warn "Requested $NUM_GPUS GPUs but only $GPU_COUNT available. Adjusting..."
    NUM_GPUS=$GPU_COUNT
fi

# -----------------------------------------------------------------------------
# Validate Data Paths
# -----------------------------------------------------------------------------
log_info "Validating data paths..."

if [[ ! -d "$AUDIO_ROOT" ]]; then
    log_error "Audio root not found: $AUDIO_ROOT"
    exit 1
fi

if [[ ! -d "$MIDI_ROOT" ]]; then
    log_error "MIDI root not found: $MIDI_ROOT"
    exit 1
fi

AUDIO_COUNT=$(find "$AUDIO_ROOT" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l)
MIDI_COUNT=$(find "$MIDI_ROOT" -type f -name "*.mid*" | wc -l)

log_info "Audio files found: $AUDIO_COUNT"
log_info "MIDI files found: $MIDI_COUNT"

if [[ "$AUDIO_COUNT" -eq 0 ]]; then
    log_error "No audio files found in $AUDIO_ROOT"
    exit 1
fi

if [[ "$MIDI_COUNT" -eq 0 ]]; then
    log_warn "No MIDI files found in $MIDI_ROOT (OK for audio-only training)"
fi

# -----------------------------------------------------------------------------
# Validate Config File
# -----------------------------------------------------------------------------
log_info "Validating config file: $CONFIG"

if [[ ! -f "$CONFIG" ]]; then
    log_error "Config file not found: $CONFIG"
    exit 1
fi

log_success "Config file found"

# -----------------------------------------------------------------------------
# Install Dependencies
# -----------------------------------------------------------------------------
log_info "Checking Python dependencies..."

REQUIRED_PACKAGES=(
    "torch"
    "torchaudio"
    "transformers"
    "timm"
    "einops"
    "numpy"
    "scipy"
    "librosa"
    "pretty_midi"
    "tensorboard"
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    log_warn "Missing packages: ${MISSING_PACKAGES[*]}"
    log_info "Installing missing dependencies..."
    pip install -q "${MISSING_PACKAGES[@]}"
    log_success "Dependencies installed"
else
    log_success "All dependencies satisfied"
fi

# Optional: Weights & Biases
if [[ -n "$WANDB_KEY" ]]; then
    log_info "Setting up Weights & Biases logging..."
    pip install -q wandb
    export WANDB_API_KEY="$WANDB_KEY"
    log_success "W&B configured"
fi

# -----------------------------------------------------------------------------
# Setup Output Directory
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/spectocloud_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
log_info "Output directory: $OUTPUT_DIR"

# Save run configuration
cat > "$OUTPUT_DIR/run_config.txt" << EOF
KmiDi SpectoCloud Training Run
================================
Timestamp: $TIMESTAMP
Audio Root: $AUDIO_ROOT
MIDI Root: $MIDI_ROOT
Config: $CONFIG
GPUs: $NUM_GPUS
Resume: ${RESUME:-None}

System Information:
-------------------
$(nvidia-smi)

Python Environment:
-------------------
$(pip list | grep -E "(torch|transformers|timm)")
EOF

log_success "Run configuration saved to $OUTPUT_DIR/run_config.txt"

# -----------------------------------------------------------------------------
# Dry Run Check
# -----------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    log_success "Dry run complete. Setup validated successfully!"
    log_info "To start training, run without --dry-run flag"
    exit 0
fi

# -----------------------------------------------------------------------------
# Launch Training
# -----------------------------------------------------------------------------
log_info "Starting distributed training with $NUM_GPUS GPUs..."
echo "=================================================="
echo ""

# Build training command
TRAINING_SCRIPT="training/cuda_session/train_spectocloud.py"

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    log_error "Training script '$TRAINING_SCRIPT' not found. Expected at: $(pwd)/$TRAINING_SCRIPT"
    log_error "Please ensure the training script exists or update TRAINING_SCRIPT in scripts/train_cuda_v2.sh."
    exit 1
fi

TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS $TRAINING_SCRIPT"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
TRAIN_CMD="$TRAIN_CMD --audio-root $AUDIO_ROOT"
TRAIN_CMD="$TRAIN_CMD --midi-root $MIDI_ROOT"
TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"

if [[ -n "$RESUME" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

# Log command
log_info "Command: $TRAIN_CMD"
echo "$TRAIN_CMD" > "$OUTPUT_DIR/train_command.sh"
chmod +x "$OUTPUT_DIR/train_command.sh"

# Setup logging
LOG_FILE="$OUTPUT_DIR/training.log"
log_info "Training log: $LOG_FILE"
echo ""

# Execute training with tee for both stdout and file logging
log_success "🚀 Training started at $(date)"
echo "=================================================="
echo ""

set +e  # Don't exit on training errors (we want to capture them)
$TRAIN_CMD 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo ""
echo "=================================================="
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    log_success "✅ Training completed successfully at $(date)"
    log_info "Results saved to: $OUTPUT_DIR"
    log_info "Check tensorboard logs: tensorboard --logdir $OUTPUT_DIR"
else
    log_error "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    log_error "Check logs: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

# -----------------------------------------------------------------------------
# Post-Training Summary
# -----------------------------------------------------------------------------
log_info "Training Summary:"
echo "  - Output: $OUTPUT_DIR"
echo "  - Config: $CONFIG"
echo "  - GPUs: $NUM_GPUS"
echo "  - Log: $LOG_FILE"

if [[ -f "$OUTPUT_DIR/final_metrics.json" ]]; then
    log_info "Final metrics:"
    cat "$OUTPUT_DIR/final_metrics.json"
fi

log_success "Done! 🎵"
