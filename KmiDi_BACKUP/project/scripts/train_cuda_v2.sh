#!/bin/bash
# =============================================================================
# KmiDi v2 Training Pipeline - CUDA GPU Runner
# =============================================================================
# One-command script to build, run, and execute v2 training pipeline on GPU.
#
# Usage:
#   ./scripts/train_cuda_v2.sh [options]
#
# Options:
#   --audio-root PATH      Path to audio files (optional)
#   --midi-root PATH       Path to MIDI files (optional)
#   --data-mount PATH      Path to mount as /data in container (optional)
#   --epochs NUM           Number of epochs (default: use config)
#   --skip-build           Skip Docker image build
#   --skip-manifests       Skip manifest generation
#   --skip-spectocloud     Skip Spectocloud training
#   --skip-midi            Skip MIDI generator training
#   --skip-export          Skip ONNX export
# =============================================================================

set -e

# Default values
AUDIO_ROOT=""
MIDI_ROOT=""
DATA_MOUNT=""
SKIP_BUILD=0
SKIP_MANIFESTS=0
SKIP_SPECTOCLOUD=0
SKIP_MIDI=0
SKIP_EXPORT=0
EPOCHS=""

# Parse arguments
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
        --data-mount)
            DATA_MOUNT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-manifests)
            SKIP_MANIFESTS=1
            shift
            ;;
        --skip-spectocloud)
            SKIP_SPECTOCLOUD=1
            shift
            ;;
        --skip-midi)
            SKIP_MIDI=1
            shift
            ;;
        --skip-export)
            SKIP_EXPORT=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "KmiDi v2 Training Pipeline - CUDA GPU"
echo "=========================================="
echo ""

# Build Docker image
if [ $SKIP_BUILD -eq 0 ]; then
    echo "Building Docker image..."
    cd "$REPO_ROOT"
    docker build \
        -f deployment/docker/Dockerfile.cuda \
        -t kmidi-cuda-train:v2 \
        .
    echo "✓ Docker image built"
    echo ""
else
    echo "Skipping Docker build (--skip-build)"
    echo ""
fi

# Prepare Docker run command
DOCKER_RUN_CMD="docker run --rm -it \
    --gpus all \
    --ipc=host \
    --shm-size=8g \
    -v \"$REPO_ROOT:/workspace\" \
    -w /workspace"

# Add data mount if specified
if [ -n "$DATA_MOUNT" ]; then
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -v \"$DATA_MOUNT:/data\""
fi

DOCKER_RUN_CMD="$DOCKER_RUN_CMD kmidi-cuda-train:v2"

# Step 1: Build manifests
if [ $SKIP_MANIFESTS -eq 0 ]; then
    echo "Step 1: Building dataset manifests..."
    
    MANIFEST_CMD="python scripts/build_manifests.py"
    
    if [ -n "$AUDIO_ROOT" ]; then
        MANIFEST_CMD="$MANIFEST_CMD --audio-root \"$AUDIO_ROOT\""
    fi
    
    if [ -n "$MIDI_ROOT" ]; then
        MANIFEST_CMD="$MANIFEST_CMD --midi-root \"$MIDI_ROOT\""
    fi
    
    eval "$DOCKER_RUN_CMD bash -c '$MANIFEST_CMD'"
    echo "✓ Manifests built"
    echo ""
else
    echo "Skipping manifest generation (--skip-manifests)"
    echo ""
fi

# Step 2: Train Spectocloud
if [ $SKIP_SPECTOCLOUD -eq 0 ]; then
    echo "Step 2: Training Spectocloud model..."
    
    TRAIN_CMD="cd training/cuda_session && python train_spectocloud.py --config spectocloud_training_config.yaml"
    
    eval "$DOCKER_RUN_CMD bash -c '$TRAIN_CMD'"
    echo "✓ Spectocloud training complete"
    echo ""
else
    echo "Skipping Spectocloud training (--skip-spectocloud)"
    echo ""
fi

# Step 3: Train MIDI Generator
if [ $SKIP_MIDI -eq 0 ]; then
    echo "Step 3: Training MIDI Generator model..."
    
    TRAIN_CMD="cd training/cuda_session && python train_midi_generator.py --config midi_generator_training_config.yaml"
    
    eval "$DOCKER_RUN_CMD bash -c '$TRAIN_CMD'"
    echo "✓ MIDI Generator training complete"
    echo ""
else
    echo "Skipping MIDI Generator training (--skip-midi)"
    echo ""
fi

# Step 4: Export models
if [ $SKIP_EXPORT -eq 0 ]; then
    echo "Step 4: Exporting models to ONNX..."
    
    EXPORT_CMD="cd training/cuda_session && python export_models.py --all"
    
    eval "$DOCKER_RUN_CMD bash -c '$EXPORT_CMD'"
    echo "✓ Model export complete"
    echo ""
else
    echo "Skipping model export (--skip-export)"
    echo ""
fi

echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - Checkpoints: checkpoints/"
echo "  - Exports: exports/"
echo "  - Manifests: data/manifests/"
echo ""
echo "Next steps:"
echo "  1. Review training logs and metrics"
echo "  2. Test ONNX exports with C++ bridge"
echo "  3. (Optional) Convert to CoreML on Mac"
echo ""
