#!/usr/bin/env bash
# ingest_datasets.sh - Automates download, preprocess, and packing of mtg_jamendo and fma_full
# Designed for high-speed JEPA training preparation on external SSD.

set -e

# Target root on External SSD
export KMIDI_DATA_ROOT="/Volumes/Sean's SSD"
export AUDIO_DATA_ROOT="/Volumes/Sean's SSD/Datasets"

echo "===================================================="
echo " KmiDi Dataset Ingestion: mtg_jamendo & fma_full"
echo " Target: $KMIDI_DATA_ROOT"
echo "===================================================="

cd "$(dirname "$0")/.."
ROOT=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${ROOT}"

# Use virtual environment if it exists
if [ -d "venv" ]; then
    echo "Using virtual environment: venv"
    source venv/bin/activate
fi

# Function to run ingestion for a specific dataset
ingest_dataset() {
    local ds=$1
    echo ""
    echo ">>> Starting ingestion for: $ds"
    echo ">>> [1/2] Downloading..."
    python3 scripts/utilities/prepare_datasets.py --dataset "$ds" --download
    
    echo ">>> [2/2] Preprocessing..."
    python3 scripts/utilities/prepare_datasets.py --dataset "$ds" --preprocess
    
    echo ">>> Finished: $ds"
}

# Run for mtg_jamendo
ingest_dataset "mtg_jamendo"

# Run for fma_full using aria2 for robust resume
echo ""
echo ">>> Starting ingestion for: fma_full"
echo ">>> [1/2] Downloading (using aria2)..."
chmod +x scripts/download_fma_full_aria2.sh
./scripts/download_fma_full_aria2.sh

echo ">>> [2/2] Preprocessing..."
python3 scripts/utilities/prepare_datasets.py --dataset "fma_full" --preprocess

echo ">>> Finished: fma_full"

echo ""
echo "===================================================="
echo " All datasets ingested, preprocessed, and packed."
echo " Ready for high-speed JEPA training."
echo "===================================================="
