#!/usr/bin/env bash
set -euo pipefail

# Compute SHA256 checksums for DEAM and EMO-Music directories.
# Outputs:
#   datasets/DEAM_SHA256SUMS.txt
#   datasets/EMO_Music_SHA256SUMS.txt
#
# Usage:
#   bash scripts/hash_emotion_datasets.sh
# (run after datasets/DEAM and datasets/EMO_Music are populated)

ROOT="datasets"
DEAM_DIR="${ROOT}/DEAM"
EMO_DIR="${ROOT}/EMO_Music"

[[ -d "${DEAM_DIR}" ]] || { echo "Missing ${DEAM_DIR}"; exit 1; }
[[ -d "${EMO_DIR}" ]] || { echo "Missing ${EMO_DIR}"; exit 1; }

echo "Computing SHA256 for DEAM..."
find "${DEAM_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/DEAM_SHA256SUMS.txt"

echo "Computing SHA256 for EMO-Music..."
find "${EMO_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/EMO_Music_SHA256SUMS.txt"

echo "Done. Checksums written to:"
echo "  ${ROOT}/DEAM_SHA256SUMS.txt"
echo "  ${ROOT}/EMO_Music_SHA256SUMS.txt"
