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

compute_checksums() {
  local dir="$1"
  local outfile="$2"
  local label="$3"

  # Check if the directory contains at least one regular file.
  if ! find "${dir}" -type f -print -quit | grep -q .; then
    echo "No files found in ${label}; skipping checksum generation."
    return 0
  fi

  echo "Computing SHA256 for ${label}..."
  find "${dir}" -type f -print0 | sort -z | xargs -0 sha256sum > "${outfile}"
}

compute_checksums "${DEAM_DIR}" "${ROOT}/DEAM_SHA256SUMS.txt" "DEAM"
compute_checksums "${EMO_DIR}" "${ROOT}/EMO_Music_SHA256SUMS.txt" "EMO-Music"
echo "Done. Checksums written to:"
echo "  ${ROOT}/DEAM_SHA256SUMS.txt"
echo "  ${ROOT}/EMO_Music_SHA256SUMS.txt"
