#!/usr/bin/env bash
set -euo pipefail

# Download DEAM + EMO-Music into the required layout:
# datasets/
# ├── DEAM/       {audio/, annotations/, metadata.csv}
# └── EMO_Music/  {audio/, annotations/, metadata.csv}
#
# Configure URLs via env vars to avoid hard-coding flaky mirrors:
#   DEAM_AUDIO_URL          (e.g., Zenodo DEAM_audio.zip)
#   DEAM_ANNOTATIONS_URL    (e.g., Zenodo annotations.zip)
#   EMO_MUSIC_ZIP_URL       (e.g., GitHub mirror or official emoMusic.zip)
#   DEAM_AUDIO_SHA256       (optional, for verification)
#   DEAM_ANNOTATIONS_SHA256 (optional)
#   EMO_MUSIC_SHA256        (optional)
#
# Usage:
#   DEAM_AUDIO_URL=... DEAM_ANNOTATIONS_URL=... EMO_MUSIC_ZIP_URL=... bash scripts/download_emotion_datasets.sh

ROOT="datasets"
DEAM_DIR="${ROOT}/DEAM"
EMO_DIR="${ROOT}/EMO_Music"

mkdir -p "${DEAM_DIR}" "${EMO_DIR}"

download() {
  local url="$1"
  local dest="$2"

  if [[ -f "${dest}" ]]; then
    echo "Skipping download (exists): ${dest}"
    return
  fi

  echo "Downloading ${url} -> ${dest}"
  curl -L --fail --retry 3 --retry-delay 2 -o "${dest}" "${url}" || { echo "Download failed: ${url}"; exit 1; }
}

verify_sha() {
  local file="$1"
  local expected="$2"
  if [[ -z "${expected}" ]]; then
    return
  fi
  echo "${expected}  ${file}" | sha256sum --check --status || {
    echo "Checksum mismatch for ${file}"
    exit 1
  }
}

extract_zip() {
  local zip="$1"
  local dest="$2"
  echo "Extracting ${zip} -> ${dest}"
  unzip -o "${zip}" -d "${dest}"
}

# DEAM
if [[ -n "${DEAM_AUDIO_URL:-}" ]]; then
  download "${DEAM_AUDIO_URL}" "${DEAM_DIR}/DEAM_audio.zip"
  verify_sha "${DEAM_DIR}/DEAM_audio.zip" "${DEAM_AUDIO_SHA256:-}"
  extract_zip "${DEAM_DIR}/DEAM_audio.zip" "${DEAM_DIR}"
else
  echo "DEAM_AUDIO_URL not set; skipping DEAM audio download"
fi

if [[ -n "${DEAM_ANNOTATIONS_URL:-}" ]]; then
  download "${DEAM_ANNOTATIONS_URL}" "${DEAM_DIR}/annotations.zip"
  verify_sha "${DEAM_DIR}/annotations.zip" "${DEAM_ANNOTATIONS_SHA256:-}"
  extract_zip "${DEAM_DIR}/annotations.zip" "${DEAM_DIR}"
else
  echo "DEAM_ANNOTATIONS_URL not set; skipping DEAM annotations download"
fi

# EMO-Music
if [[ -n "${EMO_MUSIC_ZIP_URL:-}" ]]; then
  download "${EMO_MUSIC_ZIP_URL}" "${EMO_DIR}/EMO_Music.zip"
  verify_sha "${EMO_DIR}/EMO_Music.zip" "${EMO_MUSIC_SHA256:-}"
  extract_zip "${EMO_DIR}/EMO_Music.zip" "${EMO_DIR}"
else
  echo "EMO_MUSIC_ZIP_URL not set; skipping EMO-Music download"
fi

echo "Done. Ensure you have: ${DEAM_DIR}/audio, ${DEAM_DIR}/annotations, ${DEAM_DIR}/metadata.csv and same for EMO_Music."
