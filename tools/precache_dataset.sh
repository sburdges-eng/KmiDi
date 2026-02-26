#!/usr/bin/env bash
set -euo pipefail

PACKAGE_S3_URI=""
WORKDIR="/opt/listening-train/work"
RESULT_JSON=""
CACHE_ROOT=""
MODE="auto"
ALLOW_STREAMING_FALLBACK=1

usage() {
  cat <<'USAGE'
Usage: precache_dataset.sh --package-s3-uri <s3://bucket/prefix> [options]

Prefers NVMe cache when available, falls back to EBS cache, then optional streaming mode.

Options:
  --package-s3-uri <uri>      Packaged dataset root URI (required)
  --workdir <path>            Working directory (default: /opt/listening-train/work)
  --result-json <path>        Output JSON path (default: <workdir>/cache/precache_result.json)
  --cache-root <path>         Explicit cache root override
  --mode <auto|nvme|ebs|stream>
  --allow-streaming-fallback <0|1>  Allow streaming fallback when cache copy fails (default: 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package-s3-uri)
      PACKAGE_S3_URI="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --result-json)
      RESULT_JSON="$2"
      shift 2
      ;;
    --cache-root)
      CACHE_ROOT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --allow-streaming-fallback)
      ALLOW_STREAMING_FALLBACK="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PACKAGE_S3_URI}" ]]; then
  echo "--package-s3-uri is required" >&2
  usage >&2
  exit 2
fi

case "${MODE}" in
  auto|nvme|ebs|stream) ;;
  *)
    echo "Invalid --mode: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${RESULT_JSON}" ]]; then
  RESULT_JSON="${WORKDIR}/cache/precache_result.json"
fi
mkdir -p "$(dirname "${RESULT_JSON}")"

write_result() {
  local cache_mode="$1"
  local cache_root="$2"
  local package_local_dir="$3"
  local streaming="$4"

  CACHE_MODE_ENV="${cache_mode}" \
  CACHE_ROOT_ENV="${cache_root}" \
  PACKAGE_LOCAL_DIR_ENV="${package_local_dir}" \
  STREAMING_ENV="${streaming}" \
  PACKAGE_S3_URI_ENV="${PACKAGE_S3_URI}" \
  RESULT_JSON_ENV="${RESULT_JSON}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path
payload = {
  "cacheMode": os.environ.get("CACHE_MODE_ENV", ""),
  "cacheRoot": os.environ.get("CACHE_ROOT_ENV", ""),
  "packageLocalDir": os.environ.get("PACKAGE_LOCAL_DIR_ENV", ""),
  "streaming": bool(int(os.environ.get("STREAMING_ENV", "0"))),
  "sourceS3Uri": os.environ.get("PACKAGE_S3_URI_ENV", ""),
}
out = Path(os.environ.get("RESULT_JSON_ENV", ""))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

detect_nvme_root() {
  if command -v lsblk >/dev/null 2>&1; then
    while IFS= read -r row; do
      local name mountpoint
      name="$(echo "${row}" | awk '{print $1}')"
      mountpoint="$(echo "${row}" | awk '{print $2}')"
      if [[ "${name}" == nvme* && -n "${mountpoint}" && "${mountpoint}" != "/" ]]; then
        if [[ -d "${mountpoint}" && -w "${mountpoint}" ]]; then
          echo "${mountpoint}"
          return 0
        fi
      fi
    done < <(lsblk -nrpo NAME,MOUNTPOINT)
  fi

  local candidates=(/mnt /local /scratch /mnt/nvme /nvme)
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" && -w "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

choose_cache_root() {
  local mode="$1"
  local chosen=""

  if [[ -n "${CACHE_ROOT}" ]]; then
    echo "${CACHE_ROOT}"
    return 0
  fi

  if [[ "${mode}" == "nvme" || "${mode}" == "auto" ]]; then
    chosen="$(detect_nvme_root || true)"
    if [[ -n "${chosen}" ]]; then
      echo "${chosen}/listening-cache"
      return 0
    fi
    if [[ "${mode}" == "nvme" ]]; then
      return 1
    fi
  fi

  if [[ "${mode}" == "stream" ]]; then
    return 1
  fi

  echo "/var/tmp/listening-cache"
  return 0
}

if [[ "${MODE}" == "stream" ]]; then
  write_result "stream" "" "" "1"
  exit 0
fi

CACHE_ROOT_RESOLVED="$(choose_cache_root "${MODE}" || true)"
if [[ -z "${CACHE_ROOT_RESOLVED}" ]]; then
  if [[ "${ALLOW_STREAMING_FALLBACK}" == "1" ]]; then
    echo "[precache_dataset] no cache volume available; using streaming mode"
    write_result "stream" "" "" "1"
    exit 0
  fi
  echo "[precache_dataset] no cache volume available and streaming fallback disabled" >&2
  exit 1
fi

if [[ "${CACHE_ROOT_RESOLVED}" == *"/listening-cache"* ]]; then
  CACHE_MODE="nvme"
else
  CACHE_MODE="ebs"
fi

PACKAGE_LOCAL_DIR="${CACHE_ROOT_RESOLVED}/package"
mkdir -p "${PACKAGE_LOCAL_DIR}"

if ! command -v aws >/dev/null 2>&1; then
  echo "[precache_dataset] aws CLI not found" >&2
  if [[ "${ALLOW_STREAMING_FALLBACK}" == "1" ]]; then
    write_result "stream" "" "" "1"
    exit 0
  fi
  exit 1
fi

echo "[precache_dataset] downloading shards to ${PACKAGE_LOCAL_DIR}"
if aws s3 cp "${PACKAGE_S3_URI}" "${PACKAGE_LOCAL_DIR}" \
  --recursive \
  --exclude "*" \
  --include "*.json" \
  --include "*.txt" \
  --include "*.jsonl.zst"; then
  write_result "${CACHE_MODE}" "${CACHE_ROOT_RESOLVED}" "${PACKAGE_LOCAL_DIR}" "0"
  exit 0
fi

echo "[precache_dataset] cache copy failed" >&2
if [[ "${ALLOW_STREAMING_FALLBACK}" == "1" ]]; then
  write_result "stream" "" "" "1"
  exit 0
fi

exit 1
