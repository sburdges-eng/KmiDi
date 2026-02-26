#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIST_FILE="${SCRIPT_DIR}/training_ops_files.txt"

usage() {
  cat <<'USAGE'
Usage: tag_training_ops_yellow.sh

Applies Finder tag "Yellow" to files listed in tools/training_ops_files.txt.
If Finder automation is blocked, prints remediation and prepends a banner.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "Missing file list: ${LIST_FILE}" >&2
  exit 2
fi

BANNER_TOP="# ========================="
BANNER_MID="# 🟡 TRAINING OPS FILE 🟡"
BANNER_BOT="# ========================="

apply_fallback_banner() {
  local file_path="$1"
  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi

  if rg -q "🟡 TRAINING OPS FILE 🟡" "${file_path}"; then
    return 0
  fi

  local tmp_path
  tmp_path="$(mktemp)"
  {
    echo "${BANNER_TOP}"
    echo "${BANNER_MID}"
    echo "${BANNER_BOT}"
    echo
    cat "${file_path}"
  } > "${tmp_path}"
  mv "${tmp_path}" "${file_path}"
}

TAG_FAILURE=0
while IFS= read -r rel_path || [[ -n "${rel_path}" ]]; do
  [[ -z "${rel_path}" ]] && continue
  [[ "${rel_path}" == \#* ]] && continue

  abs_path="${ROOT}/${rel_path}"
  if [[ ! -e "${abs_path}" ]]; then
    echo "[tag_training_ops_yellow] missing path: ${rel_path}" >&2
    continue
  fi

  if ! osascript <<OSA >/dev/null 2>&1
tell application "Finder"
  set theFile to POSIX file "${abs_path}" as alias
  set tag names of theFile to {"Yellow"}
end tell
OSA
  then
    TAG_FAILURE=1
  fi
done < "${LIST_FILE}"

if [[ ${TAG_FAILURE} -eq 0 ]]; then
  echo "Applied Finder tag Yellow to files listed in ${LIST_FILE}."
  exit 0
fi

echo "Finder automation is blocked."
echo "Remediation: enable Terminal/WezTerm in Privacy & Security -> Automation, then re-run this script."
echo "Applying fallback banner to listed files."

while IFS= read -r rel_path || [[ -n "${rel_path}" ]]; do
  [[ -z "${rel_path}" ]] && continue
  [[ "${rel_path}" == \#* ]] && continue
  abs_path="${ROOT}/${rel_path}"
  apply_fallback_banner "${abs_path}"
done < "${LIST_FILE}"

echo "Fallback banner applied."
