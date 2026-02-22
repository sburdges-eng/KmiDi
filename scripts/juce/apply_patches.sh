#!/usr/bin/env bash
# Apply local JUCE compatibility patches in a deterministic and idempotent way.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
JUCE_DIR="${PROJECT_ROOT}/external/JUCE"
PATCH_DIR="${PROJECT_ROOT}/third_party/patches/juce"

if [ ! -d "${JUCE_DIR}" ]; then
    echo "JUCE directory not found: ${JUCE_DIR}" >&2
    exit 1
fi

if ! git -C "${JUCE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "JUCE directory is not a git checkout: ${JUCE_DIR}" >&2
    exit 1
fi

if [ ! -d "${PATCH_DIR}" ]; then
    echo "Patch directory missing: ${PATCH_DIR}" >&2
    exit 1
fi

shopt -s nullglob
PATCHES=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob

if [ ${#PATCHES[@]} -eq 0 ]; then
    echo "No JUCE patch files found in ${PATCH_DIR}" >&2
    exit 1
fi

for patch in "${PATCHES[@]}"; do
    patch_name="$(basename "${patch}")"

    if git -C "${JUCE_DIR}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "JUCE patch already applied: ${patch_name}"
        continue
    fi

    if ! git -C "${JUCE_DIR}" apply --check "${patch}" >/dev/null 2>&1; then
        juce_head="$(git -C "${JUCE_DIR}" rev-parse --short HEAD)"
        echo "Failed to apply JUCE patch ${patch_name} on external/JUCE@${juce_head}" >&2
        echo "Patch drift detected. Update ${patch_name} for the current JUCE revision." >&2
        exit 1
    fi

    git -C "${JUCE_DIR}" apply "${patch}"
    echo "Applied JUCE patch: ${patch_name}"
done
