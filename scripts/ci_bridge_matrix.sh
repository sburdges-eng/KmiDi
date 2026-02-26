#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JUCE_ROOT="${JUCE_EXTERNAL_ROOT:-}"
CORE_DIR="${KELLY_CORE_DIR:-${ROOT}/../minimal_listening_core/c}"

echo "[lane] core_bridge=OFF"
if [[ -n "${JUCE_ROOT}" ]]; then
  JUCE_EXTERNAL_ROOT="${JUCE_ROOT}" \
  KELLY_USE_CORE_BRIDGE=OFF \
  BUILD_DIR="${ROOT}/build-ci-off" \
  "${ROOT}/scripts/run_contract_checks.sh"
else
  KELLY_USE_CORE_BRIDGE=OFF \
  BUILD_DIR="${ROOT}/build-ci-off" \
  "${ROOT}/scripts/run_contract_checks.sh"
fi

echo ""
echo "[lane] core_bridge=ON"
if [[ -n "${JUCE_ROOT}" ]]; then
  JUCE_EXTERNAL_ROOT="${JUCE_ROOT}" \
  KELLY_USE_CORE_BRIDGE=ON \
  KELLY_CORE_DIR="${CORE_DIR}" \
  BUILD_DIR="${ROOT}/build-ci-on" \
  "${ROOT}/scripts/run_contract_checks.sh"
else
  KELLY_USE_CORE_BRIDGE=ON \
  KELLY_CORE_DIR="${CORE_DIR}" \
  BUILD_DIR="${ROOT}/build-ci-on" \
  "${ROOT}/scripts/run_contract_checks.sh"
fi

echo ""
echo "CI bridge matrix complete (OFF + ON lanes)."
