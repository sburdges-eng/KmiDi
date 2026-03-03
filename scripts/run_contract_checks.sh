#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CORE_BRIDGE_MODE="${KELLY_USE_CORE_BRIDGE:-OFF}"

validate_juce_root() {
  local juce_root="$1"
  [[ -n "${juce_root}" ]] || return 1
  for required in "CMakeLists.txt" "extras/Build/CMake/JUCEModuleSupport.cmake" "modules/juce_core/juce_core.cpp"; do
    [[ -f "${juce_root}/${required}" ]] || return 1
  done
  return 0
}

detect_local_juce_root() {
  local candidates=(
    "${ROOT}/external/JUCE"
    "${ROOT}/../JUCE"
    "${HOME}/external/JUCE"
    "${HOME}/JUCE"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if validate_juce_root "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

case "${CORE_BRIDGE_MODE}" in
  ON|OFF) ;;
  *)
    echo "Invalid KELLY_USE_CORE_BRIDGE value: ${CORE_BRIDGE_MODE} (expected ON or OFF)" >&2
    exit 2
    ;;
esac

echo "[1/5] Configure project"
CONFIG_ARGS=("-DKELLY_ENABLE_TESTS=ON" "-DKELLY_USE_CORE_BRIDGE=${CORE_BRIDGE_MODE}")
JUCE_ROOT="${JUCE_EXTERNAL_ROOT:-}"
if [[ -n "${JUCE_ROOT}" ]]; then
  if ! validate_juce_root "${JUCE_ROOT}"; then
    echo "Invalid JUCE_EXTERNAL_ROOT: ${JUCE_ROOT}" >&2
    echo "Set JUCE_EXTERNAL_ROOT to a full JUCE checkout (for example JUCE 7.0.9)." >&2
    exit 2
  fi
else
  if JUCE_ROOT="$(detect_local_juce_root)"; then
    echo "Auto-detected local JUCE checkout: ${JUCE_ROOT}"
  fi
fi

if [[ -n "${JUCE_ROOT}" ]]; then
  CONFIG_ARGS+=("-DJUCE_EXTERNAL_ROOT=${JUCE_ROOT}")
  CONFIG_ARGS+=("-DOFFLINE_BUILD=ON")
else
  CONFIG_ARGS+=("-DJUCE_EXTERNAL_ROOT=") # Clear any cached JUCE_EXTERNAL_ROOT in existing build dir.
  CONFIG_ARGS+=("-DOFFLINE_BUILD=OFF")
  echo "No local JUCE checkout detected; CMake will attempt to fetch JUCE from GitHub."
fi

if [[ "${CORE_BRIDGE_MODE}" == "ON" ]]; then
  CORE_DIR="${KELLY_CORE_DIR:-${ROOT}/../minimal_listening_core/c}"
  for required in "include/intent_engine.h" "include/intent_result.h" "include/value_state.h"; do
    if [[ ! -f "${CORE_DIR}/${required}" ]]; then
      echo "Invalid KELLY_CORE_DIR: missing ${required}" >&2
      echo "Set KELLY_CORE_DIR to minimal_listening_core/c before enabling bridge mode." >&2
      exit 2
    fi
  done
  if [[ ! -f "${CORE_DIR}/liblistening.a" && ! -f "${CORE_DIR}/Makefile" ]]; then
    echo "Invalid KELLY_CORE_DIR: requires liblistening.a or Makefile to build it." >&2
    exit 2
  fi
  CONFIG_ARGS+=("-DKELLY_CORE_DIR=${CORE_DIR}")
fi

if [[ ${#CONFIG_ARGS[@]} -gt 0 ]]; then
  cmake -S "${ROOT}" -B "${BUILD_DIR}" "${CONFIG_ARGS[@]}"
else
  cmake -S "${ROOT}" -B "${BUILD_DIR}"
fi

echo "[2/5] Build runtime contract tests"
cmake --build "${BUILD_DIR}" --target runtime_contract_tests

echo "[3/5] Run runtime contract tests"
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R runtime_contract_tests

echo "[4/5] Run listening contract audit"
"${ROOT}/audit_listening_contract.sh"

echo "[5/5] Verify skill handoff-state contract"
python3 "${ROOT}/scripts/check_skill_handoff_state.py" --repo-root "${ROOT}"

echo ""
echo "Contract checks complete: runtime tests + listening contract audit + handoff-state checks are green."
