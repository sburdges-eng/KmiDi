#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${1:-artifacts/debug_baseline}"
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/*.log "${LOG_DIR}/status.txt" "${LOG_DIR}/bug_taxonomy_report.md" \
  "${LOG_DIR}/bug_taxonomy_baseline_candidate.txt"

run_cmd() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"

  echo "[$(date -u +%FT%TZ)] START ${name}" | tee -a "${LOG_DIR}/status.txt"
  set +e
  "$@" >"${log}" 2>&1
  local rc=$?
  set -e
  echo "[$(date -u +%FT%TZ)] END ${name} rc=${rc}" | tee -a "${LOG_DIR}/status.txt"
}

# BUILD_DESKTOP is force-disabled without KMIDI_BUILD_QT_UI (v1 stabilization), so it is
# not passed here — the diagnostic build must not require Qt. KellyFFI is off because the
# Rust layer is covered by cargo_check below; building it here would double the cost.
run_cmd cmake_config cmake -S . -B build/diag-debug -G Ninja -DBUILD_TESTS=ON -DBUILD_PLUGINS=OFF -DBUILD_KELLY_FFI=OFF
# Build everything (not just KellyCore) so the test executables registered by
# BUILD_TESTS=ON exist when ctest runs.
run_cmd cmake_build cmake --build build/diag-debug
run_cmd ctest ctest --test-dir build/diag-debug --output-on-failure --no-tests=error
run_cmd pytest pytest tests -q
run_cmd npm_build npm run build
# --locked (not --offline --frozen): the registry is not vendored, so cbindgen and friends
# must be fetchable; Cargo.lock still pins every version.
run_cmd cargo_check cargo check --manifest-path engine/intent_ir/Cargo.toml --locked

python3 scripts/diagnostics/bug_taxonomy_parser.py --logs-dir "${LOG_DIR}" \
  --out "${LOG_DIR}/bug_taxonomy_report.md" \
  --counts-out "${LOG_DIR}/bug_taxonomy_baseline_candidate.txt"
echo "Done. Report: ${LOG_DIR}/bug_taxonomy_report.md"
