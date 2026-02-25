#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${1:-artifacts/debug_baseline}"
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/*.log "${LOG_DIR}/status.txt" "${LOG_DIR}/bug_taxonomy_report.md"

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

run_cmd cmake_config cmake -S . -B build/diag-debug -G Ninja -DBUILD_TESTS=ON -DBUILD_DESKTOP=ON -DBUILD_PLUGINS=OFF
run_cmd cmake_build cmake --build build/diag-debug --target KellyCore
run_cmd ctest ctest --test-dir build/diag-debug --output-on-failure
run_cmd pytest pytest tests -q
run_cmd npm_build npm run build
run_cmd cargo_check cargo check --manifest-path src-tauri/Cargo.toml --offline --frozen

python3 scripts/diagnostics/bug_taxonomy_parser.py --logs-dir "${LOG_DIR}" --out "${LOG_DIR}/bug_taxonomy_report.md"
echo "Done. Report: ${LOG_DIR}/bug_taxonomy_report.md"
