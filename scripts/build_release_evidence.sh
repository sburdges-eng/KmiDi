#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ROOT}/output/release/${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

INTERACTION_INPUT="${INTERACTION_INPUT:-${ROOT}/output/release/interaction_assertions.jsonl}"

run_and_capture() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/${name}.log"
  if "$@" >"${logfile}" 2>&1; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

write_metadata() {
  {
    echo "generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cwd=${ROOT}"
    echo "branch=$(git -C "${ROOT}" branch --show-current || true)"
    echo "commit=$(git -C "${ROOT}" rev-parse HEAD || true)"
  } >"${OUT_DIR}/metadata.txt"
  git -C "${ROOT}" status --short > "${OUT_DIR}/git_status.txt" || true
}

build_kpi() {
  python3 "${ROOT}/scripts/generate_trust_kpi_report.py" \
    --input "${INTERACTION_INPUT}" \
    --output "${OUT_DIR}/trust_kpi_report.md" \
    --window-label "release-candidate-window"
}

write_known_risks() {
  cat > "${OUT_DIR}/known_risks.md" <<'MD'
# Known Risks

- Runtime test and bridge matrix execution may be blocked without a full local JUCE checkout when offline.
- DAW transport integration checks are not covered by static listening contract audit.
- Bridge ON lane depends on `minimal_listening_core/c` headers and `liblistening.a` build availability.
- Trust KPI confidence depends on interaction sample size and assertion quality.
MD
}

write_summary() {
  local contract_audit="$1"
  local audit="$2"
  local security="$3"
  local runtime="$4"
  local matrix="$5"
  local overall="PASS"
  if [[ "${contract_audit}" != "PASS" || "${audit}" != "PASS" || "${security}" != "PASS" ]]; then
    overall="FAIL"
  elif [[ "${runtime}" != "PASS" || "${matrix}" != "PASS" ]]; then
    overall="CONDITIONAL"
  fi

  cat > "${OUT_DIR}/release_signoff.md" <<MD
# Release Sign-off Summary

- Evidence generated: ${STAMP}
- Branch: $(git -C "${ROOT}" branch --show-current || echo unknown)
- Commit: $(git -C "${ROOT}" rev-parse HEAD || echo unknown)

## Checks

| Check | Status | Log |
|---|---|---|
| Listening contract audit | ${contract_audit} | logs/contract_audit.log |
| Contract violation audit | ${audit} | logs/audit.log |
| Security and Compliance Preflight | ${security} | logs/security_compliance.log |
| Runtime Contract Checks | ${runtime} | logs/runtime_contract_checks.log |
| Bridge Matrix (OFF/ON) | ${matrix} | logs/bridge_matrix.log |

## KPI

- Report: trust_kpi_report.md
- Input: ${INTERACTION_INPUT}

## Known Risks

- See: known_risks.md
- Security preflight report: security_compliance_report.md

## Sign-off Gate

- Overall status: ${overall}
- Rule: listening contract audit, contract violation audit, and security preflight must pass. Runtime/matrix failures downgrade status to CONDITIONAL until rerun on a fully provisioned environment.
MD
}

echo "Generating release evidence at: ${OUT_DIR}"
write_metadata

CONTRACT_AUDIT_STATUS="$(run_and_capture contract_audit "${ROOT}/audit_listening_contract.sh")"
AUDIT_STATUS="$(run_and_capture audit python3 "${ROOT}/scripts/audit_contract_violations.py" --repo-root "${ROOT}")"
SECURITY_STATUS="$(
  run_and_capture security_compliance \
  "${ROOT}/scripts/security_compliance_preflight.sh" \
  --repo-root "${ROOT}" \
  --report "${OUT_DIR}/security_compliance_report.md"
)"
RUNTIME_STATUS="$(run_and_capture runtime_contract_checks "${ROOT}/scripts/run_contract_checks.sh")"
MATRIX_STATUS="$(run_and_capture bridge_matrix "${ROOT}/scripts/ci_bridge_matrix.sh")"

build_kpi > "${LOG_DIR}/kpi.log" 2>&1 || true
write_known_risks
write_summary "${CONTRACT_AUDIT_STATUS}" "${AUDIT_STATUS}" "${SECURITY_STATUS}" "${RUNTIME_STATUS}" "${MATRIX_STATUS}"

echo "Release evidence complete."
echo "Artifacts: ${OUT_DIR}"
