#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_PATH=""
FAILED=0
RESULTS=()

usage() {
  cat <<'EOF'
Usage: security_compliance_preflight.sh [--repo-root <path>] [--report <path>]

Runs pre-deployment security/compliance checks:
- hardcoded secret pattern scan (source + scripts)
- local secret file ignore rules
- dependency pin sanity (JUCE GIT_TAG semver pin)
- network surface compile flags disabled by default
- optional core bridge dependency guards
- strict shell mode in critical CI/release scripts
- shared skill handoff-state contract consistency
EOF
}

add_result() {
  local check_name="$1"
  local status="$2"
  local detail="$3"
  RESULTS+=("${check_name}|${status}|${detail}")
  if [[ "${status}" != "PASS" ]]; then
    FAILED=1
  fi
}

check_secret_patterns() {
  local scan_paths=(
    "${ROOT}/CMakeLists.txt"
    "${ROOT}/common"
    "${ROOT}/engine"
    "${ROOT}/plugin"
    "${ROOT}/midi"
    "${ROOT}/audio"
    "${ROOT}/adapters"
    "${ROOT}/interfaces"
    "${ROOT}/scripts"
    "${ROOT}/tests"
    "${ROOT}/cmake"
    "${ROOT}/.github"
    "${ROOT}/README.md"
  )
  local existing_paths=()
  local path
  local matches

  for path in "${scan_paths[@]}"; do
    if [[ -e "${path}" ]]; then
      existing_paths+=("${path}")
    fi
  done

  if [[ ${#existing_paths[@]} -eq 0 ]]; then
    add_result "Secret Scan" "WARN" "No scan paths found."
    return
  fi

  matches="$(
    rg -n -H -i \
      -e 'AKIA[0-9A-Z]{16}' \
      -e 'ghp_[A-Za-z0-9]{36}' \
      -e 'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY' \
      -e '(api[_-]?key|secret|password|access[_-]?token|auth[_-]?token)[[:space:]]*[:=][[:space:]]*"[^"]{12,}"' \
      "${existing_paths[@]}" || true
  )"

  if [[ -n "${matches}" ]]; then
    local sample
    sample="$(printf '%s\n' "${matches}" | sed -n '1,5p')"
    add_result "Secret Scan" "FAIL" "Potential secret patterns found. Sample: ${sample}"
    return
  fi

  add_result "Secret Scan" "PASS" "No hardcoded secret patterns detected in source and scripts."
}

check_local_secret_gitignore() {
  if [[ ! -f "${ROOT}/.gitignore" ]]; then
    add_result "Local Secret Ignore" "FAIL" ".gitignore not found."
    return
  fi

  if grep -qx '\.env' "${ROOT}/.gitignore" && grep -qx '\.env\.\*' "${ROOT}/.gitignore"; then
    add_result "Local Secret Ignore" "PASS" ".env and .env.* are ignored."
  else
    add_result "Local Secret Ignore" "FAIL" "Add .env and .env.* to .gitignore."
  fi
}

check_dependency_pin() {
  if [[ ! -f "${ROOT}/CMakeLists.txt" ]]; then
    add_result "Dependency Pin" "FAIL" "CMakeLists.txt not found."
    return
  fi

  local tag_line
  tag_line="$(rg -n 'GIT_TAG[[:space:]]+[0-9]+\.[0-9]+\.[0-9]+' "${ROOT}/CMakeLists.txt" | head -n 1 || true)"
  if [[ -n "${tag_line}" ]]; then
    add_result "Dependency Pin" "PASS" "Pinned JUCE tag found (${tag_line})."
  else
    add_result "Dependency Pin" "FAIL" "Expected JUCE GIT_TAG semver pin in CMakeLists.txt."
  fi
}

check_network_surface_defaults() {
  if [[ ! -f "${ROOT}/CMakeLists.txt" ]]; then
    add_result "Network Surface" "FAIL" "CMakeLists.txt not found."
    return
  fi

  if rg -q 'JUCE_WEB_BROWSER=0' "${ROOT}/CMakeLists.txt" && rg -q 'JUCE_USE_CURL=0' "${ROOT}/CMakeLists.txt"; then
    add_result "Network Surface" "PASS" "JUCE web/curl features disabled by default."
  else
    add_result "Network Surface" "FAIL" "Expected JUCE_WEB_BROWSER=0 and JUCE_USE_CURL=0 compile defs."
  fi
}

check_bridge_dependency_guards() {
  if [[ ! -f "${ROOT}/CMakeLists.txt" ]]; then
    add_result "Bridge Guardrails" "FAIL" "CMakeLists.txt not found."
    return
  fi

  if rg -q 'KELLY_CORE_DIR does not exist' "${ROOT}/CMakeLists.txt" \
    && rg -q 'missing include/intent_engine.h' "${ROOT}/CMakeLists.txt" \
    && rg -q 'missing include/intent_result.h' "${ROOT}/CMakeLists.txt"; then
    add_result "Bridge Guardrails" "PASS" "Bridge mode validates dependency presence before build."
  else
    add_result "Bridge Guardrails" "FAIL" "Missing required bridge dependency validation checks."
  fi
}

check_script_hardening() {
  local scripts_to_check=(
    "${ROOT}/audit_listening_contract.sh"
    "${ROOT}/scripts/run_contract_checks.sh"
    "${ROOT}/scripts/ci_bridge_matrix.sh"
    "${ROOT}/scripts/build_release_evidence.sh"
  )
  local script
  local missing=()

  for script in "${scripts_to_check[@]}"; do
    if [[ ! -f "${script}" ]]; then
      missing+=("${script##${ROOT}/}(missing)")
      continue
    fi
    if ! rg -q 'set -euo pipefail' "${script}"; then
      missing+=("${script##${ROOT}/}")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    add_result "Script Hardening" "FAIL" "Expected strict mode in: $(IFS=', '; echo "${missing[*]}")"
  else
    add_result "Script Hardening" "PASS" "Critical scripts use set -euo pipefail."
  fi
}

check_skill_handoff_state_contract() {
  local checker="${ROOT}/scripts/check_skill_handoff_state.py"
  if [[ ! -f "${checker}" ]]; then
    add_result "Skill Handoff State" "FAIL" "Missing required checker script: scripts/check_skill_handoff_state.py"
    return
  fi

  local output=""
  if output="$(python3 "${checker}" --repo-root "${ROOT}" 2>&1)"; then
    add_result "Skill Handoff State" "PASS" "${output}"
    return
  fi

  local first_line
  first_line="$(printf '%s\n' "${output}" | sed -n '1p')"
  if [[ -z "${first_line}" ]]; then
    first_line="Shared handoff-state contract check failed."
  fi
  add_result "Skill Handoff State" "FAIL" "${first_line}"
}

write_report() {
  local status_label="PASS"
  if [[ ${FAILED} -ne 0 ]]; then
    status_label="FAIL"
  fi

  {
    echo "# Security and Compliance Preflight"
    echo
    echo "- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- Repository: ${ROOT}"
    echo "- Overall status: ${status_label}"
    echo
    echo "| Check | Status | Detail |"
    echo "|---|---|---|"
    local row
    for row in "${RESULTS[@]}"; do
      IFS='|' read -r check_name status detail <<<"${row}"
      detail="${detail//$'\n'/<br>}"
      echo "| ${check_name} | ${status} | ${detail} |"
    done
  } >"${REPORT_PATH}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      ROOT="$2"
      shift 2
      ;;
    --report)
      REPORT_PATH="$2"
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

check_run_contract_placeholders() {
  local contract_path="${ROOT}/config/run_contract.yaml"
  if [[ ! -f "${contract_path}" ]]; then
    add_result "Run Contract" "WARN" "config/run_contract.yaml not found."
    return
  fi

  local output
  if output="$(python3 - "${ROOT}" "${contract_path}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
contract_path = Path(sys.argv[2])
sys.path.insert(0, str(root))

from scripts.training_common import load_yaml_or_json, validate_run_contract  # type: ignore

payload = load_yaml_or_json(contract_path)
if not isinstance(payload, dict):
    raise ValueError("run contract root must be an object")
validate_run_contract(payload)
print("run contract semantic validation passed")
PY
  )"; then
    add_result "Run Contract" "PASS" "${output}"
  else
    local detail
    detail="$(printf '%s\n' "${output}" | sed -n '1p')"
    if [[ -z "${detail}" ]]; then
      detail="run contract semantic validation failed"
    fi
    add_result "Run Contract" "FAIL" "${detail}"
  fi
}

check_run_contract_placeholders
check_secret_patterns
check_local_secret_gitignore
check_dependency_pin
check_network_surface_defaults
check_bridge_dependency_guards
check_script_hardening
check_skill_handoff_state_contract

for row in "${RESULTS[@]}"; do
  IFS='|' read -r check_name status detail <<<"${row}"
  printf '%-22s %-5s %s\n' "${check_name}" "${status}" "${detail}"
done

if [[ -n "${REPORT_PATH}" ]]; then
  write_report
  echo "Report: ${REPORT_PATH}"
fi

if [[ ${FAILED} -ne 0 ]]; then
  exit 1
fi
