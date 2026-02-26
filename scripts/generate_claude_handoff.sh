#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${ROOT}/output/claude"
STATUS_FILE="${OUT_DIR}/git_status_short.txt"
REPORT_FILE="${OUT_DIR}/handoff_summary.md"

mkdir -p "${OUT_DIR}"
git -C "${ROOT}" status --short > "${STATUS_FILE}" || true

cat > "${REPORT_FILE}" <<EOF
# Codex/Claude Handoff Summary

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Runtime Mode
- Feature flags were removed.
- Adapter registry is deterministic and always includes no-op + emotion thesaurus enrichment.
- Only 'KELLY_USE_CORE_BRIDGE' remains as a supported mode switch.

## Verification Commands
- './scripts/run_contract_checks.sh'
- './scripts/ci_bridge_matrix.sh'
- './scripts/build_release_evidence.sh'

## Key Artifacts
- Canonical all-off contract check: 'output/claude/canonical_all_off.log'
- Canonical emotion behavior check: 'output/claude/canonical_emotion_on.log'
- Runtime tests all-off: 'output/claude/runtime_contract_tests_all_off.log'
- Runtime tests emotion-on compatibility lane: 'output/claude/runtime_contract_tests_emotion_on.log'

## Open Items / Risks
- Worktree is very dirty; stage only intentional files for merge.
- If external automation referenced removed feature-flag scripts/workflows, it must be updated.

## Latest git status --short
See: 'output/claude/git_status_short.txt'
EOF

printf "Wrote handoff report: %s\n" "${REPORT_FILE}"
