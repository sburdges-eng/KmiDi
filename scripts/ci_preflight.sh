#!/usr/bin/env bash
# Run the same CI preflight checks locally.
# Usage: bash scripts/ci_preflight.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FAILED=0

cd "${ROOT}"

echo "═══════════════════════════════════════════════════"
echo " CI Preflight Gate — Local Runner"
echo "═══════════════════════════════════════════════════"

# ── G1: Manifest schema validation ──────────────────────
echo ""
echo "[G1] Manifest schema validation"
if python3 -m pytest tests/test_manifest_integrity.py -v --tb=short 2>&1; then
    echo "PASS [G1]"
else
    echo "FAIL [G1]"
    FAILED=1
fi

# ── G2: Shard naming + checksums ────────────────────────
echo ""
echo "[G2] Shard naming & checksum fixture check"
if python3 -m pytest tests/test_ci_shard_naming.py -v --tb=short 2>&1; then
    echo "PASS [G2]"
else
    echo "FAIL [G2]"
    FAILED=1
fi

# ── G3: Teacher fetch guardrail ─────────────────────────
echo ""
echo "[G3] Teacher fetch guardrail"
if python3 -m pytest tests/test_ci_teacher_guardrails.py -v --tb=short 2>&1; then
    echo "PASS [G3]"
else
    echo "FAIL [G3]"
    FAILED=1
fi

# ── G4: No teacher prefix in local-safe paths ──────────
echo ""
echo "[G4] No teacher prefix in local-safe defaults"
if python3 tests/test_ci_no_teacher_in_local.py 2>&1; then
    echo "PASS [G4]"
else
    echo "FAIL [G4]"
    FAILED=1
fi

# ── G5: Run contract validation ────────────────────────
echo ""
echo "[G5] Run contract required keys"
if python3 -m pytest tests/test_ci_run_contract.py -v --tb=short 2>&1; then
    echo "PASS [G5]"
else
    echo "FAIL [G5]"
    FAILED=1
fi

# ── G6: Pipeline unit tests ───────────────────────────
echo ""
echo "[G6] Pipeline unit tests"
if python3 -m pytest tests/test_splits_deterministic.py tests/test_license_gate.py tests/test_incremental_sync.py -v --tb=short 2>&1; then
    echo "PASS [G6]"
else
    echo "FAIL [G6]"
    FAILED=1
fi

# ── G7: Listening contract audit ───────────────────────
echo ""
echo "[G7] Listening contract audit"
if command -v rg >/dev/null 2>&1; then
    bash "${ROOT}/audit_listening_contract.sh" 2>&1
    echo "PASS [G7] (audit report only; does not fail build)"
else
    echo "SKIP [G7]: ripgrep (rg) not installed"
fi

echo ""
echo "═══════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
    echo " All preflight gates PASSED."
else
    echo " PREFLIGHT FAILED — fix issues before pushing."
    exit 1
fi
