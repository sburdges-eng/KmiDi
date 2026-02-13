#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROV="$ROOT_DIR/docs/ml/DATA_PROVENANCE.md"
CHECKLIST="$ROOT_DIR/docs/ml/APPROVAL_CHECKLIST.md"

if [[ ! -f "$PROV" ]]; then
  echo "FAIL: missing provenance doc"
  exit 2
fi

echo "Gate line:"
rg '^- ALLOW_TRAINING=' "$PROV" || true

echo
echo "Checklist validation:"
if "$ROOT_DIR/scripts/check_training_gate.sh" >/tmp/check_training_gate.out 2>&1; then
  echo "PASS"
else
  echo "BLOCKED"
  cat /tmp/check_training_gate.out
fi
