#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKLIST="$ROOT_DIR/docs/ml/APPROVAL_CHECKLIST.md"
PROV="$ROOT_DIR/docs/ml/DATA_PROVENANCE.md"

if [[ ! -f "$CHECKLIST" ]]; then
  echo "FAIL: missing checklist: $CHECKLIST"
  exit 2
fi
if [[ ! -f "$PROV" ]]; then
  echo "FAIL: missing provenance doc: $PROV"
  exit 2
fi

# Reuse gate script for validation.
"$ROOT_DIR/scripts/check_training_gate.sh" >/tmp/check_training_gate.out 2>&1 || {
  cat /tmp/check_training_gate.out
  echo "FAIL: cannot open gate; checklist not fully approved"
  exit 1
}

# Flip only the explicit gate line.
if rg -q '^- ALLOW_TRAINING=NO ' "$PROV"; then
  perl -0777 -i -pe 's#^- ALLOW_TRAINING=NO for all imported assets until license/source terms are explicitly documented\.#- ALLOW_TRAINING=YES (approved in APPROVAL_CHECKLIST.md).#m' "$PROV"
  echo "PASS: training gate opened in DATA_PROVENANCE.md"
else
  if rg -q '^- ALLOW_TRAINING=YES' "$PROV"; then
    echo "PASS: training gate already open"
  else
    echo "FAIL: expected ALLOW_TRAINING gate line not found in $PROV"
    exit 1
  fi
fi
