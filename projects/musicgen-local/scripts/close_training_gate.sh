#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROV="$ROOT_DIR/docs/ml/DATA_PROVENANCE.md"

if [[ ! -f "$PROV" ]]; then
  echo "FAIL: missing provenance doc: $PROV"
  exit 2
fi

if rg -q '^- ALLOW_TRAINING=YES' "$PROV"; then
  perl -0777 -i -pe 's#^- ALLOW_TRAINING=YES \(approved in APPROVAL_CHECKLIST\.md\)\.#- ALLOW_TRAINING=NO for all imported assets until license/source terms are explicitly documented.#m' "$PROV"
  echo "PASS: training gate closed"
else
  if rg -q '^- ALLOW_TRAINING=NO ' "$PROV"; then
    echo "PASS: training gate already closed"
  else
    echo "FAIL: expected ALLOW_TRAINING line not found"
    exit 1
  fi
fi
