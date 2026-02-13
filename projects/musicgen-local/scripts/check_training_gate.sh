#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKLIST="$ROOT_DIR/docs/ml/APPROVAL_CHECKLIST.md"

if [[ ! -f "$CHECKLIST" ]]; then
  echo "FAIL: missing approval checklist: $CHECKLIST"
  exit 2
fi

# Parse markdown table rows for source signoff matrix.
# Expected columns:
# Source Root | Asset Type | Evidence Doc | Owner | Approval Date | Allowed Uses | Commercial Use | Training Allowed | Notes
rows=$(awk '
  BEGIN {in_table=0}
  /^\| Source Root \| Asset Type \| Evidence Doc \| Owner \| Approval Date \| Allowed Uses \| Commercial Use \| Training Allowed \| Notes \|$/ {in_table=1; next}
  in_table && /^\|---/ {next}
  in_table && /^\|/ {print}
  in_table && !/^\|/ {exit}
' "$CHECKLIST")

if [[ -z "${rows}" ]]; then
  echo "FAIL: could not parse signoff matrix rows in $CHECKLIST"
  exit 2
fi

fail=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  # Strip leading/trailing pipe and split.
  IFS='|' read -r _ c1 c2 c3 c4 c5 c6 c7 c8 c9 _ <<< "$line"
  source_root="$(echo "$c1" | xargs)"
  owner="$(echo "$c4" | xargs)"
  approval_date="$(echo "$c5" | xargs)"
  allowed_uses="$(echo "$c6" | xargs)"
  commercial="$(echo "$c7" | xargs)"
  training="$(echo "$c8" | xargs)"

  # Skip placeholder/header-like rows if any
  [[ "$source_root" == "Source Root" ]] && continue

  # Enforce explicit YES for commercial+training and non-empty owner/date/uses.
  if [[ "$commercial" != "YES" || "$training" != "YES" || -z "$owner" || -z "$approval_date" || -z "$allowed_uses" ]]; then
    echo "BLOCKED: $source_root"
    echo "  owner=$owner approval_date=$approval_date allowed_uses=$allowed_uses commercial=$commercial training=$training"
    fail=1
  fi
done <<< "$rows"

if [[ "$fail" -ne 0 ]]; then
  echo "FAIL: training gate closed (check $CHECKLIST)"
  exit 1
fi

echo "PASS: training gate open"
