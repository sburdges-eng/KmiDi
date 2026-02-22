#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKLIST="$ROOT_DIR/docs/ml/APPROVAL_CHECKLIST.md"
MIDI_SRC_MAN="$ROOT_DIR/ml/data/manifests/sources/midi/2026-02-13-midi-sources.txt"
MODEL_SRC_MAN="$ROOT_DIR/ml/data/manifests/sources/models/2026-02-13-model-artifacts.txt"
OUT_DIR="$ROOT_DIR/ml/data/manifests/active"
mkdir -p "$OUT_DIR"

# Build list of approved roots (commercial YES + training YES)
approved_roots=$(awk -F'|' '
  BEGIN {in_table=0}
  /^\| Source Root \| Asset Type \| Evidence Doc \| Owner \| Approval Date \| Allowed Uses \| Commercial Use \| Training Allowed \| Notes \|$/ {in_table=1; next}
  in_table && /^\|---/ {next}
  in_table && /^\|/ {
    root=$2; comm=$8; train=$9;
    gsub(/^ +| +$/, "", root);
    gsub(/^ +| +$/, "", comm);
    gsub(/^ +| +$/, "", train);
    gsub(/`/, "", root);
    if (comm=="YES" && train=="YES") print root;
  }
  in_table && !/^\|/ {exit}
' "$CHECKLIST" | sed '/^$/d' | sort -u)

if [[ -z "$approved_roots" ]]; then
  echo "FAIL: no approved roots found in checklist"
  exit 1
fi

MIDI_OUT="$OUT_DIR/midi-approved.txt"
MODEL_OUT="$OUT_DIR/models-approved.txt"
: > "$MIDI_OUT"
: > "$MODEL_OUT"

while IFS= read -r r; do
  [[ -z "$r" ]] && continue
  rg -N "^$r/" "$MIDI_SRC_MAN" >> "$MIDI_OUT" || true
  rg -N "^$r/" "$MODEL_SRC_MAN" >> "$MODEL_OUT" || true
done <<< "$approved_roots"

LC_ALL=C sort -u "$MIDI_OUT" -o "$MIDI_OUT"
LC_ALL=C sort -u "$MODEL_OUT" -o "$MODEL_OUT"

echo "approved_roots=$(echo "$approved_roots" | wc -l | tr -d ' ')"
echo "midi_entries=$(wc -l < "$MIDI_OUT" | tr -d ' ')"
echo "model_entries=$(wc -l < "$MODEL_OUT" | tr -d ' ')"
echo "midi_manifest=$MIDI_OUT"
echo "model_manifest=$MODEL_OUT"
