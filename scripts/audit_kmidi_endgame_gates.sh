#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target-repo-root>" >&2
  exit 2
fi

TARGET_ROOT="$1"

INTENT_FILE="$TARGET_ROOT/src/engine/IntentPipeline.cpp"
CHORD_FILE="$TARGET_ROOT/src/midi/ChordGenerator.cpp"
PROCESSOR_FILE="$TARGET_ROOT/src/plugin/PluginProcessor.cpp"
EDITOR_FILE="$TARGET_ROOT/src/plugin/PluginEditor.cpp"
KTYPES_FILE="$TARGET_ROOT/src/common/KellyTypes.h"
TYPES_FILE="$TARGET_ROOT/src/common/Types.h"

for f in "$INTENT_FILE" "$CHORD_FILE" "$PROCESSOR_FILE" "$EDITOR_FILE" "$KTYPES_FILE" "$TYPES_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 3
  fi
done

pass_count=0
fail_count=0

status_icon() {
  if [[ "$1" == "PASS" ]]; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

emit_row() {
  local gate="$1"
  local status="$2"
  local evidence="$3"
  local fix="$4"
  echo "| $gate | $(status_icon "$status") | $evidence | $fix |"
  if [[ "$status" == "PASS" ]]; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
}

first_match_or_dash() {
  local pattern="$1"
  local file="$2"
  local out
  out="$(rg -n "$pattern" "$file" | head -n 1 || true)"
  if [[ -z "$out" ]]; then
    echo "-"
  else
    echo "\`$out\`"
  fi
}

echo "# KmiDi Listening Endgame Gate Report"
echo ""
echo "Target: \`$TARGET_ROOT\`"
echo ""
echo "| Gate | Status | Evidence | Required Fix |"
echo "|---|---|---|---|"

# Gate 1: Absence is signal (ABSTAIN)
if rg -n "abstain\(|ABSTAIN" "$INTENT_FILE" >/dev/null 2>&1; then
  emit_row \
    "I-3/R-1 Empty input abstains" \
    "PASS" \
    "$(first_match_or_dash "abstain\(|ABSTAIN" "$INTENT_FILE")" \
    "Keep abstain behavior enforced by tests."
else
  emit_row \
    "I-3/R-1 Empty input abstains" \
    "FAIL" \
    "No \`abstain\` path found in \`src/engine/IntentPipeline.cpp\`." \
    "Add explicit empty/ambiguous input checks and \`IntentResult::abstain()\` return path."
fi

# Gate 2: Constraints override everything (INVALID)
if rg -n "return IntentResult::invalid\(\)" "$INTENT_FILE" >/dev/null 2>&1; then
  emit_row \
    "I-4 Constraint violations hard-stop output" \
    "PASS" \
    "$(first_match_or_dash "return IntentResult::invalid\(\)" "$INTENT_FILE")" \
    "Keep invalid gating in all generation paths."
else
  emit_row \
    "I-4 Constraint violations hard-stop output" \
    "FAIL" \
    "No \`return IntentResult::invalid()\` found in \`src/engine/IntentPipeline.cpp\`." \
    "Promote constraint/rule-break violations to hard-fail INVALID before MIDI generation."
fi

# Gate 3: No default escalation
if rg -n "escalationTokenPresent" "$CHORD_FILE" >/dev/null 2>&1; then
  emit_row \
    "I-2/E-1 No harmonic escalation without explicit token" \
    "PASS" \
    "$(first_match_or_dash "escalationTokenPresent" "$CHORD_FILE")" \
    "Keep token-gated extension logic only."
else
  local_ext_line="$(first_match_or_dash "CHORD_EXTENSION_INTENSITY_THRESHOLD|addExtensions" "$CHORD_FILE")"
  emit_row \
    "I-2/E-1 No harmonic escalation without explicit token" \
    "FAIL" \
    "Extension logic appears intensity/default-driven: $local_ext_line" \
    "Replace intensity-driven extension enablement with explicit escalation-token guard."
fi

# Gate 4: No synthetic fallback intent
synthetic_line_1="$(first_match_or_dash "wound\\.description = \"emotional state\"" "$PROCESSOR_FILE")"
synthetic_line_2="$(first_match_or_dash "sideB\\.description = \"musical expression\"" "$PROCESSOR_FILE")"
if [[ "$synthetic_line_1" == "-" && "$synthetic_line_2" == "-" ]]; then
  emit_row \
    "I-3 No synthetic intent substitution" \
    "PASS" \
    "No synthetic fallback descriptors found." \
    "Keep abstain/clarify behavior instead of fallback generation."
else
  emit_row \
    "I-3 No synthetic intent substitution" \
    "FAIL" \
    "Synthetic fallbacks found: $synthetic_line_1; $synthetic_line_2" \
    "When input is absent/underspecified, abstain instead of fabricating side/wound descriptions."
fi

# Gate 5: No auto playback from generation path
if rg -n "startPlayback\(" "$TARGET_ROOT/src/plugin" -g '!**/*.complex_backup' >/dev/null 2>&1; then
  emit_row \
    "P-1 No auto-playback on generate" \
    "FAIL" \
    "$(rg -n "startPlayback\(" "$TARGET_ROOT/src/plugin" -g '!**/*.complex_backup' | head -n 1 | sed 's/.*/`&`/')" \
    "Remove non-user playback calls; enforce explicit user action only."
else
  emit_row \
    "P-1 No auto-playback on generate" \
    "PASS" \
    "No active \`startPlayback()\` usage found in \`src/plugin\`." \
    "Keep playback opt-in only."
fi

# Gate 6: Global output gate (canProduceOutput)
if rg -n "canProduceOutput" "$TARGET_ROOT/src/common" "$TARGET_ROOT/src/plugin" "$TARGET_ROOT/src/midi" >/dev/null 2>&1; then
  emit_row \
    "Global canProduceOutput gate present in output paths" \
    "PASS" \
    "$(rg -n "canProduceOutput" "$TARGET_ROOT/src/common" "$TARGET_ROOT/src/plugin" "$TARGET_ROOT/src/midi" | head -n 1 | sed 's/.*/`&`/')" \
    "Retain gate check before generation/playback."
else
  emit_row \
    "Global canProduceOutput gate present in output paths" \
    "FAIL" \
    "No \`canProduceOutput\` symbol found across common/plugin/midi modules." \
    "Add \`canProduceOutput(IntentResult)\` and enforce it in processor + generators."
fi

# Gate 7: No invention defaults (optional melody/bass)
if rg -n "std::optional<.*melody|std::optional<.*bass" "$KTYPES_FILE" >/dev/null 2>&1; then
  emit_row \
    "I-1 No invention defaults for optional layers" \
    "PASS" \
    "$(first_match_or_dash "std::optional<.*melody|std::optional<.*bass" "$KTYPES_FILE")" \
    "Keep optional layers explicit."
else
  emit_row \
    "I-1 No invention defaults for optional layers" \
    "FAIL" \
    "$(first_match_or_dash "std::vector<MidiNote> melody|std::vector<MidiNote> bass" "$KTYPES_FILE")" \
    "Convert melody/bass (and optional layers) to \`std::optional\` and only emit when explicitly requested."
fi

# Gate 8: UI abstains on empty request
if rg -n "woundText\\.isEmpty\\(\\).*return" "$EDITOR_FILE" >/dev/null 2>&1; then
  emit_row \
    "R-1 UI abstains when request input is empty" \
    "PASS" \
    "$(first_match_or_dash "woundText\\.isEmpty\\(\\).*return" "$EDITOR_FILE")" \
    "Keep empty-input early return."
else
  emit_row \
    "R-1 UI abstains when request input is empty" \
    "FAIL" \
    "$(first_match_or_dash "void PluginEditor::onGenerateClicked\\(" "$EDITOR_FILE") + no early empty-input return" \
    "Add explicit empty-input return in \`onGenerateClicked\` before \`generateMidi()\`."
fi

echo ""
echo "## Summary"
echo ""
echo "- PASS: $pass_count"
echo "- FAIL: $fail_count"
echo "- Endgame readiness: $([[ $fail_count -eq 0 ]] && echo "READY" || echo "NOT READY")"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi

