#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

required_docs=(
  "docs/ARCHITECTURE.md"
  "docs/REPO_MODULE_MAP.md"
  "docs/INTENT_IR_AUTHORITY.md"
  "docs/NATIVE_RUNTIME_OWNERSHIP.md"
  "docs/JUCE_RT_RULES.md"
  "docs/FFI_OWNERSHIP_AND_ABI.md"
  "docs/PERSISTENCE_AND_MIGRATION.md"
  "docs/AGENT_ALLOWED_SURFACES.md"
  "docs/HUMAN_OWNED_SURFACES.md"
  "docs/ARCHITECTURE_REVIEW_CHECKLIST.md"
)

fail() {
  echo "ARCHITECTURE CHECK FAILED: $*" >&2
  exit 1
}

for doc in "${required_docs[@]}"; do
  [[ -f "$doc" ]] || fail "missing required doc: $doc"
done

grep -q "Status: canonical handoff for completed workbook passes A, B, C, D, E, F, and G" docs/ARCHITECTURE.md \
  || fail "docs/ARCHITECTURE.md does not advertise completed passes A-G"

grep -q "Status: authoritative for approved workbook Pass B" docs/REPO_MODULE_MAP.md \
  || fail "docs/REPO_MODULE_MAP.md missing Pass B authoritative status"

grep -q "This handoff currently leaves implementation drift and future feature details open, but no workbook pass remains open" docs/ARCHITECTURE.md \
  || fail "docs/ARCHITECTURE.md still looks like it has open workbook-pass gaps"

for required_ref in \
  'docs/REPO_MODULE_MAP.md' \
  'docs/INTENT_IR_AUTHORITY.md' \
  'docs/NATIVE_RUNTIME_OWNERSHIP.md' \
  'docs/JUCE_RT_RULES.md' \
  'docs/FFI_OWNERSHIP_AND_ABI.md' \
  'docs/PERSISTENCE_AND_MIGRATION.md'; do
  grep -q "$required_ref" docs/ARCHITECTURE.md \
    || fail "docs/ARCHITECTURE.md does not reference $required_ref"
done

echo "ARCHITECTURE CHECK OK"
