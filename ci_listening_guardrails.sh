#!/usr/bin/env bash
# Static Listening Guardrails
# Ensures no forbidden patterns or accidental hardcoded paths remain in the codebase.
set -euo pipefail

echo "=> Running Static Listening Guardrails..."

SCAN_SCOPE="${GUARDRAILS_SCAN_SCOPE:-changed}"

collect_files_to_scan() {
  local range=""
  local zero_sha="0000000000000000000000000000000000000000"

  if [[ "$SCAN_SCOPE" == "all" ]]; then
    git ls-files
    return
  fi

  if [[ -n "${GITHUB_BASE_REF:-}" ]] && git rev-parse --verify "origin/${GITHUB_BASE_REF}" >/dev/null 2>&1; then
    range="origin/${GITHUB_BASE_REF}...HEAD"
  elif [[ -n "${GITHUB_EVENT_BEFORE:-}" && "${GITHUB_EVENT_BEFORE}" != "$zero_sha" ]] \
    && git rev-parse --verify "${GITHUB_EVENT_BEFORE}^{commit}" >/dev/null 2>&1; then
    range="${GITHUB_EVENT_BEFORE}...HEAD"
  elif git rev-parse --verify "HEAD~1" >/dev/null 2>&1; then
    range="HEAD~1...HEAD"
  fi

  if [[ -n "$range" ]]; then
    git diff --name-only --diff-filter=ACMRT "$range"
  else
    # Local fallback: scan only staged/unstaged tracked edits.
    git diff --name-only --diff-filter=ACMRT
    git diff --name-only --diff-filter=ACMRT --cached
  fi
}

mapfile -t FILES_TO_SCAN < <(
  collect_files_to_scan \
    | sort -u \
    | while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        [[ "$file" == "ci_listening_guardrails.sh" ]] && continue
        [[ "$file" == ".agents/"* ]] && continue
        [[ -f "$file" ]] || continue
        printf '%s\n' "$file"
      done
)

# 1) Check for hardcoded local paths that shouldn't be in CI
if [[ "${#FILES_TO_SCAN[@]}" -eq 0 ]]; then
  echo "-> No eligible files to scan (${SCAN_SCOPE} scope)."
else
  echo "-> Checking for local path leakage in ${#FILES_TO_SCAN[@]} file(s) (${SCAN_SCOPE} scope)..."
  FORBIDDEN_PATHS=("/Users/" "/home/sburdges" "/home/runner/work/KmiDi/KmiDi/KmiDi_FINAL")
  for path in "${FORBIDDEN_PATHS[@]}"; do
    if rg -q -F "$path" -- "${FILES_TO_SCAN[@]}"; then
      echo "FATAL: Hardcoded path '$path' found in scanned files."
      rg -n -F "$path" -- "${FILES_TO_SCAN[@]}"
      exit 1
    fi
  done
fi

# 2) Check for temporary/debug markers
echo "-> Checking for TODO/FIXME markers in sensitive areas..."
# This is a soft check, just to remind
if [[ "${#FILES_TO_SCAN[@]}" -gt 0 ]]; then
  rg -n "TODO\(critical\)" -- "${FILES_TO_SCAN[@]}" || true
fi

# 3) Check for prohibited keywords in music_brain (example)
echo "-> Checking music_brain for prohibited legacy patterns..."
if [ -d "music_brain" ]; then
  if rg -q "therapy_session" music_brain/; then
    echo "WARNING: legacy 'therapy_session' pattern found in music_brain."
  fi
fi

echo "=> Guardrails passed."
