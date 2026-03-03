#!/usr/bin/env bash
# Static Listening Guardrails
# Ensures no forbidden patterns or accidental hardcoded paths remain in the codebase.
set -euo pipefail

echo "=> Running Static Listening Guardrails..."

# 0) Verify ripgrep is available
if ! command -v rg >/dev/null 2>&1; then
  echo "FATAL: ripgrep (rg) is not installed." >&2
  echo "       Install it from https://github.com/BurntSushi/ripgrep or via your package manager (e.g. brew install ripgrep / apt-get install ripgrep)." >&2
  exit 1
fi

# 1) Check for hardcoded local paths that shouldn't be in CI
echo "-> Checking for local path leakage..."
FORBIDDEN_PATHS=("/Users/" "/home/sburdges" "/home/runner/work/KmiDi/KmiDi/KmiDi_FINAL")
for path in "${FORBIDDEN_PATHS[@]}"; do
  if rg -q "$path" -g "!.git/*" -g "!.agents/*" -g "!ci_listening_guardrails.sh"; then
    echo "FATAL: Hardcoded path '$path' found in codebase." >&2
    rg "$path" -g "!.git/*" -g "!.agents/*" -g "!ci_listening_guardrails.sh"
    exit 1
  fi
done

# 2) Check for temporary/debug markers
echo "-> Checking for TODO/FIXME markers in sensitive areas..."
# This is a soft check, just to remind
rg "TODO\(critical\)" -g "!.git/*" || true

# 3) Check for prohibited keywords in music_brain (example)
echo "-> Checking music_brain for prohibited legacy patterns..."
if [ -d "music_brain" ]; then
  if rg -q "therapy_session" music_brain/; then
    echo "WARNING: legacy 'therapy_session' pattern found in music_brain."
  fi
fi

echo "=> Guardrails passed."
