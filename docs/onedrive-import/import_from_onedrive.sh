#!/usr/bin/env bash
# Copy key KmiDi-remote docs from OneDrive into this directory.
# Run from repo root or from docs/onedrive-import when OneDrive is synced
# (e.g. after opening the folder so files are "available offline").
set -euo pipefail

# Prefer env; then SSD copy; then OneDrive
if [[ -n "${ONEDRIVE_KMIDI_REMOTE:-}" ]]; then
  SRC="$ONEDRIVE_KMIDI_REMOTE"
elif [[ -d "/Volumes/Sean's SSD/KmiDi_MASTER_VAULT/KmiDi" ]] && [[ -f "/Volumes/Sean's SSD/KmiDi_MASTER_VAULT/KmiDi/KMIDI_CONSOLIDATION_SUMMARY.md" ]]; then
  SRC="/Volumes/Sean's SSD/KmiDi_MASTER_VAULT/KmiDi"
else
  SRC="$HOME/Library/CloudStorage/OneDrive-Personal/JUCE 2/Desktop/KmiDi-remote"
fi
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
DEST="${REPO_ROOT}/docs/onedrive-import"

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC"
  echo "Set ONEDRIVE_KMIDI_REMOTE if your path differs, or ensure OneDrive has synced."
  exit 1
fi

required=(KMIDI_CONSOLIDATION_SUMMARY.md IMPLEMENTATION_PLAN.md DESIGN_Integration_Architecture.md MERGER_INFRASTRUCTURE_COMPLETE.md)
for f in "${required[@]}"; do
  if [[ -f "$SRC/$f" ]]; then
    cp "$SRC/$f" "$DEST/$f"
    echo "  copied $f ($(wc -c < "$DEST/$f") bytes)"
  else
    echo "  skip $f (not found)"
  fi
done

echo "Done. Optional: run the for-loop in README.md to copy additional docs."
