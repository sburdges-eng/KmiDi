#!/usr/bin/env bash
# Safe helper: stages all, shows diff, asks confirmation, then commits & pushes.
set -e

MSG="${1:-chore: update project files}"
BRANCH="${2:-main}"
REMOTE="${3:-origin}"

git status --short
echo ""
git --no-pager diff --staged || true

read -p "Stage ALL changes and continue? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 1
fi

git add -A
git commit -m "$MSG" || { echo "Nothing to commit."; exit 0; }
git push "$REMOTE" "$BRANCH"
echo "Pushed to $REMOTE/$BRANCH"
