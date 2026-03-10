#!/usr/bin/env bash
# Delete remote branches that are already merged into origin/main.
# Run from repo root. Requires push access to origin.
#
# Usage:
#   ./scripts/cleanup-merged-remote-branches.sh          # dry-run (default)
#   ./scripts/cleanup-merged-remote-branches.sh --execute # delete on origin

set -e
cd "$(git rev-parse --show-toplevel)"
git fetch origin --prune -q

MERGED=$(git branch -r --merged origin/main | sed 's/^[[:space:]]*//' | grep -E '^origin/' | grep -v 'origin/HEAD' | grep -v 'origin/main')
if [ -z "$MERGED" ]; then
  echo "No merged remote branches to delete."
  exit 0
fi

COUNT=$(echo "$MERGED" | wc -l | tr -d ' ')
echo "Branches merged into origin/main ($COUNT):"
echo "$MERGED" | sed 's|^origin/|  |'
echo ""

if [ "${1:-}" = "--execute" ]; then
  echo "Deleting on origin..."
  echo "$MERGED" | while read -r ref; do
    branch="${ref#origin/}"
    git push origin --delete "$branch" || true
  done
  echo "Done. Run 'git fetch origin --prune' elsewhere to refresh."
else
  echo "Dry-run. To delete these on origin, run:"
  echo "  ./scripts/cleanup-merged-remote-branches.sh --execute"
fi
