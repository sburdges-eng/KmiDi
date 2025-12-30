#!/usr/bin/env bash
# Staging-friendly sync helper for /workspaces/KmiDi
# Prompts before pulling or pushing; avoids destructive actions.

set -euo pipefail
REPO="/workspaces/KmiDi"
BRANCH="main"

cd "$REPO" || { echo "Repo not found: $REPO"; exit 1; }

echo "📡 Syncing KmiDi (branch: $BRANCH)"
echo "================================"

echo "🔄 Fetching from origin..."
git fetch origin

echo ""
echo "📊 Current status:"
git status --short --branch

echo ""
echo "📥 Incoming changes (origin/$BRANCH):"
git log HEAD..origin/$BRANCH --oneline --decorate || true

echo ""
read -p "Pull latest from origin/$BRANCH? (y/n): " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  git pull origin "$BRANCH"
else
  echo "⏸️  Skipped pull"
fi

if [[ -n $(git status --porcelain) ]]; then
  echo ""
  read -p "Add and commit local changes now? (y/n): " -n 1 -r; echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Commit message: " MSG
    git add -A
    git commit -m "$MSG" || true
    read -p "Push to origin/$BRANCH? (y/n): " -n 1 -r; echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      git push origin "$BRANCH"
    else
      echo "⏸️  Skipped push"
    fi
  else
    echo "⏸️  Skipped commit/push"
  fi
else
  echo "✅ No local changes to commit"
fi

echo ""
echo "================================"
echo "✨ Sync complete!"