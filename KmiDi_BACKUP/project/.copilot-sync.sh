#!/bin/bash
# Copilot-Claude Sync Script
# This script helps sync changes between Claude and GitHub Copilot

# Use the current directory as the default repo path if not specified
REPO_PATH="${1:-.}"

cd "$REPO_PATH" || exit 1

echo "📡 Syncing KmiDi repository..."
echo "================================"

# Fetch latest changes from remote
echo "🔄 Fetching from remote..."
git fetch origin

# Show current status
echo ""
echo "📊 Current status:"
git status

# Show any incoming changes
echo ""
echo "📥 Incoming changes:"
git log HEAD..origin/main --oneline --decorate

# Pull changes if any (only in interactive mode)
if [[ -t 0 ]]; then
    echo ""
    read -p "Pull latest changes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git pull origin main
        echo "✅ Repository updated!"
    else
        echo "⏸️  Skipped pull"
    fi
else
    echo "ℹ️  Non-interactive mode: skipping pull prompt."
fi

echo ""
echo "================================"
echo "✨ Sync complete!"
