#!/bin/bash
# Overwrite /Users/seanburdges/Desktop/KmiDi with latest from KmiDi-remote

REMOTE="/Users/seanburdges/Desktop/KmiDi-remote"
LOCAL="/Users/seanburdges/Desktop/KmiDi"

echo "🔄 Syncing repositories..."
echo "Source: $REMOTE"
echo "Target: $LOCAL (will be overwritten)"
echo ""

# Update from GitHub first
cd "$REMOTE" || exit 1
echo "📥 Pulling latest from GitHub..."
git pull origin main

# Overwrite local with remote
echo ""
echo "📋 Overwriting local KmiDi..."
rsync -av --delete --exclude='.git/' --exclude='.DS_Store' "$REMOTE/" "$LOCAL/"

echo ""
echo "✅ Sync complete!"
