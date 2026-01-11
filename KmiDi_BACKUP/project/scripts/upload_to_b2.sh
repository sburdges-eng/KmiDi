#!/bin/bash
# =============================================================================
# Upload Datasets to Backblaze B2
# =============================================================================
# Run this on your LOCAL MAC to upload datasets from /Volumes/sbdrive to B2
#
# Usage:
#   ./scripts/upload_to_b2.sh              # Upload all datasets
#   ./scripts/upload_to_b2.sh m4singer     # Upload specific dataset
#   ./scripts/upload_to_b2.sh --status     # Check upload status
#
# Prerequisites:
#   1. rclone installed: brew install rclone
#   2. rclone configured: rclone config (add b2 remote)
#   3. B2 bucket created
# =============================================================================

set -e

LOCAL_DATA="/Volumes/sbdrive/audio/datasets"
B2_BUCKET="${B2_BUCKET_NAME:-kmidi-datasets}"
B2_REMOTE="b2:$B2_BUCKET"

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    brew install rclone
fi

# Check if b2 remote is configured
if ! rclone listremotes | grep -q "^b2:"; then
    echo "ERROR: rclone 'b2' remote not configured"
    echo ""
    echo "Run: rclone config"
    echo "  n) New remote"
    echo "  name: b2"
    echo "  Storage: b2"
    echo "  account: <your B2 account ID>"
    echo "  key: <your B2 application key>"
    exit 1
fi

show_status() {
    echo "=== B2 Bucket Contents ==="
    rclone lsd "$B2_REMOTE" 2>/dev/null || echo "(empty)"
    echo ""
    echo "=== B2 Usage ==="
    rclone size "$B2_REMOTE" 2>/dev/null || echo "0 bytes"
}

upload_dataset() {
    local name=$1
    local src="$LOCAL_DATA/$name"
    local dst="$B2_REMOTE/$name"

    if [ ! -d "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    echo "Uploading $name..."
    echo "  Source: $src"
    echo "  Dest:   $dst"
    echo ""

    # Use --transfers=4 for parallel uploads
    rclone sync "$src" "$dst" \
        --progress \
        --transfers=4 \
        --checkers=8 \
        --fast-list \
        --exclude="*.zip" \
        --exclude="*.tar.gz"

    echo "Done: $name"
}

main() {
    echo "=============================================="
    echo "  Upload Datasets to Backblaze B2"
    echo "  Local: $LOCAL_DATA"
    echo "  B2:    $B2_REMOTE"
    echo "=============================================="

    if [ "$1" == "--status" ] || [ "$1" == "-s" ]; then
        show_status
        exit 0
    fi

    # Check local data exists
    if [ ! -d "$LOCAL_DATA" ]; then
        echo "ERROR: $LOCAL_DATA not found"
        echo "Is /Volumes/sbdrive mounted?"
        exit 1
    fi

    echo ""
    echo "Local dataset sizes:"
    du -sh "$LOCAL_DATA"/*/ 2>/dev/null || echo "No datasets found"
    echo ""

    if [ -n "$1" ] && [ "$1" != "--all" ]; then
        # Upload specific dataset
        upload_dataset "$1"
    else
        # Upload all datasets
        echo "Uploading all datasets..."
        echo ""

        # Upload in order of size (smallest first for quick progress)
        for ds in moodylyrics wasabi genius_lyrics lakh_midi m4singer; do
            if [ -d "$LOCAL_DATA/$ds" ]; then
                upload_dataset "$ds"
                echo ""
            fi
        done
    fi

    echo ""
    echo "=== Upload Complete ==="
    show_status
}

main "$@"
