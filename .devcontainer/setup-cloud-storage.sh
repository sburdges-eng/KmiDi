#!/bin/bash
# =============================================================================
# Cloud Storage Setup for Codespace - Backblaze B2 + rclone
# =============================================================================
# This script configures rclone to mount Backblaze B2 storage in Codespace
# for direct access to training datasets.
#
# Prerequisites:
#   1. Backblaze B2 account (free tier: 10GB storage, 1GB/day download)
#   2. B2 application key (set as Codespace secrets)
#
# Codespace Secrets Required:
#   B2_ACCOUNT_ID     - Your B2 account ID (or application key ID)
#   B2_APPLICATION_KEY - Your B2 application key
#   B2_BUCKET_NAME    - Your bucket name (e.g., kmidi-datasets)
# =============================================================================

set -e

MOUNT_POINT="/data/datasets"
CACHE_DIR="/tmp/rclone-cache"

echo "=== Setting up Backblaze B2 Cloud Storage ==="

# Check for required secrets
if [ -z "$B2_ACCOUNT_ID" ] || [ -z "$B2_APPLICATION_KEY" ]; then
    echo "ERROR: B2_ACCOUNT_ID and B2_APPLICATION_KEY must be set as Codespace secrets"
    echo ""
    echo "To set up:"
    echo "1. Go to https://www.backblaze.com/b2/cloud-storage.html"
    echo "2. Create account and bucket"
    echo "3. Generate application key"
    echo "4. Add secrets in GitHub repo Settings > Secrets > Codespaces"
    exit 1
fi

B2_BUCKET="${B2_BUCKET_NAME:-kmidi-datasets}"

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Configure rclone for B2
echo "Configuring rclone for Backblaze B2..."
mkdir -p ~/.config/rclone

cat > ~/.config/rclone/rclone.conf << EOF
[b2]
type = b2
account = ${B2_ACCOUNT_ID}
key = ${B2_APPLICATION_KEY}
hard_delete = true
EOF

# Test connection
echo "Testing B2 connection..."
if rclone lsd b2: 2>/dev/null; then
    echo "B2 connection successful!"
else
    echo "ERROR: Could not connect to B2. Check your credentials."
    exit 1
fi

# Create mount point and cache directory
sudo mkdir -p "$MOUNT_POINT"
sudo chown $(whoami) "$MOUNT_POINT"
mkdir -p "$CACHE_DIR"

# Mount B2 bucket (background process)
echo "Mounting B2 bucket: $B2_BUCKET -> $MOUNT_POINT"
rclone mount "b2:$B2_BUCKET" "$MOUNT_POINT" \
    --vfs-cache-mode full \
    --vfs-cache-max-size 10G \
    --cache-dir "$CACHE_DIR" \
    --allow-non-empty \
    --daemon

# Wait for mount
sleep 2

if mountpoint -q "$MOUNT_POINT"; then
    echo ""
    echo "=== SUCCESS ==="
    echo "B2 bucket mounted at: $MOUNT_POINT"
    echo ""
    echo "Dataset paths:"
    echo "  $MOUNT_POINT/m4singer/"
    echo "  $MOUNT_POINT/lakh_midi/"
    echo "  $MOUNT_POINT/genius_lyrics/"
    echo "  $MOUNT_POINT/moodylyrics/"
    echo ""
    ls -la "$MOUNT_POINT" 2>/dev/null || echo "(empty - upload datasets first)"
else
    echo "ERROR: Mount failed"
    exit 1
fi
