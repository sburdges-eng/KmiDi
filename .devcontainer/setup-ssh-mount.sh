#!/bin/bash
# =============================================================================
# Mount Local Mac Datasets via SSH (Run in Codespace)
# =============================================================================
# Prerequisites:
#   1. Local Mac has SSH enabled (System Preferences → Sharing → Remote Login)
#   2. SSH tunnel is running (./scripts/setup_ssh_tunnel.sh on Mac)
#   3. Or use Tailscale/ZeroTier for direct connection
#
# Codespace Secrets:
#   MAC_SSH_HOST     - Your Mac's IP or hostname (or localhost:2222 with tunnel)
#   MAC_SSH_USER     - Your Mac username
#   MAC_SSH_KEY      - Private SSH key (base64 encoded)
#   MAC_DATA_PATH    - Path to datasets (default: /Volumes/sbdrive/audio/datasets)
# =============================================================================

set -e

MOUNT_POINT="/data/datasets"

echo "=== Setting up SSH Mount to Local Mac ==="

# Check for required secrets
if [ -z "$MAC_SSH_HOST" ] || [ -z "$MAC_SSH_USER" ]; then
    echo "ERROR: MAC_SSH_HOST and MAC_SSH_USER must be set as Codespace secrets"
    echo ""
    echo "To set up:"
    echo "1. Enable SSH on your Mac (System Preferences → Sharing → Remote Login)"
    echo "2. Get your Mac's IP: ifconfig | grep 'inet '"
    echo "3. Add secrets in GitHub repo Settings → Secrets → Codespaces:"
    echo "   - MAC_SSH_HOST: your-mac-ip or localhost (if using tunnel)"
    echo "   - MAC_SSH_USER: your Mac username"
    echo "   - MAC_SSH_KEY: base64-encoded private key (optional)"
    echo "   - MAC_DATA_PATH: /Volumes/sbdrive/audio/datasets"
    exit 1
fi

MAC_DATA="${MAC_DATA_PATH:-/Volumes/sbdrive/audio/datasets}"
SSH_PORT="${MAC_SSH_PORT:-22}"

# Install sshfs if not present
if ! command -v sshfs &> /dev/null; then
    echo "Installing sshfs..."
    sudo apt-get update && sudo apt-get install -y sshfs
fi

# Set up SSH key if provided
if [ -n "$MAC_SSH_KEY" ]; then
    echo "Setting up SSH key..."
    mkdir -p ~/.ssh
    echo "$MAC_SSH_KEY" | base64 -d > ~/.ssh/mac_key
    chmod 600 ~/.ssh/mac_key
    SSH_KEY_OPT="-o IdentityFile=~/.ssh/mac_key"
else
    SSH_KEY_OPT=""
fi

# Create mount point
sudo mkdir -p "$MOUNT_POINT"
sudo chown $(whoami) "$MOUNT_POINT"

# Test SSH connection
echo "Testing SSH connection to $MAC_SSH_USER@$MAC_SSH_HOST:$SSH_PORT..."
if ssh -p "$SSH_PORT" $SSH_KEY_OPT -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    "$MAC_SSH_USER@$MAC_SSH_HOST" "echo 'SSH OK'" 2>/dev/null; then
    echo "SSH connection successful!"
else
    echo "ERROR: Could not connect to Mac via SSH"
    echo ""
    echo "Troubleshooting:"
    echo "1. Is SSH enabled on Mac? (System Preferences → Sharing → Remote Login)"
    echo "2. Is the Mac on the same network or tunnel running?"
    echo "3. Check firewall settings"
    exit 1
fi

# Mount via sshfs
echo "Mounting $MAC_DATA → $MOUNT_POINT"
sshfs -p "$SSH_PORT" $SSH_KEY_OPT \
    -o StrictHostKeyChecking=no \
    -o reconnect \
    -o ServerAliveInterval=15 \
    -o allow_other \
    "$MAC_SSH_USER@$MAC_SSH_HOST:$MAC_DATA" "$MOUNT_POINT"

# Verify mount
if mountpoint -q "$MOUNT_POINT"; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Mac datasets mounted at: $MOUNT_POINT"
    echo ""
    ls -la "$MOUNT_POINT"
else
    echo "ERROR: Mount failed"
    exit 1
fi
