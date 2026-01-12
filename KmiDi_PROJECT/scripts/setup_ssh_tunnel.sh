#!/bin/bash
# =============================================================================
# SSH Tunnel for Codespace → Local Mac Dataset Access
# =============================================================================
# This creates a reverse SSH tunnel so Codespace can access your Mac's datasets
# over SSH without exposing your Mac to the internet.
#
# RUN THIS ON YOUR LOCAL MAC (not in Codespace)
#
# How it works:
#   1. Mac connects TO GitHub Codespace via SSH (outbound connection)
#   2. Creates reverse tunnel: Codespace:2222 → Mac:22
#   3. Codespace can then SSH back through the tunnel to access /Volumes/sbdrive
#
# Usage:
#   ./scripts/setup_ssh_tunnel.sh <codespace-ssh-url>
#
# Example:
#   ./scripts/setup_ssh_tunnel.sh "ssh -p 2222 codespace@xxx.github.dev"
# =============================================================================

set -e

CODESPACE_HOST="${1:-}"
LOCAL_DATA="/Volumes/sbdrive/audio/datasets"
TUNNEL_PORT=2222

if [ -z "$CODESPACE_HOST" ]; then
    echo "Usage: $0 <codespace-ssh-command>"
    echo ""
    echo "Get your Codespace SSH command from:"
    echo "  1. Open Codespace in browser"
    echo "  2. Click ... menu → 'Open in VS Code Desktop'"
    echo "  3. Or use: gh codespace ssh"
    echo ""
    echo "Example:"
    echo "  $0 'gh codespace ssh -c <codespace-name>'"
    exit 1
fi

echo "=============================================="
echo "  SSH Tunnel: Local Mac → Codespace"
echo "=============================================="
echo ""

# Check if SSH server is running on Mac
if ! pgrep -x "sshd" > /dev/null; then
    echo "Enabling Remote Login (SSH) on Mac..."
    echo "You may need to enter your password:"
    sudo systemsetup -setremotelogin on
fi

# Get local Mac username and hostname
MAC_USER=$(whoami)
MAC_HOST=$(hostname)

echo "Local Mac: $MAC_USER@$MAC_HOST"
echo "Data path: $LOCAL_DATA"
echo ""

# Create the tunnel
echo "Creating reverse SSH tunnel..."
echo "Codespace will be able to access your Mac via: ssh -p $TUNNEL_PORT localhost"
echo ""

# SSH to Codespace with reverse tunnel
# -R creates a reverse tunnel: remote:port → local:port
$CODESPACE_HOST -R $TUNNEL_PORT:localhost:22 -N &
TUNNEL_PID=$!

echo "Tunnel established (PID: $TUNNEL_PID)"
echo ""
echo "In Codespace, run:"
echo "  ssh -p $TUNNEL_PORT $MAC_USER@localhost"
echo ""
echo "To mount datasets in Codespace:"
echo "  sshfs -p $TUNNEL_PORT $MAC_USER@localhost:$LOCAL_DATA /data/datasets"
echo ""
echo "Press Ctrl+C to close tunnel"

# Wait for tunnel to be killed
wait $TUNNEL_PID
