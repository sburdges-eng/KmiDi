#!/bin/bash
# =============================================================================
# SSH Server for Multiple Codespaces
# =============================================================================
# Run this on your LOCAL MAC to allow multiple Codespaces to connect and
# access datasets on /Volumes/sbdrive.
#
# Supports:
#   - Multiple simultaneous Codespace connections
#   - Direct access (if Mac has public IP or VPN)
#   - Tailscale/ZeroTier mesh network (recommended)
#   - ngrok tunnel (for NAT traversal)
#
# Usage:
#   ./scripts/ssh_server_for_codespaces.sh          # Start with Tailscale
#   ./scripts/ssh_server_for_codespaces.sh --ngrok  # Start with ngrok tunnel
#   ./scripts/ssh_server_for_codespaces.sh --local  # Local network only
# =============================================================================

set -e

LOCAL_DATA="/Volumes/sbdrive/audio/datasets"
SSH_PORT=22

echo "=============================================="
echo "  SSH Server for Codespace Dataset Access"
echo "=============================================="
echo ""

# Check data directory exists
if [ ! -d "$LOCAL_DATA" ]; then
    echo "ERROR: $LOCAL_DATA not found"
    echo "Is /Volumes/sbdrive mounted?"
    exit 1
fi

# Enable SSH on Mac if not already
echo "Checking SSH server..."
if ! sudo systemsetup -getremotelogin 2>/dev/null | grep -q "On"; then
    echo "Enabling Remote Login (SSH)..."
    sudo systemsetup -setremotelogin on
fi
echo "SSH server is running on port $SSH_PORT"
echo ""

# Get connection info based on mode
MODE="${1:---tailscale}"

case "$MODE" in
    --tailscale)
        echo "Mode: Tailscale (recommended)"
        echo ""
        if ! command -v tailscale &> /dev/null; then
            echo "Tailscale not installed. Install from: https://tailscale.com/download/mac"
            echo "Or use: brew install tailscale"
            exit 1
        fi

        TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
        if [ -z "$TAILSCALE_IP" ]; then
            echo "Tailscale not connected. Run: tailscale up"
            exit 1
        fi

        echo "Tailscale IP: $TAILSCALE_IP"
        echo ""
        echo "Add these Codespace secrets:"
        echo "  MAC_SSH_HOST=$TAILSCALE_IP"
        echo "  MAC_SSH_USER=$(whoami)"
        echo ""
        echo "Your Mac is accessible to any device on your Tailscale network."
        echo "Multiple Codespaces can connect simultaneously."
        ;;

    --ngrok)
        echo "Mode: ngrok tunnel"
        echo ""
        if ! command -v ngrok &> /dev/null; then
            echo "Installing ngrok..."
            brew install ngrok/ngrok/ngrok
        fi

        echo "Starting ngrok tunnel..."
        echo "Note: Free tier allows 1 tunnel. Get auth token at https://ngrok.com"
        echo ""

        # Start ngrok in background and capture URL
        ngrok tcp $SSH_PORT --log=stdout 2>&1 | while read line; do
            if echo "$line" | grep -q "url=tcp://"; then
                URL=$(echo "$line" | grep -oE 'tcp://[^[:space:]]+')
                HOST=$(echo "$URL" | sed 's|tcp://||' | cut -d: -f1)
                PORT=$(echo "$URL" | sed 's|tcp://||' | cut -d: -f2)
                echo ""
                echo "ngrok tunnel active!"
                echo ""
                echo "Add these Codespace secrets:"
                echo "  MAC_SSH_HOST=$HOST"
                echo "  MAC_SSH_PORT=$PORT"
                echo "  MAC_SSH_USER=$(whoami)"
                echo ""
            fi
            echo "$line"
        done
        ;;

    --local)
        echo "Mode: Local network only"
        echo ""

        # Get local IP
        LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")

        echo "Local IP: $LOCAL_IP"
        echo ""
        echo "This only works if Codespace and Mac are on same network."
        echo "(Usually not the case - use Tailscale or ngrok instead)"
        echo ""
        echo "Add these Codespace secrets:"
        echo "  MAC_SSH_HOST=$LOCAL_IP"
        echo "  MAC_SSH_USER=$(whoami)"
        ;;

    *)
        echo "Usage: $0 [--tailscale|--ngrok|--local]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "  Dataset Access Ready"
echo "=============================================="
echo ""
echo "Data path: $LOCAL_DATA"
echo "Contents:"
du -sh "$LOCAL_DATA"/*/ 2>/dev/null || echo "(checking...)"
echo ""
echo "In Codespace, datasets will be mounted at: /data/datasets"
echo ""
echo "Press Ctrl+C to stop (SSH server will keep running)"

# Keep script running for ngrok
if [ "$MODE" == "--ngrok" ]; then
    wait
else
    echo ""
    echo "SSH server running. Codespaces can connect."
    echo "To stop SSH: sudo systemsetup -setremotelogin off"
fi
