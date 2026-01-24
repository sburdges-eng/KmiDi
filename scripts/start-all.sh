#!/bin/bash
# Start All Services Script
# Starts all required services for development: Music Brain API, Penta Core Server, Frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# PID file directory
PID_DIR="$PROJECT_ROOT/.pids"
mkdir -p "$PID_DIR"

# Function to cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down services..."
    if [ -f "$PID_DIR/music_brain_api.pid" ]; then
        kill $(cat "$PID_DIR/music_brain_api.pid") 2>/dev/null || true
        rm "$PID_DIR/music_brain_api.pid"
    fi
    if [ -f "$PID_DIR/penta_core.pid" ]; then
        kill $(cat "$PID_DIR/penta_core.pid") 2>/dev/null || true
        rm "$PID_DIR/penta_core.pid"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Music Brain API
print_status "Starting Music Brain API on port 8000..."
cd "$PROJECT_ROOT"
python3 -m music_brain.api > "$PROJECT_ROOT/logs/music_brain_api.log" 2>&1 &
MUSIC_BRAIN_PID=$!
echo $MUSIC_BRAIN_PID > "$PID_DIR/music_brain_api.pid"
print_status "Music Brain API started (PID: $MUSIC_BRAIN_PID)"

# Wait for API to be ready
sleep 2
if curl -s http://127.0.0.1:8000/emotions > /dev/null 2>&1; then
    print_status "Music Brain API is ready"
else
    print_warning "Music Brain API may not be ready yet"
fi

# Start Penta Core Server (if available)
if [ -f "$PROJECT_ROOT/penta_core/server.py" ]; then
    print_status "Starting Penta Core Server..."
    cd "$PROJECT_ROOT"
    python3 -m penta_core.server > "$PROJECT_ROOT/logs/penta_core.log" 2>&1 &
    PENTA_CORE_PID=$!
    echo $PENTA_CORE_PID > "$PID_DIR/penta_core.pid"
    print_status "Penta Core Server started (PID: $PENTA_CORE_PID)"
else
    print_warning "Penta Core Server not found, skipping"
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

echo ""
echo "=========================================="
echo "All Services Started"
echo "=========================================="
echo ""
echo "Services running:"
echo "  ✓ Music Brain API: http://127.0.0.1:8000"
if [ -f "$PID_DIR/penta_core.pid" ]; then
    echo "  ✓ Penta Core Server: (check logs/penta_core.log for port)"
fi
echo ""
echo "Logs:"
echo "  - Music Brain API: logs/music_brain_api.log"
if [ -f "$PID_DIR/penta_core.pid" ]; then
    echo "  - Penta Core: logs/penta_core.log"
fi
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
wait
