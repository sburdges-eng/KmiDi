#!/bin/bash
# Startup script for Music Brain API
# Run this in a separate terminal window to start the API server

cd "$(dirname "$0")"

echo "Starting Music Brain API..."
echo "The API will be available at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Try python3 first, then python
if command -v python3 &> /dev/null; then
    python3 -m music_brain.api
elif command -v python &> /dev/null; then
    python -m music_brain.api
else
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi
