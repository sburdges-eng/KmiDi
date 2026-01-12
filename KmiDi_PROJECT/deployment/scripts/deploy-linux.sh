#!/bin/bash

# deploy-linux.sh - Deployment script for Linux
# =============================================
# This script automates the deployment of the KmiDi Music Generation API
# and its dependencies on a Linux system.

# --- Configuration ---
# Set these variables according to your environment
INSTALL_DIR="/opt/kmidi"
PYTHON_VERSION="3.11"
API_PORT="8000"
API_HOST="0.0.0.0"

# --- Prerequisites ---
echo "Checking prerequisites..."

# Check for Python
if ! command -v python${PYTHON_VERSION} &> /dev/null
then
    echo "Python ${PYTHON_VERSION} not found. Please install it using your distribution's package manager (e.g., sudo apt install python${PYTHON_VERSION})."
    exit 1
fi

# Check for pip
if ! command -v pip${PYTHON_VERSION} &> /dev/null
then
    echo "pip for Python ${PYTHON_VERSION} not found. Please install it (e.g., sudo apt install python3-pip)."
    exit 1
fi

# Check for Rust (for Tauri)
if ! command -v cargo &> /dev/null
then
    echo "Rust toolchain not found. Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    echo "Please restart your terminal or source your .bashrc/.zshrc and run this script again."
    exit 1
fi

# Check for libsndfile1 (runtime dependency for audio processing)
if ! dpkg -s libsndfile1 &> /dev/null && ! rpm -q libsndfile1 &> /dev/null; then
    echo "libsndfile1 not found. Please install it using your distribution's package manager (e.g., sudo apt install libsndfile1)."
    exit 1
fi

# --- Installation ---
echo "Installing KmiDi dependencies..."

# Navigate to project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project root."; exit 1; }

# Install Python dependencies
pip${PYTHON_VERSION} install -r KMiDi_PROJECT/requirements-production.txt || { echo "Failed to install Python dependencies."; exit 1; }

# Create installation directory
sudo mkdir -p "$INSTALL_DIR/api"
sudo mkdir -p "$INSTALL_DIR/music_brain"
sudo mkdir -p "$INSTALL_DIR/data"
sudo mkdir -p "$INSTALL_DIR/models"
sudo chown -R "$USER":"$USER" "$INSTALL_DIR" # Give ownership to current user

# Copy API and music_brain code
cp -R KMiDi_PROJECT/api/* "$INSTALL_DIR/api/"
cp -R KMiDi_PROJECT/music_brain/* "$INSTALL_DIR/music_brain/"
cp KMiDi_PROJECT/pyproject.toml "$INSTALL_DIR/"

# Optional: Copy data and models if they exist locally
if [ -d "KMiDi_PROJECT/data" ]; then
    cp -R KMiDi_PROJECT/data/* "$INSTALL_DIR/data/"
fi
if [ -d "KMiDi_PROJECT/models" ]; then
    cp -R KMiDi_PROJECT/models/* "$INSTALL_DIR/models/"
fi

# Build Tauri application (desktop frontend)
echo "Building Tauri desktop application..."
cd KMiDi_PROJECT/source/frontend/src-tauri || { echo "Failed to navigate to Tauri frontend."; exit 1; }
npm install # Ensure npm dependencies are installed for Tauri build
cargo tauri build --release || { echo "Failed to build Tauri app."; exit 1; }

# Copy the built Tauri app to installation directory
TAURI_APP_PATH="$(find "$PROJECT_ROOT/KMiDi_PROJECT/source/frontend/src-tauri/target/release/bundle/" -name "kmidi" -type f | head -n 1)" # Executable name is 'kmidi'
if [ -f "$TAURI_APP_PATH" ]; then
    echo "Copying Tauri app from $TAURI_APP_PATH to $INSTALL_DIR/kmidi"
    cp "$TAURI_APP_PATH" "$INSTALL_DIR/kmidi"
    chmod +x "$INSTALL_DIR/kmidi"
else
    echo "Warning: Tauri desktop application executable not found at expected path. Skipping copy."
fi

# --- Service Setup ---
echo "Setting up API service (manual start for now)..."

# Create a basic service script to start the API
cat << EOF | sudo tee "/etc/systemd/system/kmidi-api.service"
[Unit]
Description=KmiDi Music Generation API Service
After=network.target

[Service]
User=$USER
Group=$USER
WorkingDirectory=$INSTALL_DIR/api
Environment="PYTHONPATH=$INSTALL_DIR"
ExecStart=/usr/bin/python${PYTHON_VERSION} -m uvicorn main:app --host $API_HOST --port $API_PORT --workers 1
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable kmidi-api
sudo systemctl start kmidi-api

echo "API service configured and started. Check status with: sudo systemctl status kmidi-api"
echo "To launch the desktop app, run: $INSTALL_DIR/kmidi"

# --- Verification ---
echo "Verification steps:"
echo "1. Check API service status: sudo systemctl status kmidi-api"
echo "2. Test health endpoint: curl http://$API_HOST:$API_PORT/health"
echo "3. Launch the KmiDi desktop app and verify functionality: $INSTALL_DIR/kmidi"

echo "KmiDi deployment script finished for Linux."
