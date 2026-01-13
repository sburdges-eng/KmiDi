#!/bin/bash

# deploy-macos.sh - Deployment script for macOS
# =============================================
# This script automates the deployment of the KmiDi Music Generation API
# and its dependencies on a macOS system.

# --- Configuration ---
# Set these variables according to your environment
INSTALL_DIR="/usr/local/kmidi"
PYTHON_VERSION="3.12"
API_PORT="8000"
API_HOST="127.0.0.1"
# Set to "true" to also install Apple MLX for Metal-accelerated inference
INSTALL_MLX="false"

# --- Prerequisites ---
echo "Checking prerequisites..."

# Check for Homebrew
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo "Please run this script again after Homebrew installation completes."
    exit 1
fi

# Check for Python
if ! command -v python${PYTHON_VERSION} &> /dev/null
then
    echo "Python ${PYTHON_VERSION} not found. Installing Python..."
    brew install python@${PYTHON_VERSION}
    # Ensure python symlink points to the correct version if needed
    # brew link python@${PYTHON_VERSION}
fi

# Check for Rust (for Tauri)
if ! command -v cargo &> /dev/null
then
    echo "Rust toolchain not found. Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    echo "Please restart your terminal or source your .bashrc/.zshrc and run this script again."
    exit 1
fi

# --- Installation ---
echo "Installing KmiDi dependencies..."

# Navigate to project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project root."; exit 1; }

# Install Python dependencies (project root already set)
pip${PYTHON_VERSION} install -r requirements-production.txt || { echo "Failed to install Python dependencies."; exit 1; }

# Optional: Apple MLX (Metal-accelerated) for lightweight local inference
if [ "${INSTALL_MLX}" = "true" ]; then
    pip${PYTHON_VERSION} install mlx mlx-lm || { echo "Failed to install MLX packages."; exit 1; }
fi

# Create installation directory
mkdir -p "$INSTALL_DIR/api"
mkdir -p "$INSTALL_DIR/music_brain"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/models"

# Copy API and music_brain code (main sources live under source/python)
cp -R api/* "$INSTALL_DIR/api/"
cp -R source/python/music_brain/* "$INSTALL_DIR/music_brain/"
cp pyproject.toml "$INSTALL_DIR/"

# Optional: Copy data and models if they exist locally
if [ -d "data" ]; then
    cp -R data/* "$INSTALL_DIR/data/"
fi
if [ -d "models" ]; then
    cp -R models/* "$INSTALL_DIR/models/"
fi

# Build Tauri application (desktop frontend)
# This assumes `npm install` has been run in `source/frontend/src-tauri`
# and that `cargo` is in PATH
echo "Building Tauri desktop application..."
cd source/frontend/src-tauri || { echo "Failed to navigate to Tauri frontend."; exit 1; }
npm install # Ensure npm dependencies are installed for Tauri build
cargo tauri build --release || { echo "Failed to build Tauri app."; exit 1; }

# Copy the built Tauri app to installation directory
TAURI_APP_PATH="$(find "$PROJECT_ROOT/source/frontend/src-tauri/target/release/bundle/" -name "KMiDi.app" -type d | head -n 1)"
if [ -d "$TAURI_APP_PATH" ]; then
    echo "Copying Tauri app from $TAURI_APP_PATH to $INSTALL_DIR/KMiDi.app"
    cp -R "$TAURI_APP_PATH" "$INSTALL_DIR/"
else
    echo "Warning: Tauri desktop application not found at expected path. Skipping copy."
fi

# --- Service Setup ---
echo "Setting up API service (manual start for now)..."

# Create a basic service script to start the API (resource-friendly defaults)
cat << EOF > "$INSTALL_DIR/start_api.sh"
#!/bin/bash
cd "$INSTALL_DIR/api"
export PYTHONPATH="$INSTALL_DIR"

# Limit CPU thread fan-out to keep temps and RAM in check on Apple Silicon
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=\${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=\${VECLIB_MAXIMUM_THREADS:-4}
export NUMEXPR_MAX_THREADS=\${NUMEXPR_MAX_THREADS:-4}

# Keep Metal (MPS) VRAM usage under control to avoid macOS swap pressure
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=\${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.8}

# Run uvicorn in single-worker, low-concurrency mode to reduce load
exec python${PYTHON_VERSION} -m uvicorn main:app --host $API_HOST --port $API_PORT --workers 1 --limit-concurrency 4
EOF
chmod +x "$INSTALL_DIR/start_api.sh"

echo "To start the API, run: $INSTALL_DIR/start_api.sh"
echo "To launch the desktop app, run: $INSTALL_DIR/KMiDi.app/Contents/MacOS/KMiDi"

# --- Verification ---
echo "Verification steps:"
echo "1. Start the API using: $INSTALL_DIR/start_api.sh"
echo "2. Once API is running, test health endpoint: curl http://$API_HOST:$API_PORT/health"
echo "3. Launch the KmiDi desktop app and verify functionality."

echo "KmiDi deployment script finished for macOS."
