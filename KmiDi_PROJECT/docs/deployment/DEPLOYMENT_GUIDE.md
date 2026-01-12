# Deployment Guide for KmiDi Project

This guide provides comprehensive instructions for deploying the KmiDi Music Generation API and its associated services. It covers prerequisites, step-by-step deployment for various platforms, troubleshooting, and rollback procedures.

## Table of Contents
1.  [Prerequisites](#1-prerequisites)
2.  [Deployment Steps](#2-deployment-steps)
    *   [2.1 macOS Deployment](#21-macos-deployment)
    *   [2.2 Linux Deployment](#22-linux-deployment)
    *   [2.3 Windows Deployment](#23-windows-deployment)
3.  [Verification](#3-verification)
4.  [Troubleshooting](#4-troubleshooting)
5.  [Rollback Procedures](#5-rollback-procedures)

---

## 1. Prerequisites

Before deploying KmiDi, ensure you have the following installed on your target system:

*   **Python 3.9+**: Required for the FastAPI backend and Python-based modules.
    *   Verify with: `python3 --version`
*   **pip**: Python package installer.
    *   Verify with: `pip3 --version`
*   **Node.js 18+ and npm**: Required for building the Tauri desktop frontend.
    *   Verify with: `node --version`, `npm --version`
*   **Rust Toolchain with Cargo**: Required for building the Tauri desktop frontend.
    *   Verify with: `rustc --version`, `cargo --version`
    *   On macOS, ensure Xcode Command Line Tools are installed: `xcode-select --install`
*   **libsndfile1** (Linux only): A runtime dependency for audio processing.
    *   Install on Debian/Ubuntu: `sudo apt install libsndfile1`
    *   Install on Fedora/RHEL: `sudo dnf install libsndfile`
*   **curl**: For testing API endpoints.
*   **Git**: For cloning the repository.

---

## 2. Deployment Steps

These instructions assume you have cloned the KmiDi repository to your target machine.

```bash
# Navigate to the project root directory
cd /path/to/KmiDi-1/KMiDi_PROJECT
```

### 2.1 macOS Deployment

Use the `deploy-macos.sh` script to automate installation and setup.

```bash
cd deployment/scripts
chmod +x deploy-macos.sh
./deploy-macos.sh
```

**What the script does:**
*   Installs Python and Rust via Homebrew if not present.
*   Installs Python dependencies from `requirements-production.txt`.
*   Copies the API and `music_brain` code to `/usr/local/kmidi`.
*   Builds the Tauri desktop application and copies it to `/usr/local/kmidi/KMiDi.app`.
*   Creates a `start_api.sh` script for the FastAPI service.

#### 2.1.1 macOS (Apple Silicon) low-resource launch with Metal/MPS/MLX

For 16 GB machines or thermally constrained setups, use the baked-in resource-friendly defaults:

1. In `deployment/scripts/deploy-macos.sh`, set `INSTALL_MLX="true"` if you want Apple MLX (Metal-accelerated) installed. Otherwise the script will just install the standard requirements.
2. Run the deploy script as above; it generates `/usr/local/kmidi/start_api.sh` with conservative thread limits and MPS VRAM throttling.
3. Start the API with the low-resource settings:
   ```bash
   /usr/local/kmidi/start_api.sh
   ```
   The script sets:
   - `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=4`, `NUMEXPR_MAX_THREADS=4` to prevent CPU over-fan-out.
   - `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` to keep Metal (MPS) VRAM below 80% and avoid macOS swap.
   - `uvicorn --workers 1 --limit-concurrency 4` to reduce API load.
4. Keep other heavy apps closed during generation to avoid swap. If you still see swap/heat, lower `OMP_NUM_THREADS` to `2` before launching.

### 2.2 Linux Deployment

Use the `deploy-linux.sh` script to automate installation and setup.

```bash
cd deployment/scripts
chmod +x deploy-linux.sh
sudo ./deploy-linux.sh
```

**What the script does:**
*   Checks for Python, pip, Rust, and `libsndfile1`.
*   Installs Python dependencies from `requirements-production.txt`.
*   Copies the API and `music_brain` code to `/opt/kmidi`.
*   Builds the Tauri desktop application and copies the executable to `/opt/kmidi/kmidi`.
*   Sets up a `systemd` service for the FastAPI application, enabling it to start on boot.

### 2.3 Windows Deployment

Use the `deploy-windows.ps1` PowerShell script to automate installation and setup.

```powershell
# Open PowerShell as Administrator
cd path\to\KmiDi-1\KMiDi_PROJECT\deployment\scripts
.\deploy-windows.ps1
```

**What the script does:**
*   Checks for Python, pip, and Rust.
*   Installs Python dependencies from `requirements-production.txt`.
*   Copies the API and `music_brain` code to `C:\Program Files\KmiDi`.
*   Builds the Tauri desktop application (`KMiDi.exe`) and copies it to `C:\Program Files\KmiDi\KMiDi.exe`.
*   Creates a `start_api.bat` script for the FastAPI service.

---

## 3. Verification

After deployment, perform the following steps to verify the installation:

1.  **Start the FastAPI Service**: If not already started by a service manager (Linux), run the appropriate script:
    *   **macOS/Windows**: `[INSTALL_DIR]/start_api.sh` (macOS) or `[INSTALL_DIR]\start_api.bat` (Windows)
    *   **Linux**: `sudo systemctl status kmidi-api` (should be running)

2.  **Test API Health Endpoint**: Once the API is running, open a new terminal and run:
    ```bash
    curl http://127.0.0.1:8000/health
    ```
    Expected output should indicate `"status": "healthy"` and list service versions.

3.  **Launch Desktop Application**: Run the KMiDi desktop application:
    *   **macOS**: Double-click `[INSTALL_DIR]/KMiDi.app` in Finder, or run `open [INSTALL_DIR]/KMiDi.app`
    *   **Linux**: Run `[INSTALL_DIR]/kmidi`
    *   **Windows**: Double-click `[INSTALL_DIR]\KMiDi.exe`

4.  **Verify Frontend Functionality**: Interact with the desktop application to ensure music generation, emotion listing, and interrogation features work as expected.

---

## 4. Troubleshooting

*   **"API Offline" in Frontend / `curl` connection refused**: Ensure the FastAPI service is running and accessible on `http://127.0.0.1:8000` (or configured `API_HOST:API_PORT`). Check firewall rules if applicable.
    *   On Linux, check `sudo systemctl status kmidi-api` and `sudo journalctl -u kmidi-api.service` for logs.
*   **Python Module Not Found Errors**: Ensure Python dependencies are correctly installed. Re-run `pip install -r requirements-production.txt`.
*   **Rust/Tauri Build Failures**: Ensure the Rust toolchain is correctly installed (`rustup update`, `xcode-select --install` on macOS). Check specific compiler errors for details.
*   **Permissions Errors**: On Linux, ensure the KMiDi installation directory has correct permissions (`sudo chown -R $USER:$USER [INSTALL_DIR]`).
*   **Security Vulnerabilities**: If `pip-audit` or `safety check` report vulnerabilities, update the problematic packages in `requirements-production.txt` and re-install dependencies.

---

## 5. Rollback Procedures

To revert a deployment, follow these steps:

1.  **Stop Services**: Stop any running KMiDi services:
    *   **Linux**: `sudo systemctl stop kmidi-api`
    *   **macOS/Windows**: Terminate the `start_api.sh/.bat` process and close the desktop app.

2.  **Remove Installation Directory**: Delete the KMiDi installation directory:
    *   **macOS/Linux**: `sudo rm -rf [INSTALL_DIR]`
    *   **Windows**: `Remove-Item -Recurse -Force "$InstallDir"`

3.  **Remove System Services** (Linux only):
    *   `sudo systemctl disable kmidi-api`
    *   `sudo rm /etc/systemd/system/kmidi-api.service`
    *   `sudo systemctl daemon-reload`

4.  **Clean Build Artifacts** (if necessary, in project root):
    ```bash
    cd /path/to/KmiDi-1/KMiDi_PROJECT
    rm -rf build CMakeCache.txt CMakeFiles
    cd source/frontend/src-tauri
    npm run tauri clean
    cargo clean
    ```

This will remove all deployed files and services, effectively reverting the system to its state before deployment.
