# deploy-windows.ps1 - Deployment script for Windows (PowerShell)
# ==============================================================
# This script automates the deployment of the KmiDi Music Generation API
# and its dependencies on a Windows system.

# --- Configuration ---
# Set these variables according to your environment
$InstallDir = "C:\Program Files\KmiDi"
$PythonVersion = "3.11"
$ApiPort = "8000"
$ApiHost = "127.0.0.1"

# --- Prerequisites ---
Write-Host "Checking prerequisites..."

# Check for Python
$pythonPath = "C:\Python$PythonVersion\python.exe"
if (-not (Test-Path $pythonPath))
{
    Write-Host "Python $PythonVersion not found at $pythonPath. Please install it from python.org."
    exit 1
}

# Check for pip
$pipPath = "C:\Python$PythonVersion\Scripts\pip.exe"
if (-not (Test-Path $pipPath))
{
    Write-Host "pip for Python $PythonVersion not found. Please ensure pip is installed for your Python installation."
    exit 1
}

# Check for Rust (for Tauri)
if (-not (Get-Command cargo -ErrorAction SilentlyContinue))
{
    Write-Host "Rust toolchain not found. Installing rustup..."
    Invoke-WebRequest -Uri "https://win.rustup.rs" -OutFile "rustup-init.exe"
    .\rustup-init.exe -y --default-toolchain stable
    Remove-Item "rustup-init.exe"
    Write-Host "Please restart PowerShell or your system and run this script again."
    exit 1
}

# --- Installation ---
Write-Host "Installing KmiDi dependencies..."

# Navigate to project root
$ScriptDir = (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$ProjectRoot = (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

# Install Python dependencies
& "$pipPath" install -r "$ProjectRoot\KMiDi_PROJECT\requirements-production.txt" || {
    Write-Host "Failed to install Python dependencies." -ForegroundColor Red; exit 1;
}

# Create installation directory
New-Item -ItemType Directory -Force -Path "$InstallDir\api"
New-Item -ItemType Directory -Force -Path "$InstallDir\music_brain"
New-Item -ItemType Directory -Force -Path "$InstallDir\data"
New-Item -ItemType Directory -Force -Path "$InstallDir\models"

# Copy API and music_brain code
Copy-Item -Recurse -Force "$ProjectRoot\KMiDi_PROJECT\api\*" "$InstallDir\api\"
Copy-Item -Recurse -Force "$ProjectRoot\KMiDi_PROJECT\music_brain\*" "$InstallDir\music_brain\"
Copy-Item -Force "$ProjectRoot\KMiDi_PROJECT\pyproject.toml" "$InstallDir\"

# Optional: Copy data and models if they exist locally
if (Test-Path "$ProjectRoot\KMiDi_PROJECT\data") {
    Copy-Item -Recurse -Force "$ProjectRoot\KMiDi_PROJECT\data\*" "$InstallDir\data\"
}
if (Test-Path "$ProjectRoot\KMiDi_PROJECT\models") {
    Copy-Item -Recurse -Force "$ProjectRoot\KMiDi_PROJECT\models\*" "$InstallDir\models\"
}

# Build Tauri application (desktop frontend)
Write-Host "Building Tauri desktop application..."
Set-Location "$ProjectRoot\KMiDi_PROJECT\source\frontend\src-tauri"

npm install # Ensure npm dependencies are installed for Tauri build
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to install npm dependencies." -ForegroundColor Red; exit 1; }

& cargo tauri build --release || {
    Write-Host "Failed to build Tauri app." -ForegroundColor Red; exit 1;
}

# Copy the built Tauri app to installation directory
$TauriAppPath = Get-ChildItem -Path "$ProjectRoot\KMiDi_PROJECT\source\frontend\src-tauri\target\release\bundle\" -Filter "KMiDi*.exe" -Recurse | Select-Object -First 1 | Select-Object -ExpandProperty FullName

if ($TauriAppPath -ne $null) {
    Write-Host "Copying Tauri app from $TauriAppPath to $InstallDir\KMiDi.exe"
    Copy-Item -Force "$TauriAppPath" "$InstallDir\KMiDi.exe"
} else {
    Write-Host "Warning: Tauri desktop application executable not found at expected path. Skipping copy." -ForegroundColor Yellow
}

# --- Service Setup ---
Write-Host "Setting up API service..."

# Create a basic service script to start the API
$apiServiceScriptContent = @"
@echo off
cd "$InstallDir\api"
set PYTHONPATH="$InstallDir"
"C:\Python$PythonVersion\python.exe" -m uvicorn main:app --host $ApiHost --port $ApiPort --workers 1
"@

$apiServiceScriptPath = Join-Path $InstallDir "start_api.bat"
$apiServiceScriptContent | Set-Content -Path $apiServiceScriptPath

Write-Host "To start the API, run: $apiServiceScriptPath"
Write-Host "To launch the desktop app, run: $InstallDir\KMiDi.exe"

# --- Verification ---
Write-Host "Verification steps:"
Write-Host "1. Start the API using: $apiServiceScriptPath"
Write-Host "2. Once API is running, test health endpoint: curl http://$ApiHost:$ApiPort/health (Requires curl to be installed)"
Write-Host "3. Launch the KMiDi desktop app and verify functionality: $InstallDir\KMiDi.exe"

Write-Host "KMiDi deployment script finished for Windows."
