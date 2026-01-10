# KmiDi Deployment Script - Windows PowerShell
# ============================================
# Automated deployment script for Windows environments

# Set error handling
$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$DeployDir = if ($env:DEPLOY_DIR) { $env:DEPLOY_DIR } else { "C:\KmiDi" }
$VenvDir = "$DeployDir\venv"
$DataDir = if ($env:DATA_DIR) { $env:DATA_DIR } else { "C:\KmiDi\data" }
$LogDir = "$DeployDir\logs"

# Logging functions
function Log-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Log-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Log-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Log-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Test-Prerequisites {
    Log-Info "Checking prerequisites..."
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Log-Success "Python found: $pythonVersion"
    }
    catch {
        Log-Error "Python is not installed or not in PATH"
        Log-Info "Download Python from: https://www.python.org/downloads/"
        exit 1
    }
    
    # Check pip
    try {
        $pipVersion = pip --version 2>&1
        Log-Success "pip found: $pipVersion"
    }
    catch {
        Log-Error "pip is not installed or not in PATH"
        exit 1
    }
    
    Log-Success "Prerequisites check passed"
}

# Create directory structure
function New-DirectoryStructure {
    Log-Info "Creating directory structure..."
    
    $directories = @(
        $DeployDir,
        $DataDir,
        $LogDir,
        "$DeployDir\config"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Log-Success "Created: $dir"
        }
        else {
            Log-Info "Already exists: $dir"
        }
    }
    
    Log-Success "Directory structure created"
}

# Create virtual environment
function New-VirtualEnvironment {
    Log-Info "Setting up Python virtual environment..."
    
    if (Test-Path $VenvDir) {
        Log-Warning "Virtual environment already exists, recreating..."
        Remove-Item -Path $VenvDir -Recurse -Force
    }
    
    python -m venv $VenvDir
    
    if (-not $?) {
        Log-Error "Failed to create virtual environment"
        exit 1
    }
    
    # Activate virtual environment
    & "$VenvDir\Scripts\Activate.ps1"
    
    # Upgrade pip
    python -m pip install --upgrade pip setuptools wheel
    
    Log-Success "Virtual environment created"
}

# Install dependencies
function Install-Dependencies {
    Log-Info "Installing Python dependencies..."
    
    # Activate virtual environment
    & "$VenvDir\Scripts\Activate.ps1"
    
    # Install production requirements
    $requirementsFile = "$ProjectRoot\requirements-production.txt"
    if (Test-Path $requirementsFile) {
        pip install -r $requirementsFile
    }
    else {
        Log-Warning "requirements-production.txt not found, using api\requirements.txt"
        $apiRequirements = "$ProjectRoot\api\requirements.txt"
        if (Test-Path $apiRequirements) {
            pip install -r $apiRequirements
        }
        else {
            Log-Error "No requirements file found"
            exit 1
        }
    }
    
    # Install project in development mode
    pip install -e $ProjectRoot
    
    Log-Success "Dependencies installed"
}

# Copy application files
function Copy-ApplicationFiles {
    Log-Info "Copying application files..."
    
    # Copy application code
    Copy-Item -Path "$ProjectRoot\music_brain" -Destination "$DeployDir\music_brain" -Recurse -Force
    Copy-Item -Path "$ProjectRoot\api" -Destination "$DeployDir\api" -Recurse -Force
    Copy-Item -Path "$ProjectRoot\pyproject.toml" -Destination "$DeployDir\pyproject.toml" -Force
    
    # Copy configuration files
    $envExample = "$ProjectRoot\.env.example"
    if (Test-Path $envExample) {
        $envFile = "$DeployDir\config\.env"
        if (-not (Test-Path $envFile)) {
            Copy-Item -Path $envExample -Destination $envFile
            Log-Warning "Please edit $envFile with your configuration"
        }
    }
    
    Log-Success "Application files copied"
}

# Create Windows Service (NSSM - Non-Sucking Service Manager)
function New-WindowsService {
    Log-Info "Creating Windows service configuration..."
    
    $nssmPath = "$DeployDir\nssm.exe"
    
    # Check if NSSM is available
    if (-not (Test-Path $nssmPath)) {
        Log-Warning "NSSM (Non-Sucking Service Manager) not found"
        Log-Info "To install as Windows service, download NSSM from: https://nssm.cc/download"
        Log-Info "Or run manually with:"
        Log-Info "  cd $DeployDir"
        Log-Info "  $VenvDir\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
        return
    }
    
    # Install service
    & $nssmPath install "KmiDiAPI" "$VenvDir\Scripts\python.exe" "-m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4"
    & $nssmPath set "KmiDiAPI" AppDirectory "$DeployDir"
    & $nssmPath set "KmiDiAPI" AppStdout "$LogDir\api.out.log"
    & $nssmPath set "KmiDiAPI" AppStderr "$LogDir\api.err.log"
    & $nssmPath set "KmiDiAPI" AppEnvironmentExtra "PYTHONPATH=$DeployDir" "KELLY_AUDIO_DATA_ROOT=$DataDir" "LOG_LEVEL=INFO"
    
    # Start service
    & $nssmPath start "KmiDiAPI"
    
    Log-Success "Windows service created and started"
}

# Health check
function Test-Health {
    Log-Info "Performing health check..."
    
    Start-Sleep -Seconds 3
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Log-Success "Health check passed"
        }
        else {
            Log-Warning "Health check returned status code: $($response.StatusCode)"
        }
    }
    catch {
        Log-Warning "Health check failed. Service may still be starting."
        Log-Info "Check logs: Get-Content $LogDir\api.err.log -Tail 20"
    }
}

# Main deployment function
function Main {
    Log-Info "Starting KmiDi deployment for Windows..."
    Log-Info "Deploy directory: $DeployDir"
    Log-Info "Data directory: $DataDir"
    Log-Info "Virtual environment: $VenvDir"
    
    Test-Prerequisites
    New-DirectoryStructure
    New-VirtualEnvironment
    Install-Dependencies
    Copy-ApplicationFiles
    New-WindowsService
    Test-Health
    
    Log-Success "Deployment complete!"
    Write-Host ""
    Log-Info "Next steps:"
    Log-Info "  1. Edit configuration: $DeployDir\config\.env"
    Log-Info "  2. Check service status: Get-Service KmiDiAPI"
    Log-Info "  3. View logs: Get-Content $LogDir\api.err.log -Tail 20"
    Log-Info "  4. Test API: Invoke-WebRequest -Uri http://localhost:8000/health"
    Log-Info "  5. Stop service: Stop-Service KmiDiAPI"
    Log-Info "  6. Start service: Start-Service KmiDiAPI"
}

# Check if running as administrator (required for Windows services)
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Run main function
if (Test-Administrator) {
    Main
}
else {
    Log-Warning "Not running as administrator. Some features may not work."
    Log-Info "For full service installation, run PowerShell as Administrator"
    Main
}
