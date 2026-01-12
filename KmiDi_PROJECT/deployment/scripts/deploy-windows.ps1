# ============================================================================
# KmiDi Windows Deployment Script (PowerShell)
# ============================================================================
# Usage: .\deploy-windows.ps1 [options]
#
# Options:
#   -Dev        Deploy in development mode (with hot reload)
#   -Prod       Deploy in production mode (default)
#   -ApiOnly    Deploy only the FastAPI service
#   -Full       Deploy all services (API + Streamlit + LLM)
#   -Stop       Stop all running containers
#   -Clean      Stop and remove all containers, volumes
#   -Logs       Show container logs
#   -Status     Show deployment status
#   -Help       Show this help message
#
# Prerequisites:
#   - Docker Desktop for Windows installed and running
#   - WSL2 backend recommended
#   - PowerShell 5.1+ or PowerShell Core 7+
#
# ============================================================================

param(
    [switch]$Dev,
    [switch]$Prod,
    [switch]$ApiOnly,
    [switch]$Full,
    [switch]$Stop,
    [switch]$Clean,
    [switch]$Logs,
    [switch]$Status,
    [switch]$Help
)

# Configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName
$DockerDir = Join-Path $ProjectRoot "deployment\docker"

# Default settings
$Mode = if ($Dev) { "dev" } else { "prod" }
$Services = if ($Full) { "full" } else { "api" }

# Functions
function Write-Banner {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "║             KmiDi - Windows Deployment Script                ║" -ForegroundColor Blue
    Write-Host "║                    Music Intelligence API                    ║" -ForegroundColor Blue
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "▶ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✖ $Message" -ForegroundColor Red
}

function Write-Success {
    param([string]$Message)
    Write-Host "✔ $Message" -ForegroundColor Green
}

function Test-Prerequisites {
    Write-Step "Checking prerequisites..."

    # Check Docker
    try {
        $null = Get-Command docker -ErrorAction Stop
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop for Windows."
        Write-Host "  Download: https://docs.docker.com/desktop/install/windows-install/"
        exit 1
    }

    # Check Docker daemon
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Docker not running"
        }
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }

    # Check Docker Compose
    try {
        $composeVersion = docker compose version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Compose not available"
        }
    }
    catch {
        Write-Error "Docker Compose is not available. Please update Docker Desktop."
        exit 1
    }

    Write-Success "Prerequisites check passed"
}

function Initialize-Environment {
    Write-Step "Setting up environment..."

    Set-Location $ProjectRoot

    # Create .env if it doesn't exist
    $envFile = Join-Path $ProjectRoot ".env"
    $envExample = Join-Path $ProjectRoot "env.example"

    if (-not (Test-Path $envFile)) {
        if (Test-Path $envExample) {
            Copy-Item $envExample $envFile
            Write-Warning "Created .env from env.example. Please review and update settings."
        }
        else {
            Write-Error "No env.example found. Cannot create .env file."
            exit 1
        }
    }

    # Create necessary directories
    $dirs = @("output", "models", "logs", "data")
    foreach ($dir in $dirs) {
        $dirPath = Join-Path $ProjectRoot $dir
        if (-not (Test-Path $dirPath)) {
            New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
        }
    }

    # Set Windows-specific defaults
    $envContent = Get-Content $envFile -Raw
    if ($envContent -notmatch "KELLY_AUDIO_DATA_ROOT") {
        $defaultPath = Join-Path $env:USERPROFILE "kmidi-data"
        Add-Content $envFile "`nKELLY_AUDIO_DATA_ROOT=$defaultPath"
    }

    Write-Success "Environment configured"
}

function Build-Images {
    Write-Step "Building Docker images..."

    Set-Location $DockerDir

    if ($Mode -eq "dev") {
        docker compose build --no-cache daiw-dev
    }
    else {
        docker compose build
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build Docker images"
        exit 1
    }

    Write-Success "Docker images built"
}

function Deploy-Services {
    Write-Step "Deploying services..."

    Set-Location $DockerDir

    switch ($Services) {
        "api" {
            if ($Mode -eq "dev") {
                docker compose up -d daiw-dev
            }
            else {
                # Build and run production image
                docker build -t kmidi-api:prod -f Dockerfile.prod $ProjectRoot

                # Stop existing container if running
                docker stop kmidi-api 2>$null
                docker rm kmidi-api 2>$null

                # Run new container
                docker run -d `
                    --name kmidi-api `
                    -p 8000:8000 `
                    -v "${ProjectRoot}\data:/data:ro" `
                    -v "${ProjectRoot}\models:/models:ro" `
                    -v "${ProjectRoot}\output:/output:rw" `
                    --env-file "${ProjectRoot}\.env" `
                    --restart unless-stopped `
                    kmidi-api:prod
            }
        }
        "full" {
            docker compose up -d
        }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to deploy services"
        exit 1
    }

    Write-Success "Services deployed"
}

function Show-Status {
    Write-Step "Deployment Status"
    Write-Host ""

    # Show running containers
    Write-Host "Running Containers:" -ForegroundColor Blue
    docker ps --filter "name=kmidi" --filter "name=daiw" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Write-Host ""

    # Health checks
    Write-Host "Health Checks:" -ForegroundColor Blue

    # Check API
    try {
        $apiHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
        Write-Success "API: status=$($apiHealth.status)"
    }
    catch {
        Write-Warning "API: Not responding on port 8000"
    }

    # Check Streamlit
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
        Write-Success "Streamlit: Running on port 8501"
    }
    catch {
        Write-Warning "Streamlit: Not responding on port 8501"
    }

    Write-Host ""
    Write-Host "Endpoints:" -ForegroundColor Blue
    Write-Host "  API:        http://localhost:8000"
    Write-Host "  API Docs:   http://localhost:8000/docs"
    Write-Host "  Metrics:    http://localhost:8000/metrics"
    Write-Host "  Streamlit:  http://localhost:8501"
}

function Stop-Services {
    Write-Step "Stopping services..."

    docker stop kmidi-api 2>$null
    docker rm kmidi-api 2>$null

    Set-Location $DockerDir
    docker compose down 2>$null

    Write-Success "Services stopped"
}

function Clear-All {
    Write-Step "Cleaning up all containers and volumes..."

    Stop-Services

    Set-Location $DockerDir
    docker compose down -v --remove-orphans 2>$null

    # Remove images
    docker rmi kmidi-api:prod 2>$null
    docker rmi daiw-music-brain:latest 2>$null
    docker rmi daiw-music-brain:dev 2>$null

    Write-Success "Cleanup complete"
}

function Show-Logs {
    Set-Location $DockerDir
    docker compose logs -f
}

function Show-Help {
    Write-Host "Usage: .\deploy-windows.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Dev        Deploy in development mode (with hot reload)"
    Write-Host "  -Prod       Deploy in production mode (default)"
    Write-Host "  -ApiOnly    Deploy only the FastAPI service"
    Write-Host "  -Full       Deploy all services"
    Write-Host "  -Stop       Stop all running containers"
    Write-Host "  -Clean      Stop and remove all containers, volumes"
    Write-Host "  -Logs       Show container logs"
    Write-Host "  -Status     Show deployment status"
    Write-Host "  -Help       Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy-windows.ps1                 # Deploy API in production mode"
    Write-Host "  .\deploy-windows.ps1 -Dev            # Deploy in development mode"
    Write-Host "  .\deploy-windows.ps1 -Full           # Deploy all services"
    Write-Host "  .\deploy-windows.ps1 -Status         # Check deployment status"
    Write-Host "  .\deploy-windows.ps1 -Stop           # Stop all services"
}

# Main execution
Write-Banner

if ($Help) {
    Show-Help
    exit 0
}

if ($Stop) {
    Stop-Services
    Write-Host ""
    Write-Success "Done!"
    exit 0
}

if ($Clean) {
    Clear-All
    Write-Host ""
    Write-Success "Done!"
    exit 0
}

if ($Logs) {
    Show-Logs
    exit 0
}

if ($Status) {
    Show-Status
    Write-Host ""
    Write-Success "Done!"
    exit 0
}

# Default: deploy
Test-Prerequisites
Initialize-Environment
Build-Images
Deploy-Services
Write-Host ""
Show-Status
Write-Host ""
Write-Success "Done!"
