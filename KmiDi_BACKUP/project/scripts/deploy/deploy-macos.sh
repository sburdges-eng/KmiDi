#!/bin/bash
# KmiDi Deployment Script - macOS
# ================================
# Automated deployment script for macOS using launchd

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/.kmidi}"
VENV_DIR="${VENV_DIR:-$DEPLOY_DIR/venv}"
DATA_DIR="${DATA_DIR:-$HOME/.kmidi/data}"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.kmidi.api.plist"

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
check_macos() {
    if [ "$(uname)" != "Darwin" ]; then
        log_error "This script is for macOS only"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is not installed. Install with: brew install python3"
        exit 1
    fi
    
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create deployment directory structure
create_directory_structure() {
    log_info "Creating directory structure..."
    
    mkdir -p "$DEPLOY_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$DEPLOY_DIR/logs"
    mkdir -p "$DEPLOY_DIR/config"
    mkdir -p "$HOME/Library/LaunchAgents"
    
    log_success "Directory structure created"
}

# Create virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists, recreating..."
        rm -rf "$VENV_DIR"
    fi
    
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created"
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    source "$VENV_DIR/bin/activate"
    
    # Install production requirements
    if [ -f "$PROJECT_ROOT/requirements-production.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements-production.txt"
    else
        log_warning "requirements-production.txt not found, using api/requirements.txt"
        pip install -r "$PROJECT_ROOT/api/requirements.txt"
    fi
    
    # Install project in development mode
    pip install -e "$PROJECT_ROOT"
    
    log_success "Dependencies installed"
}

# Copy application files
copy_application_files() {
    log_info "Copying application files..."
    
    # Copy application code
    cp -r "$PROJECT_ROOT/music_brain" "$DEPLOY_DIR/"
    cp -r "$PROJECT_ROOT/api" "$DEPLOY_DIR/"
    cp "$PROJECT_ROOT/pyproject.toml" "$DEPLOY_DIR/"
    
    # Copy configuration files
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        if [ ! -f "$DEPLOY_DIR/config/.env" ]; then
            cp "$PROJECT_ROOT/.env.example" "$DEPLOY_DIR/config/.env"
            log_warning "Please edit $DEPLOY_DIR/config/.env with your configuration"
        fi
    fi
    
    log_success "Application files copied"
}

# Create launchd plist
create_launchd_plist() {
    log_info "Creating launchd plist file..."
    
    cat > "$LAUNCHD_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kmidi.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_DIR/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>api.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
        <string>--workers</string>
        <string>4</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$DEPLOY_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$DEPLOY_DIR</string>
        <key>KELLY_AUDIO_DATA_ROOT</key>
        <string>$DATA_DIR</string>
        <key>LOG_LEVEL</key>
        <string>INFO</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$DEPLOY_DIR/logs/api.out.log</string>
    <key>StandardErrorPath</key>
    <string>$DEPLOY_DIR/logs/api.err.log</string>
</dict>
</plist>
EOF
    
    # Load environment variables from .env if it exists
    if [ -f "$DEPLOY_DIR/config/.env" ]; then
        log_info "Loading environment variables from .env file..."
        # Note: launchd doesn't support .env files directly
        # Users should export variables in the plist or use a wrapper script
    fi
    
    log_success "Launchd plist created"
}

# Start service
start_service() {
    log_info "Starting KmiDi API service..."
    
    # Unload if already running
    if launchctl list | grep -q "com.kmidi.api"; then
        log_info "Unloading existing service..."
        launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
    fi
    
    # Load and start
    launchctl load "$LAUNCHD_PLIST"
    
    sleep 2
    
    if launchctl list | grep -q "com.kmidi.api"; then
        log_success "Service started successfully"
    else
        log_error "Service failed to start"
        log_info "Check logs: tail -f $DEPLOY_DIR/logs/api.err.log"
        exit 1
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    sleep 3
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_warning "Health check failed. Service may still be starting."
        log_info "Check service status: launchctl list | grep kmidi"
        log_info "Check logs: tail -f $DEPLOY_DIR/logs/api.err.log"
    fi
}

# Main deployment function
main() {
    log_info "Starting KmiDi deployment for macOS..."
    log_info "Deploy directory: $DEPLOY_DIR"
    log_info "Data directory: $DATA_DIR"
    log_info "Virtual environment: $VENV_DIR"
    
    check_macos
    check_prerequisites
    create_directory_structure
    setup_venv
    install_dependencies
    copy_application_files
    create_launchd_plist
    start_service
    health_check
    
    log_success "Deployment complete!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Edit configuration: $DEPLOY_DIR/config/.env"
    log_info "  2. Check service status: launchctl list | grep kmidi"
    log_info "  3. View logs: tail -f $DEPLOY_DIR/logs/api.err.log"
    log_info "  4. Test API: curl http://localhost:8000/health"
    log_info "  5. Stop service: launchctl unload $LAUNCHD_PLIST"
    log_info "  6. Start service: launchctl load $LAUNCHD_PLIST"
}

# Run main function
main "$@"
