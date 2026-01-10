#!/bin/bash
# KmiDi Deployment Script - Linux/macOS
# ======================================
# Automated deployment script for production environments

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
DEPLOY_DIR="${DEPLOY_DIR:-/opt/kmidi}"
VENV_DIR="${VENV_DIR:-$DEPLOY_DIR/venv}"
DATA_DIR="${DATA_DIR:-/data/kmidi}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=0
    
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is not installed"
        missing=1
    fi
    
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        missing=1
    fi
    
    if [ "$missing" -eq 1 ]; then
        log_error "Please install missing prerequisites"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create deployment directory structure
create_directory_structure() {
    log_info "Creating directory structure..."
    
    sudo mkdir -p "$DEPLOY_DIR"
    sudo mkdir -p "$DATA_DIR"
    sudo mkdir -p "$DEPLOY_DIR/logs"
    sudo mkdir -p "$DEPLOY_DIR/config"
    
    # Set permissions
    if [ -n "${DEPLOY_USER:-}" ]; then
        sudo chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_DIR"
        sudo chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DATA_DIR"
    fi
    
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
    sudo cp -r "$PROJECT_ROOT/music_brain" "$DEPLOY_DIR/"
    sudo cp -r "$PROJECT_ROOT/api" "$DEPLOY_DIR/"
    sudo cp "$PROJECT_ROOT/pyproject.toml" "$DEPLOY_DIR/"
    
    # Copy configuration files
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        if [ ! -f "$DEPLOY_DIR/config/.env" ]; then
            sudo cp "$PROJECT_ROOT/.env.example" "$DEPLOY_DIR/config/.env"
            log_warning "Please edit $DEPLOY_DIR/config/.env with your configuration"
        fi
    fi
    
    # Set permissions
    if [ -n "${DEPLOY_USER:-}" ]; then
        sudo chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_DIR"
    fi
    
    log_success "Application files copied"
}

# Create systemd service file
create_systemd_service() {
    log_info "Creating systemd service file..."
    
    local service_file="/etc/systemd/system/kmidi-api.service"
    
    sudo tee "$service_file" > /dev/null <<EOF
[Unit]
Description=KmiDi Music Generation API
After=network.target

[Service]
Type=simple
User=${DEPLOY_USER:-$USER}
WorkingDirectory=$DEPLOY_DIR
Environment="PYTHONPATH=$DEPLOY_DIR"
Environment="KELLY_AUDIO_DATA_ROOT=$DATA_DIR"
Environment="LOG_LEVEL=INFO"
EnvironmentFile=$DEPLOY_DIR/config/.env
ExecStart=$VENV_DIR/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    log_success "Systemd service file created"
}

# Start service
start_service() {
    log_info "Starting KmiDi API service..."
    
    sudo systemctl enable kmidi-api
    sudo systemctl start kmidi-api
    
    # Wait a moment and check status
    sleep 2
    
    if sudo systemctl is-active --quiet kmidi-api; then
        log_success "Service started successfully"
    else
        log_error "Service failed to start. Check logs with: sudo journalctl -u kmidi-api"
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
        log_info "Check service status with: sudo systemctl status kmidi-api"
    fi
}

# Main deployment function
main() {
    log_info "Starting KmiDi deployment..."
    log_info "Deploy directory: $DEPLOY_DIR"
    log_info "Data directory: $DATA_DIR"
    log_info "Virtual environment: $VENV_DIR"
    
    check_prerequisites
    create_directory_structure
    setup_venv
    install_dependencies
    copy_application_files
    
    # Create systemd service (Linux only)
    if command -v systemctl &> /dev/null && [ "$(uname)" != "Darwin" ]; then
        create_systemd_service
        start_service
        health_check
    else
        log_warning "systemd not available. Service not configured."
        log_info "To run manually:"
        log_info "  source $VENV_DIR/bin/activate"
        log_info "  cd $DEPLOY_DIR"
        log_info "  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
    fi
    
    log_success "Deployment complete!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Edit configuration: $DEPLOY_DIR/config/.env"
    log_info "  2. Check service status: sudo systemctl status kmidi-api"
    log_info "  3. View logs: sudo journalctl -u kmidi-api -f"
    log_info "  4. Test API: curl http://localhost:8000/health"
}

# Run main function
main "$@"
