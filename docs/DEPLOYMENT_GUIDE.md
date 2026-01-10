# KmiDi Deployment Guide

**Date**: 2025-01-02  
**Status**: ✅ **COMPLETE** - Deployment scripts and documentation ready

## Overview

This guide covers deploying KmiDi to production environments on Linux, macOS, and Windows.

## Deployment Options

### 1. Automated Scripts (Recommended)

Use the provided deployment scripts for quick setup:

- **Linux**: `scripts/deploy/deploy.sh`
- **macOS**: `scripts/deploy/deploy-macos.sh`
- **Windows**: `scripts/deploy/deploy-windows.ps1`

See `scripts/deploy/README.md` for detailed usage.

### 2. Docker Deployment

Use Docker for containerized deployment:

```bash
cd api
docker-compose up -d
```

See `docs/DOCKER_OPTIMIZATION.md` for Docker-specific documentation.

### 3. Manual Deployment

Follow the manual steps below for custom deployments.

## Prerequisites

### All Platforms

- **Python 3.9+** (recommended: 3.11)
- **pip** (Python package manager)
- **Git** (for cloning repository)

### Linux

- **systemd** (for service management)
- **sudo** access
- **curl** (for health checks)

### macOS

- **Homebrew** (recommended, for Python installation)
- **launchd** (built-in, for service management)

### Windows

- **PowerShell 5.1+** (for deployment script)
- **Administrator privileges** (for Windows Service installation)
- **NSSM** (optional, for Windows Service management)

## Quick Start

### Linux

```bash
# Clone repository
git clone <repository-url>
cd KmiDi-1

# Run deployment script
chmod +x scripts/deploy/deploy.sh
sudo scripts/deploy/deploy.sh

# Check status
sudo systemctl status kmidi-api
```

### macOS

```bash
# Clone repository
git clone <repository-url>
cd KmiDi-1

# Run deployment script
chmod +x scripts/deploy/deploy-macos.sh
./scripts/deploy/deploy-macos.sh

# Check status
launchctl list | grep kmidi
```

### Windows

```powershell
# Clone repository
git clone <repository-url>
cd KmiDi-1

# Run deployment script (as Administrator)
.\scripts\deploy\deploy-windows.ps1

# Check status
Get-Service KmiDiAPI
```

## Manual Deployment Steps

### 1. Install Dependencies

```bash
# Install system dependencies (Linux)
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv libsndfile1

# Install system dependencies (macOS)
brew install python3 libsndfile

# Install system dependencies (Windows)
# Download and install Python from python.org
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Application

```bash
# Install production requirements
pip install -r requirements-production.txt

# Install application
pip install -e .
```

### 4. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

Required configuration variables:
- `KELLY_AUDIO_DATA_ROOT` - Audio data directory path
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)
- `API_PORT` - API server port (default: 8000)

### 5. Create Service

#### Linux (systemd)

Create `/etc/systemd/system/kmidi-api.service`:

```ini
[Unit]
Description=KmiDi Music Generation API
After=network.target

[Service]
Type=simple
User=kmidi
WorkingDirectory=/opt/kmidi
Environment="PYTHONPATH=/opt/kmidi"
Environment="KELLY_AUDIO_DATA_ROOT=/data"
Environment="LOG_LEVEL=INFO"
ExecStart=/opt/kmidi/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable kmidi-api
sudo systemctl start kmidi-api
```

#### macOS (launchd)

Create `~/Library/LaunchAgents/com.kmidi.api.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kmidi.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/user/.kmidi/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>api.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load and start:
```bash
launchctl load ~/Library/LaunchAgents/com.kmidi.api.plist
```

#### Windows (NSSM)

Download NSSM from https://nssm.cc/download

```powershell
# Install service
nssm install KmiDiAPI "C:\KmiDi\venv\Scripts\python.exe" "-m uvicorn api.main:app --host 0.0.0.0 --port 8000"

# Set working directory
nssm set KmiDiAPI AppDirectory "C:\KmiDi"

# Start service
nssm start KmiDiAPI
```

### 6. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Expected response
{"status":"ok","version":"1.0.0","services":{"music_brain":true,"api":true}}
```

## Configuration

See `docs/ENVIRONMENT_CONFIGURATION.md` for complete configuration reference.

### Essential Configuration

Create `.env` file with:

```bash
# Storage
KELLY_AUDIO_DATA_ROOT=/data/audio

# Logging
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security (production)
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=https://yourdomain.com
```

### Production Security

1. **Set strong SECRET_KEY**:
   ```bash
   openssl rand -hex 32
   ```

2. **Restrict CORS origins**:
   ```bash
   CORS_ORIGINS=https://kmidi.com,https://api.kmidi.com
   ```

3. **Use HTTPS** (via reverse proxy like nginx)

4. **Enable firewall**:
   ```bash
   # Linux (ufw)
   sudo ufw allow 8000/tcp
   
   # macOS
   # Configure via System Preferences > Security & Privacy > Firewall
   ```

## Monitoring

### Health Checks

The API provides a health endpoint:

```bash
curl http://localhost:8000/health
```

Monitor this endpoint for service availability.

### Logs

**Linux**:
```bash
# Systemd logs
sudo journalctl -u kmidi-api -f

# Application logs
tail -f /opt/kmidi/logs/api.log
```

**macOS**:
```bash
# Launchd logs
tail -f ~/.kmidi/logs/api.err.log
```

**Windows**:
```powershell
# Service logs
Get-Content C:\KmiDi\logs\api.err.log -Tail 20 -Wait
```

### Metrics

Enable metrics endpoint (if configured):

```bash
curl http://localhost:9090/metrics
```

## Scaling

### Horizontal Scaling

Run multiple instances behind a load balancer:

1. Deploy on multiple servers
2. Configure load balancer (nginx, HAProxy, etc.)
3. Enable sticky sessions if needed

### Vertical Scaling

Increase workers for single-instance scaling:

```bash
# Edit service configuration
# Change --workers 4 to --workers 8
```

## Maintenance

### Updates

1. **Pull latest code**
2. **Recreate virtual environment** (recommended)
3. **Restart service**

```bash
# Linux
sudo systemctl restart kmidi-api

# macOS
launchctl unload ~/Library/LaunchAgents/com.kmidi.api.plist
launchctl load ~/Library/LaunchAgents/com.kmidi.api.plist

# Windows
Restart-Service KmiDiAPI
```

### Backup

Backup important data:

```bash
# Configuration
tar -czf kmidi-config-backup.tar.gz /opt/kmidi/config

# Data directory
tar -czf kmidi-data-backup.tar.gz /data/kmidi
```

### Rollback

To rollback to previous version:

1. Restore code from backup
2. Recreate virtual environment
3. Restart service

## Troubleshooting

See `scripts/deploy/README.md` for troubleshooting guide.

### Common Issues

1. **Service won't start**: Check logs, verify Python path, check permissions
2. **Port already in use**: Change `API_PORT` or stop conflicting service
3. **Permission denied**: Check file ownership and permissions
4. **Health check fails**: Verify service is running and port is accessible

## Next Steps

- Set up reverse proxy (nginx, Apache)
- Configure SSL/TLS certificates
- Set up monitoring and alerting
- Configure backup strategy
- Set up CI/CD pipeline

## References

- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md)
- [Docker Deployment](DOCKER_OPTIMIZATION.md)
- [Deployment Scripts](../scripts/deploy/README.md)
