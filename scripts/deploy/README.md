# KmiDi Deployment Scripts

Automated deployment scripts for production environments on Linux, macOS, and Windows.

## Quick Start

### Linux

```bash
chmod +x scripts/deploy/deploy.sh
sudo scripts/deploy/deploy.sh
```

### macOS

```bash
chmod +x scripts/deploy/deploy-macos.sh
./scripts/deploy/deploy-macos.sh
```

### Windows (PowerShell)

```powershell
# Run PowerShell as Administrator
.\scripts\deploy\deploy-windows.ps1
```

## Features

- ✅ **Automatic dependency installation**
- ✅ **Virtual environment setup**
- ✅ **Service configuration** (systemd/launchd/Windows Service)
- ✅ **Health checks**
- ✅ **Logging and monitoring setup**
- ✅ **Environment variable configuration**

## Detailed Usage

### Linux Deployment

The Linux deployment script creates a systemd service for automatic startup.

**Prerequisites**:
- Python 3.9+
- sudo access
- systemd (most modern Linux distributions)

**Deployment**:
```bash
# Basic deployment
sudo scripts/deploy/deploy.sh

# Custom deployment directory
DEPLOY_DIR=/opt/kmidi sudo scripts/deploy/deploy.sh

# Custom data directory
DATA_DIR=/data/kmidi sudo scripts/deploy/deploy.sh

# Custom user
DEPLOY_USER=kmidi sudo scripts/deploy/deploy.sh
```

**Service Management**:
```bash
# Start service
sudo systemctl start kmidi-api

# Stop service
sudo systemctl stop kmidi-api

# Restart service
sudo systemctl restart kmidi-api

# Check status
sudo systemctl status kmidi-api

# View logs
sudo journalctl -u kmidi-api -f
```

### macOS Deployment

The macOS deployment script creates a launchd service for automatic startup.

**Prerequisites**:
- Python 3.9+
- Homebrew (recommended)

**Deployment**:
```bash
# Basic deployment
./scripts/deploy/deploy-macos.sh

# Custom deployment directory
DEPLOY_DIR=$HOME/.kmidi ./scripts/deploy/deploy-macos.sh
```

**Service Management**:
```bash
# Check service status
launchctl list | grep kmidi

# Start service
launchctl load ~/Library/LaunchAgents/com.kmidi.api.plist

# Stop service
launchctl unload ~/Library/LaunchAgents/com.kmidi.api.plist

# View logs
tail -f ~/.kmidi/logs/api.err.log
```

### Windows Deployment

The Windows deployment script supports both manual execution and Windows Service installation.

**Prerequisites**:
- Python 3.9+
- PowerShell (Administrator privileges for service installation)
- NSSM (Non-Sucking Service Manager) for Windows Service (optional)

**Deployment**:
```powershell
# Run PowerShell as Administrator
.\scripts\deploy\deploy-windows.ps1

# Custom deployment directory
$env:DEPLOY_DIR = "D:\KmiDi"
.\scripts\deploy\deploy-windows.ps1
```

**Service Management**:
```powershell
# Check service status
Get-Service KmiDiAPI

# Start service
Start-Service KmiDiAPI

# Stop service
Stop-Service KmiDiAPI

# View logs
Get-Content C:\KmiDi\logs\api.err.log -Tail 20
```

**Manual Execution** (if service not installed):
```powershell
cd C:\KmiDi
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

After deployment, edit the configuration file:

- **Linux/macOS**: `$DEPLOY_DIR/config/.env`
- **Windows**: `C:\KmiDi\config\.env`

See `docs/ENVIRONMENT_CONFIGURATION.md` for all available configuration options.

## Post-Deployment

1. **Edit Configuration**:
   ```bash
   # Linux/macOS
   sudo nano $DEPLOY_DIR/config/.env
   
   # Windows
   notepad C:\KmiDi\config\.env
   ```

2. **Test Health Endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Check Logs**:
   ```bash
   # Linux
   sudo journalctl -u kmidi-api -f
   
   # macOS
   tail -f ~/.kmidi/logs/api.err.log
   
   # Windows
   Get-Content C:\KmiDi\logs\api.err.log -Tail 20
   ```

## Troubleshooting

### Service Won't Start

**Linux**:
```bash
# Check service logs
sudo journalctl -u kmidi-api -n 50

# Check if port is in use
sudo lsof -i :8000

# Check file permissions
sudo chown -R kmidi:kmidi /opt/kmidi
```

**macOS**:
```bash
# Check launchd errors
launchctl error 1

# Check logs
tail -f ~/.kmidi/logs/api.err.log

# Reload service
launchctl unload ~/Library/LaunchAgents/com.kmidi.api.plist
launchctl load ~/Library/LaunchAgents/com.kmidi.api.plist
```

**Windows**:
```powershell
# Check service errors
Get-EventLog -LogName Application -Source KmiDiAPI -Newest 10

# Check if port is in use
netstat -ano | findstr :8000

# Check file permissions
icacls C:\KmiDi
```

### Port Already in Use

Change the port in configuration:
```bash
# In .env file
API_PORT=8001
```

Then restart the service.

### Permission Denied

**Linux/macOS**:
```bash
# Fix ownership
sudo chown -R $USER:$USER $DEPLOY_DIR

# Fix permissions
chmod -R 755 $DEPLOY_DIR
```

**Windows**:
```powershell
# Run PowerShell as Administrator
# Or grant permissions to current user
icacls C:\KmiDi /grant $env:USERNAME:F /T
```

### Health Check Fails

1. Check service is running
2. Check port is accessible
3. Check firewall rules
4. Review logs for errors

## Updating

To update an existing deployment:

1. **Pull latest code**
2. **Re-run deployment script** (it will recreate virtual environment)
3. **Restart service**

The scripts are idempotent and safe to run multiple times.

## Uninstallation

### Linux

```bash
# Stop and disable service
sudo systemctl stop kmidi-api
sudo systemctl disable kmidi-api

# Remove service file
sudo rm /etc/systemd/system/kmidi-api.service
sudo systemctl daemon-reload

# Remove deployment directory
sudo rm -rf /opt/kmidi
```

### macOS

```bash
# Stop and unload service
launchctl unload ~/Library/LaunchAgents/com.kmidi.api.plist

# Remove service file
rm ~/Library/LaunchAgents/com.kmidi.api.plist

# Remove deployment directory
rm -rf ~/.kmidi
```

### Windows

```powershell
# Stop and remove service (if using NSSM)
nssm stop KmiDiAPI
nssm remove KmiDiAPI confirm

# Remove deployment directory
Remove-Item -Recurse -Force C:\KmiDi
```

## Advanced Configuration

### Custom Environment Variables

You can set environment variables before running the deployment script:

```bash
# Linux/macOS
export KELLY_AUDIO_DATA_ROOT=/custom/path
export LOG_LEVEL=DEBUG
./scripts/deploy/deploy.sh
```

```powershell
# Windows
$env:KELLY_AUDIO_DATA_ROOT = "D:\Custom\Path"
$env:LOG_LEVEL = "DEBUG"
.\scripts\deploy\deploy-windows.ps1
```

### Multiple Workers

Edit the service configuration to adjust worker count:

**Linux** (systemd service file):
```ini
ExecStart=/opt/kmidi/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 8
```

**macOS** (launchd plist):
```xml
<string>--workers</string>
<string>8</string>
```

**Windows** (PowerShell):
```powershell
# Edit service arguments
nssm set KmiDiAPI AppParameters "-m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 8"
```

## Security Considerations

1. **Run as non-root user** (Linux deployment creates `kmidi` user)
2. **Set secure file permissions** (600 for .env files)
3. **Use HTTPS in production** (configure reverse proxy)
4. **Enable firewall rules** (only allow necessary ports)
5. **Regular security updates** (keep Python and dependencies updated)

## Support

For issues or questions:
- Check logs first
- Review `docs/ENVIRONMENT_CONFIGURATION.md`
- Review `docs/DOCKER_OPTIMIZATION.md` for Docker deployments
