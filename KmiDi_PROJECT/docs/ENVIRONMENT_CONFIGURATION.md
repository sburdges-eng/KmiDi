# Environment Configuration Guide

**Date**: 2025-01-02  
**Status**: ✅ **COMPLETE** - Comprehensive environment configuration documented

## Overview

KmiDi uses environment variables for configuration across all components. This guide documents all available environment variables and how to configure them for different deployment scenarios.

## Quick Start

1. **Copy the example file**:
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` with your values**:
   ```bash
   # Required for production
   KELLY_AUDIO_DATA_ROOT=/path/to/audio/data
   LOG_LEVEL=INFO
   ```

3. **Load environment variables**:
   ```bash
   # Automatically loaded by python-dotenv (if installed)
   # Or manually source:
   source .env  # Linux/macOS
   # Or export manually
   ```

## Environment Variables

### Core Application Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `ENVIRONMENT` | No | `development` | Application environment: `development`, `staging`, `production` |

### Storage Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KELLY_AUDIO_DATA_ROOT` | **Yes (Production)** | `~/.kelly/audio-data` | Primary audio data directory (full path) |
| `KELLY_SSD_PATH` | No | Auto-detected | External SSD mount point |
| `KELLY_LOGS_DIR` | No | `logs/training` | Directory for training logs |

**Storage Resolution Order**:
1. `KELLY_AUDIO_DATA_ROOT` (explicit path, highest priority)
2. `KELLY_SSD_PATH/kelly-audio-data` (data directory mount point)
3. Auto-detected platform paths (local storage first, then legacy SSD mounts if remounted)
4. Fallback: `~/.kelly/audio-data` (safe, always writable)

**Note**: Files have been moved from external SSD to local storage (2025-01-09).
New default location: `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/`

### FastAPI Service Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | API server host |
| `API_PORT` | No | `8000` | API server port |
| `API_WORKERS` | No | `4` | Number of uvicorn workers |
| `CORS_ORIGINS` | No | `*` | Comma-separated list of allowed CORS origins |
| `TRUSTED_HOSTS` | No | `localhost,127.0.0.1` | Comma-separated list of trusted hosts |
| `RATE_LIMIT_PER_MINUTE` | No | `60` | Rate limit per minute per IP |
| `RATE_LIMIT_PER_HOUR` | No | `1000` | Rate limit per hour per IP |
| `API_KEY` | No | - | API key for authentication (optional) |

### Streamlit Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | No | `8501` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | No | `0.0.0.0` | Streamlit server address |

### ML/AI Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KELLY_DEVICE` | No | `auto` | Device for ML training: `auto`, `mps`, `cuda`, `cpu` |
| `KELLY_TRAINING_BUDGET` | No | `100.0` | Maximum training budget in USD |
| `USE_NEURAL_MODELS` | No | `false` | Use neural models for emotion recognition |

### Ollama LLM Configuration (Optional)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_MODEL` | No | `mistral` | Ollama model name |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama host URL |
| `OLLAMA_TIMEOUT` | No | `60` | Timeout in seconds |
| `OLLAMA_TEMPERATURE` | No | `0.7` | Temperature (0.0-1.0) |
| `OLLAMA_MAX_TOKENS` | No | `512` | Maximum tokens |

### External API Keys (Optional)

These are only needed if using specific features:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | OpenAI API key (for GPT models) |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (for Claude models) |
| `GOOGLE_API_KEY` | No | Google API key (for Gemini models) |
| `XAI_API_KEY` | No | xAI API key (for Grok models) |
| `GITHUB_TOKEN` | No | GitHub token (for repository analysis) |
| `FREESOUND_API_KEY` | No | Freesound API key (for audio samples) |

### Security Configuration (Production)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | **Yes (Production)** | - | Secret key for JWT tokens (generate with: `openssl rand -hex 32`) |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| `JWT_EXPIRATION` | No | `3600` | JWT token expiration (seconds) |
| `HTTPS_REDIRECT` | No | `false` | Enable HTTPS redirect |
| `SSL_CERT_PATH` | No | - | SSL certificate path |
| `SSL_KEY_PATH` | No | - | SSL private key path |

### Development/Testing

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEBUG` | No | `false` | Enable debug mode (development only) |
| `RELOAD` | No | `false` | Enable hot reload (development only) |
| `TEST_MODE` | No | `false` | Test mode (disables external API calls) |

## Configuration Examples

### Development

```bash
# .env (development)
LOG_LEVEL=DEBUG
ENVIRONMENT=development
KELLY_AUDIO_DATA_ROOT=./data
API_PORT=8000
RELOAD=true
DEBUG=true
```

### Production

```bash
# .env (production)
LOG_LEVEL=INFO
ENVIRONMENT=production
KELLY_AUDIO_DATA_ROOT=/data/audio
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=https://kmidi.com,https://api.kmidi.com
TRUSTED_HOSTS=api.kmidi.com,kmidi.com
SECRET_KEY=your-secret-key-here
HTTPS_REDIRECT=true
```

### Docker/Container

```bash
# Environment variables in docker-compose.yml
environment:
  - KELLY_AUDIO_DATA_ROOT=/data
  - LOG_LEVEL=INFO
  - API_HOST=0.0.0.0
  - API_PORT=8000
  - API_WORKERS=4
```

## Loading Environment Variables

### Python (automatic)

If `python-dotenv` is installed, environment variables are automatically loaded from `.env`:

```python
# In api/main.py (already configured)
from dotenv import load_dotenv
load_dotenv()  # Loads .env automatically
```

### Python (manual)

```python
import os

# Read environment variable
data_root = os.getenv("KELLY_AUDIO_DATA_ROOT", "/default/path")
log_level = os.getenv("LOG_LEVEL", "INFO")
```

### Shell Script

```bash
# Source .env file
set -a  # Automatically export all variables
source .env
set +a

# Or export manually
export KELLY_AUDIO_DATA_ROOT=/data
export LOG_LEVEL=INFO
```

### Docker

```dockerfile
# In Dockerfile
ENV KELLY_AUDIO_DATA_ROOT=/data
ENV LOG_LEVEL=INFO

# Or in docker-compose.yml
environment:
  - KELLY_AUDIO_DATA_ROOT=/data
  - LOG_LEVEL=INFO
```

## Security Best Practices

### 1. Never Commit `.env` Files

Add to `.gitignore`:
```
.env
.env.local
.env.*.local
```

### 2. Use `.env.example` as Template

- Create `.env.example` with placeholder values
- Document all required variables
- Never include secrets in `.env.example`

### 3. Generate Strong Secret Keys

```bash
# Generate secret key
openssl rand -hex 32
```

### 4. Use Different Configurations per Environment

- `.env.development` - Development settings
- `.env.staging` - Staging settings
- `.env.production` - Production settings (store securely)

### 5. Rotate Secrets Regularly

- Change API keys periodically
- Update `SECRET_KEY` on rotation
- Document rotation process

## Validation

### Check Configuration

```bash
# Validate environment variables
python scripts/validate_configs.py

# Or check manually
python -c "import os; print(os.getenv('KELLY_AUDIO_DATA_ROOT'))"
```

### Required Variables for Production

```bash
# Check required variables are set
REQUIRED_VARS="KELLY_AUDIO_DATA_ROOT LOG_LEVEL SECRET_KEY"
for var in $REQUIRED_VARS; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set"
        exit 1
    fi
done
```

## Troubleshooting

### Environment Variables Not Loading

1. **Check `.env` file exists**:
   ```bash
   ls -la .env
   ```

2. **Verify python-dotenv is installed**:
   ```bash
   pip install python-dotenv
   ```

3. **Check file permissions**:
   ```bash
   chmod 600 .env  # Secure permissions
   ```

### Variables Not Taking Effect

1. **Restart application** after changing `.env`
2. **Check for typos** in variable names
3. **Verify variable is being read**:
   ```python
   import os
   print(os.getenv("VARIABLE_NAME"))
   ```

### Security Issues

1. **Check `.env` is in `.gitignore`**:
   ```bash
   git check-ignore .env
   ```

2. **Verify no secrets in git history**:
   ```bash
   git log --all --full-history -- .env
   ```

## Examples by Deployment Type

### Local Development

```bash
# .env
LOG_LEVEL=DEBUG
KELLY_AUDIO_DATA_ROOT=./data
API_PORT=8000
RELOAD=true
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - KELLY_AUDIO_DATA_ROOT=/data
      - LOG_LEVEL=INFO
      - API_WORKERS=4
```

### Kubernetes

```yaml
# deployment.yaml
env:
  - name: KELLY_AUDIO_DATA_ROOT
    value: "/data"
  - name: LOG_LEVEL
    value: "INFO"
  - name: SECRET_KEY
    valueFrom:
      secretKeyRef:
        name: kmidi-secrets
        key: secret-key
```

### Systemd Service

```ini
# /etc/systemd/system/kmidi-api.service
[Service]
Environment="KELLY_AUDIO_DATA_ROOT=/data"
Environment="LOG_LEVEL=INFO"
Environment="API_PORT=8000"
```

## References

- [python-dotenv Documentation](https://github.com/theskumar/python-dotenv)
- [12-Factor App: Config](https://12factor.net/config)
- [Environment Variables Best Practices](https://www.twilio.com/blog/environment-variables-python)
