# KmiDi Deployment Guide

> Comprehensive guide for deploying KmiDi in production environments

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Deployment Options](#deployment-options)
5. [Configuration](#configuration)
6. [Platform-Specific Instructions](#platform-specific-instructions)
7. [Docker Deployment](#docker-deployment)
8. [Health Monitoring](#health-monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Procedures](#rollback-procedures)
11. [Security Considerations](#security-considerations)

---

## Overview

KmiDi is an emotion-driven music generation system with the following deployable components:

| Component | Port | Description |
|-----------|------|-------------|
| **FastAPI** | 8000 | REST API for music generation |
| **Streamlit** | 8501 | Web-based demo UI |
| **LLM ONNX** | 8008 | Optional language model service |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer / Reverse Proxy            │
│                        (nginx/traefik/cloudflare)            │
└─────────────────────┬───────────────────┬───────────────────┘
                      │                   │
         ┌────────────▼────────┐ ┌────────▼────────┐
         │   FastAPI Service   │ │ Streamlit UI    │
         │      (port 8000)    │ │   (port 8501)   │
         └────────────┬────────┘ └─────────────────┘
                      │
         ┌────────────▼────────┐
         │    Music Brain      │
         │   (Python Package)  │
         └────────────┬────────┘
                      │
         ┌────────────▼────────┐
         │   ML Models / Data  │
         │    (Mounted Volumes)│
         └─────────────────────┘
```

---

## Prerequisites

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB | 50+ GB |
| Python | 3.9 | 3.11 |

### Software Requirements

- **Docker** 20.10+ and Docker Compose v2
- **Python** 3.9+ (for local development)
- **Git** (for source deployment)

### Optional

- **NVIDIA GPU** with CUDA 11.8+ (for accelerated ML inference)
- **Apple Silicon** with MPS support (automatic on macOS)

---

## Quick Start

### Using Deployment Scripts (Recommended)

```bash
# macOS
./deployment/scripts/deploy-macos.sh

# Linux
./deployment/scripts/deploy-linux.sh

# Windows (PowerShell)
.\deployment\scripts\deploy-windows.ps1
```

### Manual Docker Deployment

```bash
# 1. Clone and setup
cd KmiDi_PROJECT
cp env.example .env
# Edit .env with your settings

# 2. Build and run
docker build -t kmidi-api:prod -f deployment/docker/Dockerfile.prod .
docker run -d \
  --name kmidi-api \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  kmidi-api:prod

# 3. Verify
curl http://localhost:8000/health
```

---

## Deployment Options

### Option 1: Docker (Production)

Best for: Production environments, cloud deployment

```bash
# Build production image
docker build -t kmidi-api:prod -f deployment/docker/Dockerfile.prod .

# Run with volumes
docker run -d \
  --name kmidi-api \
  -p 8000:8000 \
  -v /path/to/data:/data:ro \
  -v /path/to/models:/models:ro \
  -v /path/to/output:/output:rw \
  --env-file .env \
  --restart unless-stopped \
  kmidi-api:prod
```

### Option 2: Docker Compose (Full Stack)

Best for: Running multiple services together

```bash
cd deployment/docker
docker compose up -d

# View logs
docker compose logs -f

# Stop all
docker compose down
```

### Option 3: Local Development

Best for: Development and testing

```bash
# Install dependencies
pip install -r requirements-production.txt
pip install -e .

# Run API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or run Streamlit
streamlit run streamlit_app.py
```

### Option 4: Cloud Platforms

#### AWS ECS

```yaml
# task-definition.json
{
  "family": "kmidi-api",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-registry/kmidi-api:prod",
      "portMappings": [{"containerPort": 8000}],
      "environment": [
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3
      }
    }
  ]
}
```

#### Google Cloud Run

```bash
gcloud run deploy kmidi-api \
  --image gcr.io/your-project/kmidi-api:prod \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

#### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kmidi-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kmidi-api
  template:
    metadata:
      labels:
        app: kmidi-api
    spec:
      containers:
      - name: api
        image: your-registry/kmidi-api:prod
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "2"
```

---

## Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
cp env.example .env
```

#### Essential Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | production | Environment name |
| `LOG_LEVEL` | INFO | Logging level |
| `API_HOST` | 0.0.0.0 | API bind address |
| `API_PORT` | 8000 | API port |
| `API_WORKERS` | 4 | Number of worker processes |
| `CORS_ORIGINS` | * | Allowed CORS origins |
| `KELLY_AUDIO_DATA_ROOT` | /data | Audio data directory |

#### ML/Training Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KELLY_DEVICE` | auto | ML device (auto/cpu/cuda/mps) |
| `TRAINING_DEVICE` | auto | Training device |
| `TRAINING_BATCH_SIZE` | 64 | Batch size for training |

#### Monitoring Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | true | Enable /metrics endpoint |

### Volume Mounts

| Path | Purpose | Mode |
|------|---------|------|
| `/data` | Audio data files | read-only |
| `/models` | ML model files | read-only |
| `/output` | Generated outputs | read-write |
| `/logs` | Application logs | read-write |

---

## Platform-Specific Instructions

### macOS

```bash
# Prerequisites
brew install --cask docker

# Deploy
./deployment/scripts/deploy-macos.sh

# Check status
./deployment/scripts/deploy-macos.sh --status
```

**Apple Silicon Notes:**
- MPS acceleration is automatic
- Set `KELLY_DEVICE=mps` for explicit MPS usage
- Docker Desktop must have Rosetta enabled for some images

### Linux

```bash
# Prerequisites (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Deploy
./deployment/scripts/deploy-linux.sh

# With GPU support
./deployment/scripts/deploy-linux.sh --gpu
```

**NVIDIA GPU Notes:**
- Install NVIDIA Container Toolkit
- Verify with: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

### Windows

```powershell
# Prerequisites
# Install Docker Desktop from https://docs.docker.com/desktop/install/windows-install/
# Enable WSL2 backend

# Deploy (PowerShell as Administrator)
.\deployment\scripts\deploy-windows.ps1

# Check status
.\deployment\scripts\deploy-windows.ps1 -Status
```

---

## Docker Deployment

### Building Images

```bash
# Production API
docker build -t kmidi-api:prod -f deployment/docker/Dockerfile.prod .

# Development (with hot reload)
docker build -t kmidi-api:dev -f deployment/docker/Dockerfile.dev .

# Check image size (target: <500MB)
docker images kmidi-api:prod --format "{{.Size}}"
```

### Multi-Architecture Builds

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/kmidi-api:prod \
  -f deployment/docker/Dockerfile.prod \
  --push .
```

### Docker Compose Services

```bash
cd deployment/docker

# Start all services
docker compose up -d

# Start specific service
docker compose up -d daiw-cli

# View logs
docker compose logs -f api

# Stop all
docker compose down

# Clean up everything
docker compose down -v --remove-orphans
```

---

## Health Monitoring

### Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Full health check | `{"status": "healthy", ...}` |
| `GET /live` | Liveness probe | `{"status": "alive"}` |
| `GET /ready` | Readiness probe | `{"status": "ready"}` |
| `GET /metrics` | Prometheus metrics | Text format metrics |

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1704931200.0,
  "services": {
    "music_brain": true,
    "api": true
  },
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "memory_available_mb": 4096.0
  }
}
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kmidi-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: kmidi
    rules:
      - alert: KmiDiAPIDown
        expr: up{job="kmidi-api"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "KmiDi API is down"

      - alert: KmiDiHighErrorRate
        expr: kmidi_api_error_rate > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "KmiDi API error rate > 5%"
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs kmidi-api

# Common causes:
# - Port already in use: lsof -i :8000
# - Missing .env file
# - Invalid environment variables
```

#### Health Check Failing

```bash
# Test health endpoint
curl -v http://localhost:8000/health

# Check if music_brain is loading
docker exec kmidi-api python -c "import music_brain; print('OK')"

# Common causes:
# - Missing dependencies
# - Model files not mounted
# - Insufficient memory
```

#### Performance Issues

```bash
# Check resource usage
docker stats kmidi-api

# Increase workers (in .env)
API_WORKERS=8

# Check for memory leaks
docker exec kmidi-api python -c "import psutil; print(psutil.virtual_memory())"
```

#### GPU Not Detected

```bash
# Linux: Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check device in container
docker exec kmidi-api python -c "import torch; print(torch.cuda.is_available())"
```

### Debug Mode

```bash
# Run with debug logging
docker run -it --rm \
  -e LOG_LEVEL=DEBUG \
  -e DEBUG=true \
  -p 8000:8000 \
  kmidi-api:prod

# Interactive shell
docker run -it --rm kmidi-api:prod /bin/bash
```

---

## Rollback Procedures

### Docker

```bash
# List available images
docker images kmidi-api

# Stop current
docker stop kmidi-api
docker rm kmidi-api

# Run previous version
docker run -d \
  --name kmidi-api \
  -p 8000:8000 \
  --env-file .env \
  kmidi-api:previous-tag

# Verify
curl http://localhost:8000/health
```

### Docker Compose

```bash
# Roll back to previous version
cd deployment/docker
git checkout HEAD~1 docker-compose.yml Dockerfile.prod
docker compose up -d --build

# Or use specific image tag
docker compose pull  # if using registry
docker compose up -d
```

### Kubernetes

```bash
# View rollout history
kubectl rollout history deployment/kmidi-api

# Rollback to previous
kubectl rollout undo deployment/kmidi-api

# Rollback to specific revision
kubectl rollout undo deployment/kmidi-api --to-revision=2
```

---

## Security Considerations

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure specific `CORS_ORIGINS` (not `*`)
- [ ] Use HTTPS via reverse proxy
- [ ] Set up rate limiting
- [ ] Generate secure `SECRET_KEY`
- [ ] Run as non-root user (default in Dockerfile)
- [ ] Use read-only volumes where possible
- [ ] Enable monitoring and alerting
- [ ] Regular security updates

### Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name api.kmidi.app;

    ssl_certificate /etc/ssl/certs/kmidi.crt;
    ssl_certificate_key /etc/ssl/private/kmidi.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint (no rate limit)
    location /health {
        proxy_pass http://localhost:8000/health;
    }
}
```

### Network Security

```bash
# Create isolated network
docker network create --driver bridge kmidi-network

# Run with network isolation
docker run -d \
  --name kmidi-api \
  --network kmidi-network \
  -p 8000:8000 \
  kmidi-api:prod
```

---

## Support

For issues and questions:

1. Check this guide's troubleshooting section
2. Review container logs: `docker logs kmidi-api`
3. Check GitHub Issues
4. Contact the development team

---

*Last Updated: 2026-01-11*
