# Health Monitoring Guide

**Date**: 2025-01-02  
**Status**: ✅ **COMPLETE** - Health endpoints and monitoring configured

## Overview

KmiDi provides comprehensive health monitoring endpoints for production deployments, load balancers, and monitoring systems.

## Health Endpoints

### `/health` - Health Check

Comprehensive health check endpoint for monitoring and load balancers.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1704729600.0,
  "services": {
    "music_brain": true,
    "api": true,
    "system": {
      "cpu_percent": 15.5,
      "memory_percent": 45.2,
      "memory_available_mb": 4096
    }
  }
}
```

**Status Codes**:
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is degraded

**Usage**:
```bash
curl http://localhost:8000/health
```

### `/ready` - Readiness Probe

Checks if service is ready to accept traffic. Used by Kubernetes and load balancers.

**Endpoint**: `GET /ready`

**Response**:
```json
{
  "status": "ready",
  "timestamp": 1704729600.0
}
```

**Status Codes**:
- `200 OK` - Service is ready
- `503 Service Unavailable` - Service not ready (Music Brain unavailable)

**Usage**:
```bash
curl http://localhost:8000/ready
```

### `/live` - Liveness Probe

Checks if service is alive. Used by Kubernetes to determine if container should be restarted.

**Endpoint**: `GET /live`

**Response**:
```json
{
  "status": "alive",
  "timestamp": 1704729600.0
}
```

**Status Codes**:
- `200 OK` - Service is alive

**Usage**:
```bash
curl http://localhost:8000/live
```

### `/metrics` - Prometheus Metrics

Prometheus-compatible metrics endpoint (if enabled).

**Endpoint**: `GET /metrics`

**Configuration**: Set `ENABLE_METRICS=true` in environment variables.

**Response**:
```json
{
  "kmidi_api_requests_total": 1234,
  "kmidi_api_requests_in_flight": 5,
  "kmidi_api_request_duration_seconds": 0.125
}
```

**Status Codes**:
- `200 OK` - Metrics available
- `404 Not Found` - Metrics disabled

**Usage**:
```bash
curl http://localhost:8000/metrics
```

## Integration Examples

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Docker Compose

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kmidi-api
spec:
  template:
    spec:
      containers:
      - name: api
        image: kmidi-api:latest
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
```

### Nginx Load Balancer

```nginx
upstream kmidi_backend {
    server 127.0.0.1:8000;
    
    # Health check
    check interval=3000 rise=2 fall=3 timeout=1000;
    check_http_send "GET /health HTTP/1.0\r\n\r\n";
    check_http_expect_alive http_2xx http_3xx;
}
```

### Systemd Service

```ini
[Service]
# Health check via systemd
ExecStartPre=/usr/bin/curl -f http://localhost:8000/health || exit 1
```

## Monitoring Tools

### Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kmidi-api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard

Create dashboard with:
- Request rate
- Error rate
- Response time
- System metrics (CPU, memory)

### Uptime Monitoring

Services like UptimeRobot, Pingdom, or StatusCake:

```
Monitor URL: https://api.kmidi.com/health
Expected Status: 200
Check Interval: 1 minute
```

### Log Monitoring

Monitor logs for errors:

```bash
# Linux (systemd)
sudo journalctl -u kmidi-api -f | grep ERROR

# Docker
docker logs -f kmidi-api | grep ERROR

# Kubernetes
kubectl logs -f deployment/kmidi-api | grep ERROR
```

## Alerting

### Alert Rules (Prometheus)

```yaml
groups:
  - name: kmidi_alerts
    rules:
      - alert: KmiDiAPIDown
        expr: up{job="kmidi-api"} == 0
        for: 1m
        annotations:
          summary: "KmiDi API is down"
      
      - alert: KmiDiAPIHighErrorRate
        expr: rate(kmidi_api_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
      
      - alert: KmiDiAPIHighLatency
        expr: kmidi_api_request_duration_seconds > 1.0
        for: 5m
        annotations:
          summary: "High latency detected"
```

### Email Alerts

```bash
#!/bin/bash
# health_check_alert.sh
if ! curl -f http://localhost:8000/health; then
    echo "KmiDi API is down!" | mail -s "Alert: KmiDi API Down" admin@example.com
fi
```

### Slack Alerts

```python
import requests
import json

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            send_slack_alert("KmiDi API health check failed")
    except Exception as e:
        send_slack_alert(f"KmiDi API health check error: {e}")

def send_slack_alert(message):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    payload = {"text": message}
    requests.post(webhook_url, json=payload)
```

## Best Practices

1. **Use Appropriate Endpoints**:
   - `/health` for general monitoring
   - `/ready` for load balancer checks
   - `/live` for container orchestration

2. **Set Reasonable Intervals**:
   - Health checks: 30-60 seconds
   - Readiness checks: 5-10 seconds
   - Liveness checks: 10-30 seconds

3. **Monitor Key Metrics**:
   - Response time
   - Error rate
   - System resources (CPU, memory)
   - Service availability

4. **Set Up Alerting**:
   - Immediate alerts for downtime
   - Warning alerts for degraded performance
   - Threshold-based alerts for resource usage

5. **Log Health Check Results**:
   - Track health check history
   - Monitor trends over time
   - Identify patterns in failures

## Troubleshooting

### Health Check Fails

1. **Check Service Status**:
   ```bash
   # Linux
   sudo systemctl status kmidi-api
   
   # Docker
   docker ps | grep kmidi
   ```

2. **Check Logs**:
   ```bash
   # Linux
   sudo journalctl -u kmidi-api -n 50
   
   # Docker
   docker logs kmidi-api --tail 50
   ```

3. **Check Port**:
   ```bash
   # Check if port is in use
   sudo lsof -i :8000
   
   # Test endpoint manually
   curl -v http://localhost:8000/health
   ```

### Service Not Ready

- Music Brain service unavailable
- Check Music Brain dependencies
- Verify data directory access

### High Response Time

- Check system resources (CPU, memory)
- Review application logs
- Check database/API dependencies
- Consider scaling horizontally

## See Also

- [API Documentation](API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md)
