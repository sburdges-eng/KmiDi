# KmiDi API Documentation

**Version**: 1.0.0  
**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)  
**ReDoc**: `http://localhost:8000/redoc`

## Overview

The KmiDi Music Generation API provides REST endpoints for emotion-driven music generation, emotion analysis, and music interrogation.

## Authentication

Currently, the API does not require authentication. For production deployments, configure `API_KEY` environment variable and implement authentication middleware.

## Endpoints

### Health & Monitoring

#### `GET /health`

Health check endpoint for monitoring and load balancers.

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

#### `GET /ready`

Readiness probe - checks if service is ready to accept traffic.

**Response**:
```json
{
  "status": "ready",
  "timestamp": 1704729600.0
}
```

**Status Codes**:
- `200 OK` - Service is ready
- `503 Service Unavailable` - Service not ready

#### `GET /live`

Liveness probe - checks if service is alive.

**Response**:
```json
{
  "status": "alive",
  "timestamp": 1704729600.0
}
```

**Status Codes**:
- `200 OK` - Service is alive

#### `GET /metrics`

Prometheus-compatible metrics endpoint (if enabled).

**Query Parameters**:
- None

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

**Configuration**: Set `ENABLE_METRICS=true` in environment variables.

### Emotions

#### `GET /emotions`

List available emotional presets.

**Rate Limit**: 100 requests per minute

**Response**:
```json
{
  "emotions": ["calm", "grief", "joy", "sad", ...],
  "count": 6
}
```

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Music brain service unavailable
- `500 Internal Server Error` - Server error

**Example**:
```bash
curl http://localhost:8000/emotions
```

### Music Generation

#### `POST /generate`

Generate music from emotional intent.

**Rate Limit**: 10 requests per minute

**Request Body**:
```json
{
  "intent": {
    "core_wound": "loss of a loved one",
    "core_desire": "to remember and honor",
    "emotional_intent": "grief hidden as love",
    "technical": {
      "key": "F",
      "bpm": 72,
      "progression": ["F", "C", "Am", "Dm"],
      "genre": "acoustic"
    }
  },
  "output_format": "midi"
}
```

**Response**:
```json
{
  "status": "success",
  "result": {
    "midi_path": "/path/to/generated.mid",
    "chords": ["F", "C", "Am", "Dm"],
    "key": "F major",
    "tempo": 72,
    "metadata": {
      "intent": {...},
      "generated_at": "2025-01-02T12:00:00Z"
    }
  }
}
```

**Status Codes**:
- `200 OK` - Generation successful
- `400 Bad Request` - Invalid request
- `429 Too Many Requests` - Rate limit exceeded
- `503 Service Unavailable` - Music brain service unavailable
- `500 Internal Server Error` - Generation failed

**Example**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "grief hidden as love"
    }
  }'
```

### Interrogation

#### `POST /interrogate`

Interactive music creation through conversation.

**Rate Limit**: 20 requests per minute

**Request Body**:
```json
{
  "message": "I want something that feels like hope after loss",
  "session_id": "optional-session-id",
  "context": {
    "previous_messages": [...]
  }
}
```

**Response**:
```json
{
  "response": "I understand. Let's create something in F major, slow tempo...",
  "suggestions": {
    "key": "F",
    "tempo": 68,
    "mood": "hopeful"
  },
  "session_id": "session-uuid"
}
```

**Status Codes**:
- `200 OK` - Success
- `400 Bad Request` - Invalid request
- `429 Too Many Requests` - Rate limit exceeded
- `503 Service Unavailable` - Service unavailable
- `500 Internal Server Error` - Server error

**Example**:
```bash
curl -X POST http://localhost:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want something hopeful"
  }'
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Emotions endpoint**: 100 requests per minute
- **Generate endpoint**: 10 requests per minute
- **Interrogate endpoint**: 20 requests per minute

Rate limits are per IP address. When exceeded, the API returns:
- **Status Code**: `429 Too Many Requests`
- **Response**: `{"detail": "Rate limit exceeded: 10 per minute"}`

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong",
  "type": "ErrorType"
}
```

### Common Error Codes

- `400 Bad Request` - Invalid request format or parameters
- `404 Not Found` - Endpoint not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

## Interactive Documentation

### Swagger UI

Visit `http://localhost:8000/docs` for interactive API documentation with:
- Try-it-out functionality
- Request/response examples
- Schema definitions

### ReDoc

Visit `http://localhost:8000/redoc` for alternative documentation format with:
- Clean, readable layout
- Detailed schema information
- Search functionality

## OpenAPI Schema

The API provides an OpenAPI 3.0 schema at:
- `http://localhost:8000/openapi.json`

This can be used with API clients, testing tools, and documentation generators.

## Examples

### Python Client

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# List emotions
response = requests.get("http://localhost:8000/emotions")
emotions = response.json()["emotions"]
print(f"Available emotions: {emotions}")

# Generate music
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "intent": {
            "emotional_intent": "grief hidden as love"
        }
    }
)
result = response.json()
print(f"Generated: {result['result']['midi_path']}")
```

### JavaScript/TypeScript Client

```typescript
// Health check
const health = await fetch('http://localhost:8000/health')
  .then(r => r.json());

// Generate music
const result = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    intent: {
      emotional_intent: 'grief hidden as love'
    }
  })
}).then(r => r.json());
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List emotions
curl http://localhost:8000/emotions

# Generate music
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"intent": {"emotional_intent": "calm"}}'

# Interrogate
curl -X POST http://localhost:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{"message": "I want something peaceful"}'
```

## Monitoring

### Health Checks

Use `/health`, `/ready`, and `/live` endpoints for:
- Load balancer health checks
- Kubernetes liveness/readiness probes
- Monitoring system integration

### Metrics

Enable metrics endpoint with `ENABLE_METRICS=true` for:
- Prometheus scraping
- Performance monitoring
- Request tracking

## Production Considerations

1. **HTTPS**: Use reverse proxy (nginx, Traefik) for HTTPS
2. **Authentication**: Implement API key or OAuth2 authentication
3. **Rate Limiting**: Adjust limits based on usage patterns
4. **CORS**: Configure `CORS_ORIGINS` for production domains
5. **Logging**: Set `LOG_LEVEL=INFO` or `WARNING` for production
6. **Monitoring**: Set up alerting based on health check endpoints

## Support

For issues or questions:
- Check logs: `sudo journalctl -u kmidi-api -f` (Linux)
- Review environment configuration: `docs/ENVIRONMENT_CONFIGURATION.md`
- Check deployment guide: `docs/DEPLOYMENT_GUIDE.md`
