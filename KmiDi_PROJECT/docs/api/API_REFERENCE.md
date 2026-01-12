# KmiDi API Reference

> Complete API documentation for the KmiDi Music Generation REST API

**Version**: 1.0.0
**Base URL**: `http://localhost:8000`
**OpenAPI Spec**: `http://localhost:8000/openapi.json`

## Quick Links

- **Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Health & Monitoring](#health--monitoring)
4. [Emotions API](#emotions-api)
5. [Generation API](#generation-api)
6. [Interrogation API](#interrogation-api)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [SDKs & Examples](#sdks--examples)

---

## Getting Started

### Starting the API Server

```bash
# Development mode
cd KmiDi_PROJECT
python -m uvicorn api.main:app --reload

# Production mode
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using Docker
docker run -d -p 8000:8000 kmidi-api:prod
```

### First Request

```bash
# Verify the API is running
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1704729600.0,
  "services": {"music_brain": true, "api": true}
}
```

---

## Authentication

Currently, the API does not require authentication for public endpoints.

### Planned Authentication (Future)

```bash
# API Key authentication
curl -H "X-API-Key: your_api_key" http://localhost:8000/generate
```

---

## Health & Monitoring

### GET /health

Full health check with service status and system metrics.

**Rate Limit**: None

**Response Schema**:

```typescript
interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  timestamp: number;
  services: {
    music_brain: boolean;
    api: boolean;
  };
  system?: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_mb: number;
  };
}
```

**Example**:

```bash
curl http://localhost:8000/health
```

**Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1704729600.123,
  "services": {
    "music_brain": true,
    "api": true
  },
  "system": {
    "cpu_percent": 15.5,
    "memory_percent": 45.2,
    "memory_available_mb": 4096.0
  }
}
```

---

### GET /ready

Kubernetes-compatible readiness probe.

**Use Case**: Load balancer health checks, Kubernetes readiness gates.

**Response**:

```json
{
  "status": "ready",
  "timestamp": 1704729600.0
}
```

**Status Codes**:
- `200` - Service ready to accept traffic
- `503` - Service not ready (Music Brain unavailable)

---

### GET /live

Kubernetes-compatible liveness probe.

**Use Case**: Detect hung processes, Kubernetes liveness checks.

**Response**:

```json
{
  "status": "alive",
  "timestamp": 1704729600.0,
  "uptime_seconds": 3600.5
}
```

**Status Codes**:
- `200` - Process is alive

---

### GET /metrics

Prometheus-compatible metrics endpoint.

**Configuration**: Set `ENABLE_METRICS=true` (default: true)

**Response Format**: Prometheus text exposition format

```
kmidi_api_requests_total 1234
kmidi_api_errors_total 12
kmidi_api_request_duration_seconds_sum 45.6
kmidi_api_request_duration_seconds_avg 0.037
kmidi_api_error_rate 0.0097
kmidi_api_uptime_seconds 3600.0
kmidi_api_requests_per_second 0.34
```

---

## Emotions API

### GET /emotions

List all available emotional presets.

**Rate Limit**: 100 requests/minute

**Response Schema**:

```typescript
interface EmotionsResponse {
  emotions: string[];
  count: number;
}
```

**Example**:

```bash
curl http://localhost:8000/emotions
```

**Response**:

```json
{
  "emotions": [
    "calm",
    "grief",
    "joy",
    "anger",
    "nostalgia",
    "hope"
  ],
  "count": 6
}
```

---

## Generation API

### POST /generate

Generate music from emotional intent.

**Rate Limit**: 10 requests/minute

**Request Schema**:

```typescript
interface GenerateRequest {
  intent: {
    core_wound?: string;      // Optional: underlying emotional source
    core_desire?: string;     // Optional: what the music should achieve
    emotional_intent: string; // Required: primary emotional expression
    technical?: {
      key?: string;           // e.g., "C", "F#", "Bb"
      bpm?: number;           // 40-200
      progression?: string[]; // e.g., ["F", "C", "Am", "Dm"]
      genre?: string;         // e.g., "acoustic", "electronic"
    };
  };
  output_format?: "midi" | "audio";  // Default: "midi"
}
```

**Response Schema**:

```typescript
interface GenerateResponse {
  status: "success" | "error";
  result: {
    affect: {
      primary: string;      // Detected primary emotion
      secondary?: string;   // Secondary emotion (if detected)
      intensity: number;    // 0.0 - 1.0
    };
    plan: {
      root_note: string;    // e.g., "C"
      mode: string;         // e.g., "aeolian", "dorian"
      tempo_bpm: number;
      length_bars: number;
      chord_symbols: string[];
      complexity: number;   // 0.0 - 1.0
    };
  };
  output_format: string;
  request_id: string;
  generation_time_seconds: number;
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "grief hidden as love",
      "technical": {
        "key": "F",
        "bpm": 72
      }
    }
  }'
```

**Response**:

```json
{
  "status": "success",
  "result": {
    "affect": {
      "primary": "grief",
      "secondary": "tenderness",
      "intensity": 0.67
    },
    "plan": {
      "root_note": "C",
      "mode": "aeolian",
      "tempo_bpm": 70,
      "length_bars": 32,
      "chord_symbols": ["Cm", "Ab", "Fm", "Cm"],
      "complexity": 0.5
    }
  },
  "output_format": "midi",
  "request_id": "a1b2c3d4",
  "generation_time_seconds": 0.023
}
```

---

## Interrogation API

### POST /interrogate

Conversational interface for refining musical intent.

**Rate Limit**: 30 requests/minute

**Request Schema**:

```typescript
interface InterrogateRequest {
  message: string;           // User's message
  session_id?: string;       // Optional: maintain conversation context
  context?: {
    [key: string]: any;      // Additional context
  };
}
```

**Response Schema**:

```typescript
interface InterrogateResponse {
  status: "success";
  reply: string;             // AI response
  session_id: string;        // Session ID for follow-up
  suggestions: string[];     // Suggested next steps
  request_id: string;
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to express grief through music"
  }'
```

**Response**:

```json
{
  "status": "success",
  "reply": "Noted: I want to express grief through music. Consider clarifying the desired mood or groove.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "suggestions": [
    "Try: 'I want to express grief through a slow, minor key progression'"
  ],
  "request_id": "e5f6g7h8"
}
```

---

## Error Handling

All errors follow a consistent format:

```typescript
interface ErrorResponse {
  detail: string;
  status_code: number;
  request_id: string;
  type?: string;
}
```

### Error Codes

| Code | Description | Example |
|------|-------------|---------|
| `400` | Bad Request | Invalid JSON, missing required fields |
| `404` | Not Found | Endpoint doesn't exist |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error |
| `503` | Service Unavailable | Music Brain not loaded |

**Example Error**:

```json
{
  "detail": "Rate limit exceeded: 10 per minute",
  "status_code": 429,
  "request_id": "abc123"
}
```

---

## Rate Limiting

| Endpoint | Limit |
|----------|-------|
| `/emotions` | 100/minute |
| `/generate` | 10/minute |
| `/interrogate` | 30/minute |
| `/health`, `/ready`, `/live` | Unlimited |

Rate limits are per IP address. Headers returned:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1704730260
```

---

## SDKs & Examples

### Python

```python
import requests

class KmiDiClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def health(self):
        return requests.get(f"{self.base_url}/health").json()

    def emotions(self):
        return requests.get(f"{self.base_url}/emotions").json()

    def generate(self, emotional_intent, **kwargs):
        payload = {
            "intent": {
                "emotional_intent": emotional_intent,
                **kwargs
            }
        }
        return requests.post(f"{self.base_url}/generate", json=payload).json()

    def interrogate(self, message, session_id=None):
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        return requests.post(f"{self.base_url}/interrogate", json=payload).json()

# Usage
client = KmiDiClient()
result = client.generate("hopeful after loss")
print(f"Generated: {result['result']['plan']['chord_symbols']}")
```

### JavaScript/TypeScript

```typescript
class KmiDiClient {
  constructor(private baseUrl = 'http://localhost:8000') {}

  async health() {
    return fetch(`${this.baseUrl}/health`).then(r => r.json());
  }

  async generate(emotionalIntent: string, options?: object) {
    return fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        intent: { emotional_intent: emotionalIntent, ...options }
      })
    }).then(r => r.json());
  }
}

// Usage
const client = new KmiDiClient();
const result = await client.generate('calm and peaceful');
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# List emotions
curl http://localhost:8000/emotions

# Generate music
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"intent": {"emotional_intent": "grief"}}'

# Interrogate
curl -X POST http://localhost:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{"message": "I want something peaceful"}'
```

---

## OpenAPI Specification

The full OpenAPI 3.0 specification is available at:

- **JSON**: `http://localhost:8000/openapi.json`
- **Interactive**: `http://localhost:8000/docs`

You can use this spec to:
- Generate client SDKs (using OpenAPI Generator)
- Import into Postman/Insomnia
- Create API documentation
- Set up API testing

---

## See Also

- [Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md)
- [CLI Usage Guide](../cli/CLI_GUIDE.md)
- [GUI Manual](../gui/GUI_MANUAL.md)
- [Quick Start](../QUICKSTART_GUIDE.md)

---

*Last Updated: 2026-01-11*
