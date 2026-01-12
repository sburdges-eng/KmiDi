# KmiDi Music Generation API Reference

This document provides a comprehensive reference for the KmiDi Music Generation API, built with FastAPI. The API allows for emotion-driven music generation, emotional intent interrogation, and health monitoring.

## Table of Contents
1.  [API Endpoints](#1-api-endpoints)
    *   [1.1 GET /health](#11-get-health)
    *   [1.2 GET /ready](#12-get-ready)
    *   [1.3 GET /live](#13-get-live)
    *   [1.4 GET /metrics](#14-get-metrics)
    *   [1.5 GET /emotions](#15-get-emotions)
    *   [1.6 POST /generate](#16-post-generate)
    *   [1.7 POST /interrogate](#17-post-interrogate)
2.  [Request/Response Models](#2-requestresponse-models)
    *   [2.1 EmotionalIntent](#21-emotionalintent)
    *   [2.2 TechnicalIntent](#22-technicalintent)
    *   [2.3 GenerateRequest](#23-generaterequest)
    *   [2.4 InterrogateRequest](#24-interrogaterequest)
    *   [2.5 HealthResponse](#25-healthresponse)
3.  [Error Handling](#3-error-handling)
4.  [OpenAPI Documentation (Swagger UI)](#4-openapi-documentation-swagger-ui)

---

## 1. API Endpoints

### 1.1 GET /health

**Description**: Health check endpoint for monitoring and load balancers.

**Response Model**: `HealthResponse`

**Example Response (200 OK)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1768136765.215679,
  "services": {
    "music_brain": {"available": true, "version": "1.0.0"},
    "api": true
  },
  "system": {
    "cpu_percent": 15.5,
    "memory_percent": 60.2,
    "memory_available_mb": 4096.0
  }
}
```

### 1.2 GET /ready

**Description**: Readiness probe - checks if the service is ready to accept traffic.

**Response Model**: `{"status": "ready", "timestamp": float}`

**Example Response (200 OK)**:
```json
{
  "status": "ready",
  "timestamp": 1768136765.215679
}
```

### 1.3 GET /live

**Description**: Liveness probe - checks if the service is alive. Used by Kubernetes/Docker health checks to determine if the service process is still running. Should always return 200 if the process is alive.

**Response Model**: `{"status": "alive", "timestamp": float, "uptime_seconds": float}`

**Example Response (200 OK)**:
```json
{
  "status": "alive",
  "timestamp": 1768136765.215679,
  "uptime_seconds": 3600.5
}
```

### 1.4 GET /metrics

**Description**: Prometheus-compatible metrics endpoint. Returns application and system metrics in Prometheus text format. Requires `ENABLE_METRICS` environment variable to be `true`.

**Example Response (200 OK)**:
```
kmidi_api_requests_total 1234
kmidi_api_errors_total 10
kmidi_api_request_duration_seconds_sum 5.67
kmidi_api_request_duration_seconds_avg 0.0045
kmidi_api_error_rate 0.008
kmidi_api_uptime_seconds 3600
kmidi_api_requests_per_second 0.34
kmidi_api_endpoint_counts{endpoint="GET /health"} 500
kmidi_api_endpoint_counts{endpoint="POST /generate"} 200
kmidi_api_error_counts{error="5xx"} 10
kmidi_api_system_cpu_percent 12.3
kmidi_api_system_memory_percent 75.1
kmidi_api_system_memory_available_bytes 1073741824
```

### 1.5 GET /emotions

**Description**: Lists all available emotional presets that the KmiDi system can process.

**Response Model**: `{"emotions": list[str], "count": int}`

**Example Request (curl)**:
```bash
curl http://127.0.0.1:8000/emotions
```

**Example Response (200 OK)**:
```json
{
  "emotions": [
    "anger",
    "anxiety",
    "calm",
    "grief",
    "nostalgia",
    "tension_building"
  ],
  "count": 6
}
```

### 1.6 POST /generate

**Description**: Generates music based on a provided emotional intent.

**Request Model**: `GenerateRequest`

**Example Request (curl)**:
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "I feel peaceful and calm, like a quiet morning",
      "technical": {
        "key": "C",
        "bpm": 90,
        "progression": ["I", "V", "vi", "IV"],
        "genre": "indie"
      }
    },
    "output_format": "midi"
  }'
```

**Example Response (200 OK)**:
```json
{
  "status": "success",
  "result": {
    "affect": {
      "primary": "peaceful",
      "secondary": "serene",
      "intensity": 0.8
    },
    "plan": {
      "root_note": "C",
      "mode": "major",
      "tempo_bpm": 90,
      "length_bars": 16,
      "chord_symbols": ["C", "G", "Am", "F"],
      "complexity": 0.5
    }
  },
  "output_format": "midi",
  "request_id": "abc12345",
  "generation_time_seconds": 0.45
}
```

### 1.7 POST /interrogate

**Description**: Provides a conversational interface for refining emotional intent. It can maintain session context across multiple messages.

**Request Model**: `InterrogateRequest`

**Example Request (curl)**:
```bash
curl -X POST http://127.0.0.1:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Make it feel more grounded",
    "session_id": "optional-session-id",
    "context": {}
  }'
```

**Example Response (200 OK)**:
```json
{
  "status": "success",
  "reply": "Noted: Make it feel more grounded. Consider clarifying the desired mood or groove.",
  "session_id": "optional-session-id",
  "suggestions": [
    "Consider specifying: emotion, tempo, key, or genre"
  ],
  "request_id": "f460aec0"
}
```

---

## 2. Request/Response Models

### 2.1 EmotionalIntent

```python
class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: str
    technical: Optional[TechnicalIntent] = None
```

### 2.2 TechnicalIntent

```python
class TechnicalIntent(BaseModel):
    key: Optional[str] = None
    bpm: Optional[int] = None
    progression: Optional[list[str]] = None
    genre: Optional[str] = None
```

### 2.3 GenerateRequest

```python
class GenerateRequest(BaseModel):
    intent: EmotionalIntent
    output_format: Optional[str] = "midi"
```

### 2.4 InterrogateRequest

```python
class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
```

### 2.5 HealthResponse

```python
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    services: Dict[str, Any]
    system: Optional[Dict[str, Any]] = None
```

---

## 3. Error Handling

API endpoints are designed with comprehensive error handling:

*   **HTTP Exceptions**: Handled by `@app.exception_handler(HTTPException)`, returning a JSON response with `detail`, `status_code`, and `request_id`.
*   **Global Exceptions**: Unhandled exceptions are caught by `@app.exception_handler(Exception)`, providing a generic "Internal server error" message with error type and `request_id`.
*   **Rate Limiting**: Implemented with `slowapi` to prevent abuse, returning `429 Too Many Requests` when limits are exceeded.

All error responses include a `request_id` for tracing and debugging purposes.

---

## 4. OpenAPI Documentation (Swagger UI)

Interactive API documentation is automatically generated by FastAPI and available at the following URLs when the API server is running:

*   **Swagger UI**: `http://127.0.0.1:8000/docs`
*   **ReDoc UI**: `http://127.0.0.1:8000/redoc`

These interfaces allow you to explore all available endpoints, their request/response schemas, and even test them directly from your browser.
