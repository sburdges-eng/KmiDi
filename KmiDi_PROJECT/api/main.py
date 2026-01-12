"""
Production FastAPI Service for KmiDi Music Generation.

Provides REST API endpoints for:
- Music generation from emotional intent
- Emotion listing and interrogation
- Health checks and monitoring
"""

import sys
import os
import time
import uuid
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file (if available)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import music_brain API
try:
    from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS
    from music_brain.structure.comprehensive_engine import TherapySession

    MUSIC_BRAIN_AVAILABLE = True
except ImportError:
    MUSIC_BRAIN_AVAILABLE = False
    TherapySession = None  # type: ignore
    EMOTIONAL_PRESETS = {}

# Configure logging from environment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not FASTAPI_AVAILABLE:
    raise ImportError(
        "FastAPI dependencies not installed. " "Install with: pip install fastapi uvicorn slowapi"
    )

# Initialize FastAPI app
app = FastAPI(
    title="KmiDi Music Generation API",
    version="1.0.0",
    description="Production API for emotion-driven music generation",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (configure from environment)
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = cors_origins_env.split(",") if cors_origins_env != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware (configure for production)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["api.kmidi.com", "localhost"]
# )


# Request/Response Models
class TechnicalIntent(BaseModel):
    key: Optional[str] = None
    bpm: Optional[int] = None
    progression: Optional[list[str]] = None
    genre: Optional[str] = None


class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: str
    technical: Optional[TechnicalIntent] = None


class GenerateRequest(BaseModel):
    intent: EmotionalIntent
    output_format: Optional[str] = "midi"


class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    services: Dict[str, Any]
    system: Optional[Dict[str, Any]] = None


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring and load balancers."""
    # Check service availability
    services_healthy = {
        "music_brain": {"available": MUSIC_BRAIN_AVAILABLE, "version": "1.0.0"},
        "api": True,
    }

    # Determine overall status
    overall_status = "healthy" if all(services_healthy.values()) else "degraded"

    # Add system metrics if available
    system_metrics = None
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        system_metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
        }
    except ImportError:
        pass
    except Exception:
        pass  # psutil available but failed, skip metrics

    return {
        "status": overall_status,
        "version": "1.0.0",
        "timestamp": time.time(),
        "services": services_healthy,
        "system": system_metrics,
    }


# Request tracking and metrics
class RequestMetrics:
    """Track API request metrics for monitoring."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0.0
        self.endpoint_counts = {}
        self.error_counts = {}
        self.start_time = time.time()

    def record_request(self, endpoint: str, duration: float, status_code: int = 200):
        """Record a request."""
        self.request_count += 1
        self.total_duration += duration
        self.endpoint_counts[endpoint] = self.endpoint_counts.get(endpoint, 0) + 1
        if status_code >= 400:
            self.error_count += 1
            error_type = f"{status_code // 100}xx"
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self.start_time
        avg_duration = self.total_duration / self.request_count if self.request_count > 0 else 0.0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0.0

        return {
            "kmidi_api_requests_total": self.request_count,
            "kmidi_api_errors_total": self.error_count,
            "kmidi_api_request_duration_seconds_sum": self.total_duration,
            "kmidi_api_request_duration_seconds_avg": avg_duration,
            "kmidi_api_error_rate": error_rate,
            "kmidi_api_uptime_seconds": uptime,
            "kmidi_api_requests_per_second": self.request_count / uptime if uptime > 0 else 0.0,
            "kmidi_api_endpoint_counts": self.endpoint_counts,
            "kmidi_api_error_counts": self.error_counts,
        }


# Global metrics instance
metrics = RequestMetrics()


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Add to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Request timing middleware
@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    endpoint = f"{request.method} {request.url.path}"

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        metrics.record_request(endpoint, duration, response.status_code)
        return response
    except Exception:
        duration = time.time() - start_time
        metrics.record_request(endpoint, duration, 500)
        raise


# Metrics endpoint (Prometheus-compatible)
@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    if not os.getenv("ENABLE_METRICS", "true").lower() == "true":
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")

    # Get current metrics
    metrics_data = metrics.get_metrics()

    # Format as Prometheus text format
    prometheus_lines = []
    for key, value in metrics_data.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (endpoint_counts, error_counts)
            for sub_key, sub_value in value.items():
                prometheus_lines.append(f'{key}{{endpoint="{sub_key}"}} {sub_value}')
        elif isinstance(value, (int, float)):
            prometheus_lines.append(f"{key} {value}")

    # Add system metrics if available
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        prometheus_lines.append(f"kmidi_api_system_cpu_percent {cpu_percent}")
        prometheus_lines.append(f"kmidi_api_system_memory_percent {memory.percent}")
        prometheus_lines.append(f"kmidi_api_system_memory_available_bytes {memory.available}")
    except ImportError:
        pass

    return "\n".join(prometheus_lines) + "\n"


# Readiness probe
@app.get("/ready")
async def ready():
    """Readiness probe - checks if service is ready to accept traffic."""
    if not MUSIC_BRAIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service not ready: Music Brain unavailable")

    return {"status": "ready", "timestamp": time.time()}


# Liveness probe
@app.get("/live")
async def live():
    """Liveness probe - checks if service is alive.

    Used by Kubernetes/Docker health checks to determine if the service
    process is still running. Should always return 200 if the process is alive.
    """
    return {
        "status": "alive",
        "timestamp": time.time(),
        "uptime_seconds": time.time() - metrics.start_time,
    }


# List emotions endpoint
@app.get("/emotions")
@limiter.limit("100/minute")
async def list_emotions(request: Request):
    """List available emotional presets."""
    try:
        if not MUSIC_BRAIN_AVAILABLE:
            raise HTTPException(status_code=503, detail="Music brain service unavailable")

        emotions = sorted(EMOTIONAL_PRESETS.keys()) if EMOTIONAL_PRESETS else []
        return {"emotions": emotions, "count": len(emotions)}
    except Exception as e:
        logger.exception("Failed to list emotions")
        raise HTTPException(status_code=500, detail=str(e))


# Generate music endpoint
@app.post("/generate")
@limiter.limit("10/minute")
async def generate_music(request: Request, generate_request: GenerateRequest):
    """Generate music from emotional intent.

    Request body:
    - intent: EmotionalIntent with emotional_intent (required) and optional technical parameters
    - output_format: "midi" (default) or "audio"

    Returns:
    - status: "success" or "error"
    - result: Generated music data (MIDI path, chords, metadata)
    - request_id: Request ID for tracking
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"[{request_id}] Generate request: " f"{generate_request.intent.emotional_intent[:50]}..."
    )

    try:
        if not MUSIC_BRAIN_AVAILABLE:
            raise HTTPException(status_code=503, detail="Music brain service unavailable")

        # Map request to therapy session pipeline
        intent = generate_request.intent
        chaos = 0.5
        motivation = 7

        if intent.technical and intent.technical.bpm:
            # Use bpm as proxy for motivation
            motivation = max(1, min(10, int(intent.technical.bpm / 20)))

        # Generate music using therapy session (supports chaos and motivation)
        start_time = time.time()
        if TherapySession is None:
            raise HTTPException(
                status_code=503,
                detail="TherapySession not available - Music Brain service unavailable",
            )
        session = TherapySession()
        affect = session.process_core_input(intent.emotional_intent)
        session.set_scales(motivation, chaos)
        plan = session.generate_plan()

        # Convert plan to result format compatible with MusicBrain output
        result = {
            "affect": {
                "primary": affect,
                "secondary": (
                    session.state.affect_result.secondary if session.state.affect_result else None
                ),
                "intensity": (
                    session.state.affect_result.intensity if session.state.affect_result else 0.0
                ),
            },
            "plan": {
                "root_note": plan.root_note,
                "mode": plan.mode,
                "tempo_bpm": plan.tempo_bpm,
                "length_bars": plan.length_bars,
                "chord_symbols": plan.chord_symbols,
                "complexity": plan.complexity,
            },
        }
        generation_time = time.time() - start_time

        logger.info(f"[{request_id}] Generation completed in {generation_time:.2f}s")

        return {
            "status": "success",
            "result": result,
            "output_format": generate_request.output_format,
            "request_id": request_id,
            "generation_time_seconds": generation_time,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Music generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Interrogate endpoint
@app.post("/interrogate")
@limiter.limit("30/minute")
async def interrogate(request: Request, interrogate_request: InterrogateRequest):
    """Interrogate emotional intent (conversational interface).

    This endpoint provides a conversational interface for refining emotional intent.
    It can maintain session context across multiple messages.

    Request body:
    - message: User's message/question
    - session_id: Optional session ID for maintaining context
    - context: Optional context dictionary

    Returns:
    - status: "success"
    - reply: AI-generated response
    - session_id: Session ID (new or existing)
    - suggestions: Suggested next steps or clarifications
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"[{request_id}] Interrogate request: {interrogate_request.message[:50]}...")

    try:
        if not MUSIC_BRAIN_AVAILABLE:
            raise HTTPException(status_code=503, detail="Music brain service unavailable")

        # Generate session ID if not provided
        session_id = interrogate_request.session_id or str(uuid.uuid4())

        # Placeholder: echo back with tip (would use AI in production)
        reply = (
            f"Noted: {interrogate_request.message}. "
            f"Consider clarifying the desired mood or groove."
        )

        # Generate suggestions based on message
        suggestions = []
        message_lower = interrogate_request.message.lower()
        if "grief" in message_lower or "sad" in message_lower:
            suggestions.append(
                "Try: 'I want to express grief through a slow, minor key progression'"
            )
        elif "happy" in message_lower or "joy" in message_lower:
            suggestions.append(
                "Try: 'I want to express joy through an upbeat, major key progression'"
            )
        else:
            suggestions.append("Consider specifying: emotion, tempo, key, or genre")

        return {
            "status": "success",
            "reply": reply,
            "session_id": session_id,
            "suggestions": suggestions,
            "request_id": request_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Interrogation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interrogation failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with request ID."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(f"[{request_id}] HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"[{request_id}] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "request_id": request_id,
        },
    )


if __name__ == "__main__":
    # Get configuration from environment variables
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    api_workers = int(os.getenv("API_WORKERS", "4"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "api.main:app",
        host=api_host,
        port=api_port,
        workers=api_workers if not reload else 1,  # Workers incompatible with reload
        reload=reload,
        log_level=log_level.lower(),
    )
