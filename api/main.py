"""
Production FastAPI Service for KmiDi Music Generation.

Provides REST API endpoints for:
- Music generation from emotional intent
- Emotion listing and interrogation
- Health checks and monitoring
"""

import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
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
    from music_brain.emotion_api import MusicBrain
    from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS
    from music_brain.structure.comprehensive_engine import TherapySession
    MUSIC_BRAIN_AVAILABLE = True
except ImportError:
    MUSIC_BRAIN_AVAILABLE = False
    MusicBrain = None
    TherapySession = None
    EMOTIONAL_PRESETS = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not FASTAPI_AVAILABLE:
    raise ImportError("FastAPI dependencies not installed. Install with: pip install fastapi uvicorn slowapi")

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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
    services: Dict[str, bool]


# Dependency for rate limiting
def get_rate_limit_key(request: Request) -> str:
    return get_remote_address(request)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "music_brain": MUSIC_BRAIN_AVAILABLE,
            "api": True,
        }
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
    """Generate music from emotional intent."""
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
        
        # Generate music using MusicBrain API
        brain = MusicBrain(use_neural=False)
        result = brain.generate_from_text(intent.emotional_intent)
        
        return {
            "status": "success",
            "result": result,
            "output_format": generate_request.output_format,
        }
    except Exception as e:
        logger.exception("Music generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Interrogate endpoint
@app.post("/interrogate")
@limiter.limit("30/minute")
async def interrogate(request: Request, interrogate_request: InterrogateRequest):
    """Interrogate emotional intent (conversational interface)."""
    try:
        if not MUSIC_BRAIN_AVAILABLE:
            raise HTTPException(status_code=503, detail="Music brain service unavailable")
        
        # Placeholder: echo back with tip
        reply = f"Noted: {interrogate_request.message}. Consider clarifying the desired mood or groove."
        
        return {
            "status": "success",
            "reply": reply,
            "session_id": interrogate_request.session_id,
        }
    except Exception as e:
        logger.exception("Interrogation failed")
        raise HTTPException(status_code=500, detail=f"Interrogation failed: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

