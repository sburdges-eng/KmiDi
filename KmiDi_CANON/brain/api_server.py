"""
KmiDi Brain HTTP API server — port 8000.

Serves the endpoints consumed by the body (useMusicBrain.ts):
  GET /emotions, POST /generate, POST /interrogate,
  GET/PUT /config/humanizer, POST /spectocloud/render, POST/GET /lyrics.

Run from repo root: python -m KmiDi_CANON.brain.api_server
Or: uvicorn KmiDi_CANON.brain.api_server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root and brain dir are on path so music_brain and KmiDi_CANON resolve
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BRAIN_DIR = Path(__file__).resolve().parent
for p in (str(_REPO_ROOT), str(_BRAIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

app = FastAPI(title="KmiDi Brain API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy init so we don't pull the whole brain on import
_api: Any = None
_capability_router: Any = None


def _get_api():
    global _api
    if _api is None:
        from KmiDi_CANON.brain.music_brain.api import api as _api_impl
        _api = _api_impl
    return _api


def _get_capability_router():
    """Lazy init coexist router when KMI_DI_MODEL_MODE is shadow|jepa."""
    global _capability_router
    if _capability_router is None:
        import os
        mode = os.environ.get("KMI_DI_MODEL_MODE", "current").lower().strip()
        if mode in ("shadow", "jepa"):
            try:
                from KmiDi_CANON.brain.ml import create_coexist_router
                router = create_coexist_router()
                _capability_router = router
            except ImportError:
                _capability_router = False  # Mark as tried
        else:
            _capability_router = False
    return _capability_router if _capability_router not in (None, False) else None


# ---------- Request/response models (match body contracts) ----------

class GenerateRequest(BaseModel):
    intent: Dict[str, Any]
    output_format: Optional[str] = None


class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Any] = None


class SpectocloudRenderRequest(BaseModel):
    midi_events: Optional[List[Dict[str, Any]]] = None
    midi_file_path: Optional[str] = None
    duration: Optional[float] = None
    emotion_trajectory: Optional[List[Dict[str, Any]]] = None
    mode: Optional[str] = "static"
    frame_idx: Optional[int] = None
    output_path: Optional[str] = None
    fps: Optional[int] = 24
    rotate: Optional[bool] = False
    anchor_density: Optional[str] = None
    n_particles: Optional[int] = None

    class Config:
        extra = "allow"


# ---------- Endpoints ----------

@app.get("/emotions")
def get_emotions() -> List[str]:
    """Return list of available emotion names for UI."""
    try:
        from music_brain.session.intent_schema import RULE_BREAKING_EFFECTS
        emotions = set()
        for r in RULE_BREAKING_EFFECTS.values():
            for e in r.get("example_emotions", []):
                emotions.add(e)
        return sorted(emotions) if emotions else ["grief", "hope", "rage", "anxiety", "peace", "tenderness"]
    except Exception:
        return ["grief", "hope", "rage", "anxiety", "peace", "tenderness"]


@app.post("/generate")
def generate(request: GenerateRequest) -> Dict[str, Any]:
    """Generate from emotional intent. Builds minimal CompleteSongIntent and runs process_song_intent."""
    api = _get_api()
    try:
        from music_brain.session.intent_schema import CompleteSongIntent
        intent_dict = request.intent
        tech = intent_dict.get("technical") or {}
        bpm = tech.get("bpm", 82)
        flat = {
            "mood_primary": intent_dict.get("emotional_intent", ""),
            "narrative_arc": intent_dict.get("emotional_intent", ""),
            "technical_key": tech.get("key", "F"),
            "technical_mode": tech.get("mode", "major"),
            "technical_tempo_range": (bpm, bpm),
            "technical_genre": tech.get("genre", ""),
        }
        intent = CompleteSongIntent.from_flat(**flat)
        result = api.process_song_intent(intent)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interrogate")
def interrogate(request: InterrogateRequest) -> Dict[str, Any]:
    """
    Interrogate session using quick_interrogate.
    
    [Deferred] Full conversational Q&A requires LLM reasoning engine integration.
    Current: uses interrogator questions or fallback prompt.
    """
    try:
        from music_brain.session.interrogator import quick_interrogate
        questions = quick_interrogate(request.message)
        answer = questions[0] if questions else "Tell me more about the feeling you want."
        return {"answer": answer, "questions": questions, "session_id": request.session_id}
    except Exception as e:
        return {"answer": f"Tell me more about the sound you want.", "session_id": request.session_id}


@app.get("/config/humanizer")
def get_humanizer_config() -> Dict[str, Any]:
    """Return humanizer config (default_style, ppq, bpm, analysis)."""
    api = _get_api()
    try:
        presets = api.list_humanization_presets()
        default = presets[0] if presets else "default"
        info = api.get_humanization_preset_info(default) if presets else {}
        return {
            "default_style": default,
            "ppq": info.get("ppq", 480),
            "bpm": info.get("bpm", 82),
            "analysis": {
                "flam_threshold_ms": 30.0,
                "buzz_threshold_ms": 15.0,
                "drag_threshold_ms": 25.0,
                "alternation_window_ms": 80.0,
            },
        }
    except Exception:
        return {
            "default_style": "default",
            "ppq": 480,
            "bpm": 82,
            "analysis": {
                "flam_threshold_ms": 30.0,
                "buzz_threshold_ms": 15.0,
                "drag_threshold_ms": 25.0,
                "alternation_window_ms": 80.0,
            },
        }


@app.put("/config/humanizer")
def update_humanizer_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update humanizer config.
    
    [Deferred] Persisting humanizer config requires session/preferences storage.
    Current: returns current config (updates not persisted).
    """
    # TODO: Wire to music_brain.session or preferences store when available
    return get_humanizer_config()


@app.post("/spectocloud/render")
def spectocloud_render(payload: SpectocloudRenderRequest) -> Dict[str, Any]:
    """Render spectocloud from MIDI events or file. Spine CONTRACTS §5b."""
    api = _get_api()
    body = payload.model_dump(exclude_none=True)
    result = api.render_spectocloud(body)
    # Coexist shadow: run specto_understanding when mode=shadow
    router = _get_capability_router()
    if router and result.get("status") == "completed":
        try:
            _ = router.run(
                "specto_understanding",
                {"midi_events": body.get("midi_events"), "midi_file_path": body.get("midi_file_path"), "result": result},
            )
        except Exception:
            pass
    return result


@app.post("/lyrics")
def set_lyrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set user lyrics or generate fragments using lyrical_mirror.
    
    Accepts: {"lyrics": str, "source": "user"|"generated", "therapy_text": optional}
    If therapy_text provided, generates fragments via music_brain.text.lyrical_mirror.
    """
    try:
        lyrics_text = payload.get("lyrics", "")
        source = payload.get("source", "user")
        therapy_text = payload.get("therapy_text")
        
        if therapy_text and not lyrics_text:
            # Generate fragments from therapy text
            from music_brain.text.lyrical_mirror import generate_lyrical_fragments
            fragments = generate_lyrical_fragments(session_text=therapy_text, max_lines=6)
            lyrics_text = "\n".join(fragments)
            source = "generated"
        
        lines = len(lyrics_text.splitlines()) if lyrics_text else 0
        word_count = len(lyrics_text.split()) if lyrics_text else 0
        
        # TODO: Persist to session storage when available
        return {"status": "ok", "source": source, "lines": lines, "word_count": word_count}
    except Exception as e:
        return {"status": "error", "details": str(e), "source": "user", "lines": 0, "word_count": 0}


@app.get("/lyrics")
def get_lyrics() -> Dict[str, Any]:
    """
    Get lyrics state.
    
    [Deferred] Retrieval requires session storage integration.
    Current: returns empty state (lyrics not persisted across requests).
    """
    # TODO: Retrieve from session storage when available
    return {"lyrics": "", "source": "user", "generated": None}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
