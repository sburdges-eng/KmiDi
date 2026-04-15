"""
DAiW API Wrapper - Clean interface for desktop app and future REST API.

This module provides a simplified, consistent API surface for all music_brain
functionality, making it easier to integrate with desktop apps, web services,
or other interfaces.
"""
from typing import Annotated, Dict, List, Optional, Any, Tuple
import os
import sys
import logging
import asyncio
import time

try:
    from fastapi import APIRouter, Body, FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel, Field, ValidationError
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False
from pathlib import Path  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
from music_brain.utils.serialization import pydantic_to_dict  # noqa: E402
from music_brain.utils.error_handlers import api_error_handler  # noqa: E402
from music_brain.utils.io_utils import load_json_config, save_json_config  # noqa: E402

# Constants for input validation

# Core imports
from music_brain.audio import (  # noqa: E402
    AudioAnalyzer,
    render_midi_to_audio,
)
try:
    from music_brain.harmony_utils import (
        HarmonyGenerator,
        HarmonyResult,
    )
    from music_brain.harmony_utils.harmony_generator import (
        generate_midi_from_harmony,
    )
except ImportError:  # pragma: no cover - compatibility shim for partial installs
    class HarmonyResult(dict):
        """Fallback harmony result container."""

    class HarmonyGenerator:
        """Fallback harmony generator used when harmony package is partial."""

        def generate_from_intent(self, intent):
            return HarmonyResult()

        def generate_basic_progression(self, key="C", mode="major", style="pop"):
            return HarmonyResult()

    def generate_midi_from_harmony(*args, **kwargs):
        return None
try:
    from music_brain.groove import (
        extract_groove,
        apply_groove,
        GrooveTemplate,
        humanize_midi_file,
        GrooveSettings,
        settings_from_preset,
        list_presets,
        get_preset,
    )
except ImportError:  # pragma: no cover - compatibility shim for partial installs
    class GrooveTemplate(dict):
        pass

    class GrooveSettings(dict):
        pass

    def extract_groove(*args, **kwargs):
        return GrooveTemplate()

    def apply_groove(*args, **kwargs):
        return None

    def humanize_midi_file(*args, **kwargs):
        return None

    def settings_from_preset(*args, **kwargs):
        return GrooveSettings()

    def list_presets(*args, **kwargs):
        return []

    def get_preset(*args, **kwargs):
        return {}
from music_brain.structure import (  # noqa: E402
    analyze_chords,
    detect_sections,
)
from music_brain.structure.progression import (  # noqa: E402
    diagnose_progression,
    generate_reharmonizations,
)
from music_brain.structure.comprehensive_engine import (  # noqa: E402
    TherapySession,
    render_plan_to_midi,
    HarmonyPlan,
)
from music_brain.session.intent_schema import (  # noqa: E402
    CompleteSongIntent,
    suggest_rule_break,
    validate_intent,
    list_all_rules,
)
from music_brain.session.intent_processor import process_intent  # noqa: E402
from music_brain.engine_api.schema import CompleteSongIntentRequest  # noqa: E402
from music_brain.api_schemas.generate_v1 import (  # noqa: E402
    EmotionalIntent,
    GENERATE_REQUEST_OPENAPI_EXAMPLES,
    GenerateRequest,
    InterrogateRequest,
    LyricsRequest,
    TechnicalIntent,
)
from music_brain.pipeline import IntentPipeline  # noqa: E402
try:
    from music_brain.data_utils.emotional_mapping import EMOTIONAL_PRESETS
except ImportError:  # pragma: no cover - compatibility shim for partial installs
    EMOTIONAL_PRESETS = {}
from music_brain.voice import (  # noqa: E402
    AutoTuneProcessor,
    get_auto_tune_preset,
    VoiceModulator,
    get_modulation_preset,
    VoiceSynthesizer,
    get_voice_profile,
    VoiceClassifier,
)
try:
    from music_brain.groove_kmidi.drum_humanizer import DrumHumanizer
except ImportError:  # pragma: no cover - compatibility shim for partial installs
    class DrumHumanizer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass


class _DummyAudioAnalyzer:
    """Fallback audio analyzer stub for environments without full deps."""

    def detect_bpm(self, samples, sample_rate):
        return 120.0, {}

    def detect_key(self, samples, sample_rate):
        return "C", "major"

    def analyze_audio(self, samples, sample_rate):
        return {"bpm": 120.0, "key": "C", "mode": "major"}


class DAiWAPI:
    """
    Unified API wrapper for DAiW functionality.

    Provides a clean, consistent interface for all music_brain operations,
    making it easier to integrate with desktop apps, web services, or CLI tools.
    """

    def __init__(self):
        self.harmony_generator = HarmonyGenerator()
        self.auto_tune_processor = AutoTuneProcessor()
        self.voice_modulator = VoiceModulator()
        self.voice_synthesizer = VoiceSynthesizer()
        self.audio_analyzer = _DummyAudioAnalyzer()
        self.drum_humanizer = self._build_humanizer()

    def _build_humanizer(self) -> DrumHumanizer:
        """Create DrumHumanizer, pulling config from config/humanizer.json if present."""
        cfg_path = Path("config/humanizer.json")
        if cfg_path.exists():
            try:
                return DrumHumanizer(config_path=str(cfg_path))
            except Exception:
                logging.exception("Failed to load humanizer config; using defaults.")
        return DrumHumanizer()

    def reload_humanizer(self) -> None:
        """Reload humanizer configuration from disk."""
        self.drum_humanizer = self._build_humanizer()

    # ========== Harmony Generation ==========

    def generate_harmony_from_intent(
        self,
        intent: CompleteSongIntent,
        output_midi: Optional[str] = None,
        tempo_bpm: int = 82
    ) -> Dict[str, Any]:
        """
        Generate harmony from a CompleteSongIntent.

        Args:
            intent: CompleteSongIntent object
            output_midi: Optional path to save MIDI file
            tempo_bpm: Tempo for MIDI output

        Returns:
            Dict with harmony result and optional MIDI path
        """
        harmony = self.harmony_generator.generate_from_intent(intent)

        result = {
            "harmony": {
                "chords": harmony.chords,
                "key": harmony.key,
                "mode": harmony.mode,
                "rule_break_applied": harmony.rule_break_applied,
                "emotional_justification": harmony.emotional_justification,
            },
            "voicings": [
                {
                    "root": v.root,
                    "notes": v.notes,
                    "duration_beats": v.duration_beats,
                    "roman_numeral": v.roman_numeral,
                }
                for v in harmony.voicings
            ],
        }

        if output_midi:
            generate_midi_from_harmony(harmony, output_midi, tempo_bpm=tempo_bpm)
            result["midi_path"] = output_midi

        return result

    def generate_basic_progression(
        self,
        key: str = "C",
        mode: str = "major",
        pattern: str = "I-V-vi-IV",
        output_midi: Optional[str] = None,
        tempo_bpm: int = 82
    ) -> Dict[str, Any]:
        """
        Generate a basic chord progression.

        Args:
            key: Musical key (e.g., "C", "F", "Bb")
            mode: Mode (major, minor, dorian, etc.)
            pattern: Roman numeral pattern (e.g., "I-V-vi-IV")
            output_midi: Optional path to save MIDI file
            tempo_bpm: Tempo for MIDI output

        Returns:
            Dict with harmony result
        """
        harmony = self.harmony_generator.generate_basic_progression(
            key=key,
            mode=mode,
            pattern=pattern
        )

        result = {
            "harmony": {
                "chords": harmony.chords,
                "key": harmony.key,
                "mode": harmony.mode,
                "rule_break_applied": harmony.rule_break_applied,
                "emotional_justification": harmony.emotional_justification,
            },
        }

        if output_midi:
            generate_midi_from_harmony(harmony, output_midi, tempo_bpm=tempo_bpm)
            result["midi_path"] = output_midi

        return result

    # ========== Groove Operations ==========

    def extract_groove_from_midi(
        self,
        midi_path: str
    ) -> Dict[str, Any]:
        """
        Extract groove pattern from a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dict with groove analysis data
        """
        groove = extract_groove(midi_path)
        return groove.to_dict()

    def apply_groove_to_midi(
        self,
        midi_path: str,
        genre: str = "funk",
        intensity: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Apply a genre groove template to a MIDI file.

        Args:
            midi_path: Path to input MIDI file
            genre: Genre template (funk, jazz, rock, etc.)
            intensity: Groove intensity (0.0-1.0)
            output_path: Optional output path (auto-generated if None)

        Returns:
            Path to output MIDI file
        """
        if output_path is None:
            output_path = str(Path(midi_path).with_suffix('.grooved.mid'))

        apply_groove(midi_path, genre=genre, output=output_path, intensity=intensity)
        return output_path

    def humanize_drums(
        self,
        midi_path: str,
        complexity: float = 0.5,
        vulnerability: float = 0.5,
        preset: Optional[str] = None,
        drum_channel: int = 9,
        enable_ghost_notes: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply humanization to drum track in MIDI file.

        Args:
            midi_path: Path to input MIDI file
            complexity: Timing chaos (0.0-1.0)
            vulnerability: Dynamic fragility (0.0-1.0)
            preset: Optional preset name (overrides complexity/vulnerability)
            drum_channel: MIDI channel for drums (default 9 = channel 10)
            enable_ghost_notes: Whether to add ghost notes
            output_path: Optional output path

        Returns:
            Dict with result info and output path
        """
        if preset:
            settings = settings_from_preset(preset)
            complexity = settings.complexity
            vulnerability = settings.vulnerability
        else:
            settings = GrooveSettings(
                complexity=complexity,
                vulnerability=vulnerability,
                enable_ghost_notes=enable_ghost_notes
            )

        if output_path is None:
            output_path = str(Path(midi_path).with_suffix('.humanized.mid'))

        result_path = humanize_midi_file(
            input_path=midi_path,
            output_path=output_path,
            complexity=complexity,
            vulnerability=vulnerability,
            drum_channel=drum_channel,
            settings=settings,
        )

        return {
            "output_path": result_path,
            "complexity": complexity,
            "vulnerability": vulnerability,
            "preset_used": preset,
        }

    # ========== Chord Analysis ==========

    def analyze_midi_chords(
        self,
        midi_path: str,
        include_sections: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze chord progression in a MIDI file.

        Args:
            midi_path: Path to MIDI file
            include_sections: Whether to also detect sections

        Returns:
            Dict with chord analysis and optional sections
        """
        progression = analyze_chords(midi_path)

        result = {
            "key": progression.key,
            "chords": progression.chords,
            "roman_numerals": progression.roman_numerals,
            "borrowed_chords": progression.borrowed_chords,
        }

        if include_sections:
            sections = detect_sections(midi_path)
            result["sections"] = [
                {
                    "name": s.name,
                    "start_bar": s.start_bar,
                    "end_bar": s.end_bar,
                    "energy": s.energy,
                }
                for s in sections
            ]

        return result

    def diagnose_progression(
        self,
        progression: str
    ) -> Dict[str, Any]:
        """
        Diagnose issues in a chord progression string.

        Args:
            progression: Chord progression (e.g., "F-C-Am-Dm")

        Returns:
            Dict with diagnosis results
        """
        return diagnose_progression(progression)

    def suggest_reharmonizations(
        self,
        progression: str,
        style: str = "jazz",
        count: int = 3
    ) -> List[Dict[str, str]]:
        """
        Generate reharmonization suggestions.

        Args:
            progression: Chord progression string
            style: Reharmonization style (jazz, pop, rnb, etc.)
            count: Number of suggestions

        Returns:
            List of reharmonization suggestions
        """
        return generate_reharmonizations(progression, style=style, count=count)

    # ========== Audio Analysis ==========

    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file, returning tempo, key, spectrum, and chords.
        """
        analyzer = getattr(self, "audio_analyzer", AudioAnalyzer())
        if hasattr(analyzer, "analyze_file"):
            result = analyzer.analyze_file(audio_path)
            return result.to_dict() if hasattr(result, "to_dict") else result
        return {"bpm": 0.0, "key": "C"}

    def analyze_audio_waveform(self, samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        analyzer = getattr(self, "audio_analyzer", AudioAnalyzer(sample_rate=sample_rate))
        result = analyzer.analyze_audio(
            samples, sample_rate) if hasattr(
            analyzer, "analyze_audio") else analyzer.analyze_waveform(
            samples, sample_rate)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def detect_audio_bpm(self, samples: np.ndarray, sample_rate: int) -> float:
        analyzer = getattr(self, "audio_analyzer", AudioAnalyzer(sample_rate=sample_rate))
        result = analyzer.detect_bpm(samples, sample_rate)
        if isinstance(result, tuple):
            return result[0]
        return float(result) if result is not None else 0.0

    def detect_audio_key(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, str]:
        analyzer = getattr(self, "audio_analyzer", AudioAnalyzer(sample_rate=sample_rate))
        result = analyzer.detect_key(samples, sample_rate)
        if isinstance(result, tuple):
            return result
        return (str(result), "")

    # ========== Voice Processing ==========

    def auto_tune_vocals(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preset: str = "transparent",
        key: Optional[str] = None,
        mode: str = "major",
    ) -> str:
        settings = get_auto_tune_preset(preset)
        processor = AutoTuneProcessor(settings)
        return processor.process_file(input_path, output_path, key, mode)

    def modulate_voice(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preset: str = "intimate_whisper",
    ) -> str:
        settings = get_modulation_preset(preset)
        modulator = VoiceModulator(settings)
        return modulator.process_file(input_path, output_path)

    def synthesize_voice(
        self,
        lyrics: str,
        melody_midi: List[int],
        tempo_bpm: int = 82,
        output_path: str = "guide_vocal.wav",
        profile: str = "guide_vulnerable",
    ) -> str:
        config = get_voice_profile(profile)
        synthesizer = VoiceSynthesizer(config)
        return synthesizer.synthesize_guide(
            lyrics=lyrics,
            melody_midi=melody_midi,
            tempo_bpm=tempo_bpm,
            output_path=output_path,
        )

    def speak_text_prompt(
        self,
        text: str,
        output_path: str = "spoken_prompt.wav",
        profile: str = "guide_confident",
        tempo_bpm: int = 80,
    ) -> str:
        config = get_voice_profile(profile)
        synthesizer = VoiceSynthesizer(config)
        return synthesizer.speak_text(
            text=text,
            output_path=output_path,
            profile=profile,
            tempo_bpm=tempo_bpm,
        )

    def classify_voice_file(
        self,
        audio_path: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        classifier = VoiceClassifier()
        return classifier.classify(audio_path, top_k=top_k)

    # ========== Therapy Session ==========

    def therapy_session(
        self,
        text: str,
        motivation: int = 7,
        chaos_tolerance: float = 0.5,
        output_midi: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process emotional text through therapy session and generate MIDI.

        Args:
            text: Emotional text input
            motivation: Motivation level (1-10)
            chaos_tolerance: Chaos tolerance (0.0-1.0)
            output_midi: Optional path to save MIDI file

        Returns:
            Dict with analysis and plan, plus optional MIDI path
        """
        session = TherapySession()
        affect = session.process_core_input(text)
        session.set_scales(motivation, chaos_tolerance)
        plan = session.generate_plan()

        result = {
            "affect": {
                "primary": affect,
                "secondary": (
                    session.state.affect_result.secondary
                    if session.state.affect_result else None
                ),
                "intensity": (
                    session.state.affect_result.intensity
                    if session.state.affect_result else 0.0
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

        if output_midi:
            midi_path = render_plan_to_midi(plan, output_midi)
            result["midi_path"] = midi_path

        return result

    # ========== Intent Processing ==========

    def process_song_intent(
        self,
        intent: CompleteSongIntent,
        output_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a CompleteSongIntent and generate all musical elements.

        Args:
            intent: CompleteSongIntent object
            output_json: Optional path to save results as JSON

        Returns:
            Dict with all generated elements
        """
        result = process_intent(intent)

        # Convert to serializable format with safe dict access
        # Provide defaults for all keys in case process_intent returns incomplete data
        output = {
            "intent_summary": result.get('intent_summary', {}),
        }

        # Safely extract harmony data
        harmony = result.get('harmony')
        if harmony:
            output["harmony"] = {
                "chords": getattr(harmony, 'chords', []),
                "roman_numerals": getattr(harmony, 'roman_numerals', []),
                "rule_broken": getattr(harmony, 'rule_broken', ""),
                "rule_effect": getattr(harmony, 'rule_effect', ""),
            }
        else:
            output["harmony"] = {
                "chords": [],
                "roman_numerals": [],
                "rule_broken": "",
                "rule_effect": "",
            }

        # Safely extract groove data
        groove = result.get('groove')
        if groove:
            output["groove"] = {
                "pattern_name": getattr(groove, 'pattern_name', ""),
                "tempo_bpm": getattr(groove, 'tempo_bpm', 120),
                "swing_factor": getattr(groove, 'swing_factor', 0.0),
                "rule_broken": getattr(groove, 'rule_broken', ""),
                "rule_effect": getattr(groove, 'rule_effect', ""),
            }
        else:
            output["groove"] = {
                "pattern_name": "",
                "tempo_bpm": 120,
                "swing_factor": 0.0,
                "rule_broken": "",
                "rule_effect": "",
            }

        # Safely extract arrangement data
        arrangement = result.get('arrangement')
        if arrangement:
            output["arrangement"] = {
                "sections": getattr(arrangement, 'sections', []),
                "dynamic_arc": getattr(arrangement, 'dynamic_arc', []),
                "rule_broken": getattr(arrangement, 'rule_broken', ""),
            }
        else:
            output["arrangement"] = {
                "sections": [],
                "dynamic_arc": [],
                "rule_broken": "",
            }

        # Safely extract production data
        production = result.get('production')
        if production:
            output["production"] = {
                "vocal_treatment": getattr(production, 'vocal_treatment', ""),
                "eq_notes": getattr(production, 'eq_notes', ""),
                "dynamics_notes": getattr(production, 'dynamics_notes', ""),
                "rule_broken": getattr(production, 'rule_broken', ""),
            }
        else:
            output["production"] = {
                "vocal_treatment": "",
                "eq_notes": "",
                "dynamics_notes": "",
                "rule_broken": "",
            }

        if output_json:
            save_json_config(Path(output_json), output)

        return output

    def suggest_rule_breaks(
        self,
        emotion: str
    ) -> List[Dict[str, str]]:
        """
        Get rule-breaking suggestions for an emotion.

        Args:
            emotion: Target emotion (e.g., "grief", "anger")

        Returns:
            List of rule-breaking suggestions
        """
        return suggest_rule_break(emotion)

    def list_available_rules(self) -> Dict[str, List[str]]:
        """
        List all available rule-breaking options.

        Returns:
            Dict mapping categories to lists of rules
        """
        return list_all_rules()

    def validate_song_intent(
        self,
        intent: CompleteSongIntent
    ) -> List[str]:
        """
        Validate a CompleteSongIntent.

        Args:
            intent: CompleteSongIntent to validate

        Returns:
            List of validation issues (empty if valid)
        """
        return validate_intent(intent)

    # ========== Preset Management ==========

    def list_humanization_presets(self) -> List[str]:
        """List available humanization presets."""
        return list_presets()

    def get_humanization_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get information about a humanization preset."""
        return get_preset(preset_name)

    def _convert_request_to_complete_intent(self, request: Any) -> CompleteSongIntent:
        """
        Convert UI request payload to CompleteSongIntent.

        Uses the deterministic 3-stage IntentPipeline (normalize → validate → expand).
        """
        return IntentPipeline().run(request)


# ========== Global Session State (stateless DAiWAPI — Phase 4) ==========
_SESSION_STATE: Dict[str, Any] = {
    "user_lyrics": None,
    "user_lyrics_source": "none",
    "last_generated_lyrics": None,
}


def set_lyrics(lyrics: str, source: str = "user") -> Dict[str, Any]:
    """Persist user-provided lyrics (or clear when empty) and return a lightweight summary."""
    cleaned = (lyrics or "").strip()
    if not cleaned:
        _SESSION_STATE["user_lyrics"] = None
        _SESSION_STATE["user_lyrics_source"] = "none"
        return {
            "status": "cleared",
            "source": _SESSION_STATE["user_lyrics_source"],
            "lines": 0,
            "word_count": 0,
        }

    _SESSION_STATE["user_lyrics"] = cleaned
    _SESSION_STATE["user_lyrics_source"] = source or "user"
    _SESSION_STATE["last_generated_lyrics"] = None

    return {
        "status": "stored",
        "source": _SESSION_STATE["user_lyrics_source"],
        "lines": len(cleaned.splitlines()),
        "word_count": len(cleaned.split()),
        "preview": cleaned[:140],
    }


def get_lyrics() -> Dict[str, Any]:
    """Return the active lyric payload and provenance."""
    return {
        "lyrics": _SESSION_STATE["user_lyrics"],
        "source": _SESSION_STATE["user_lyrics_source"],
        "generated": _SESSION_STATE["last_generated_lyrics"],
    }


def intent_field(intent: Any, name: str, default: str = "") -> str:
    """Safely pull a field from either a dict-like intent or pydantic model."""
    if intent is None:
        return default
    if isinstance(intent, dict):
        return str(intent.get(name, default) or default)
    return str(getattr(intent, name, default) or default)


def generate_structured_lyrics(intent: Any) -> str:
    """Lightweight, dependency-free fallback lyric generator."""
    wound = intent_field(intent, "core_wound", "unspecified wound")
    desire = intent_field(intent, "core_desire", "longing")
    emotion = intent_field(intent, "emotional_intent", "unspecified emotion")

    verse1 = [
        f"I carry {wound} in the seams of the day",
        f"Tracing {emotion} shadows that won't fade away",
        f"Still I keep moving with {desire} in my hands",
        "Hoping the light remembers where I stand",
    ]
    chorus = [
        f"Hold me when the night gets loud with {emotion}",
        f"Sing back the truth, I'm more than {wound}",
        f"Step into the dawn, let {desire} bloom",
        "Every note a bridge across the room",
    ]
    verse2 = [
        "I tune my breathing to the softest drum",
        "Let the melody confess what I've become",
        "If I am fading, let the chorus know",
        "I was a spark before the river froze",
    ]

    generated = "\n".join(
        ["[Verse 1]"] + verse1
        + ["", "[Chorus]"] + chorus
        + ["", "[Verse 2]"] + verse2
        + ["", "[Chorus]", *chorus]
    )
    _SESSION_STATE["last_generated_lyrics"] = generated
    return generated


def select_lyric_payload(intent: Any) -> Tuple[str, str]:
    """Decide which text drives generation; user lyrics > generated > intent."""
    if _SESSION_STATE["user_lyrics"]:
        return _SESSION_STATE["user_lyrics"], _SESSION_STATE["user_lyrics_source"]
    if intent:
        generated = generate_structured_lyrics(intent)
        return generated, "generated"
    return "emotional intent", "intent"


# Convenience instance
api = DAiWAPI()

__all__ = [
    "DAiWAPI",
    "api",
    "app",
    "EmotionalIntent",
    "TechnicalIntent",
    "GenerateRequest",
    "InterrogateRequest",
    "LyricsRequest",
]


# ---------- Minimal HTTP API (FastAPI) ----------
# This provides the server that `python -m music_brain.api` is expected to start.

app = None  # overwritten below when FastAPI is available

if FASTAPI_AVAILABLE:
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    _static_dir = Path(__file__).parent / "static"
    _HTTP_500_DETAIL = "Internal server error."

    app = FastAPI(
        title="KmiDi Sound Engine",
        version="0.2.0",
        description=(
            "Your creative command center — compose, feel, and shape "
            "music through intent.  Every endpoint maps to a musical action. "
            "PRD-aligned TTG v1: POST /v1/generate (intent_schema_version, timeline, "
            "orchestration)."
        ),
    )

    if _static_dir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(_static_dir)),
            name="static",
        )

    # Add CORS middleware to allow requests from frontend.
    # Override with KMIDI_CORS_ORIGINS env var (comma-separated) in production.
    _default_cors_origins = [
        "http://localhost:1420",   # Vite dev server
        "http://127.0.0.1:1420",  # Vite dev server (alternative)
        "tauri://localhost",       # Tauri app
        "http://localhost:5173",   # Alternative Vite port
    ]
    _cors_env = os.environ.get("KMIDI_CORS_ORIGINS", "")
    _cors_origins = (
        [o.strip() for o in _cors_env.split(",") if o.strip()]
        if _cors_env
        else _default_cors_origins
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """KmiDi Sound Engine — musician-friendly landing page."""
        docs_html = _static_dir / "docs.html"
        if docs_html.is_file():
            return HTMLResponse(content=docs_html.read_text())
        return HTMLResponse(
            content="<h1>KmiDi Sound Engine</h1><p>Visit <a href='/docs'>/docs</a></p>"
        )

    @app.get("/health", summary="Pulse Check",
             description="Quick heartbeat — is the sound engine running?")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    # Sandbox for /audio/ serving: only files under this directory are allowed (prevents path traversal).
    _audio_serve_root = Path(
        os.environ.get("KMIDI_AUDIO_SERVE_ROOT", str(tempfile.gettempdir()))
    ).resolve()

    def _resolve_audio_path_sandbox(audio_path: str) -> Path:
        """Resolve audio_path and ensure it is under _audio_serve_root. Raises HTTPException 400 if outside."""
        resolved = Path(audio_path).resolve()
        try:
            resolved.relative_to(_audio_serve_root)
        except ValueError:
            logging.warning("Classify audio path outside sandbox rejected: %s", audio_path)
            raise HTTPException(
                status_code=400,
                detail="Audio path is not allowed.",
            ) from None
        return resolved

    @app.get("/audio/{file_path:path}", summary="Stream Audio",
             description="Stream any generated audio file — WAV, MP3, FLAC, OGG.")
    async def serve_audio(file_path: str):
        """Stream generated audio files for playback. Paths are restricted to KMIDI_AUDIO_SERVE_ROOT (default: temp)."""
        import urllib.parse
        decoded_path = urllib.parse.unquote(file_path)
        audio_file = Path(decoded_path).resolve()

        # Reject path traversal: file must be under the allowed root.
        try:
            audio_file.relative_to(_audio_serve_root)
        except ValueError:
            logging.warning(f"Audio path outside sandbox rejected: {decoded_path}")
            raise HTTPException(
                status_code=404,
                detail="Audio file path is not allowed.",
            ) from None

        logging.info(f"Serving audio file: {decoded_path}")
        logging.info(
            f"File exists: {audio_file.exists()}, "
            f"is_file: {audio_file.is_file() if audio_file.exists() else 'N/A'}"
        )

        if not audio_file.exists():
            logging.error(f"Audio file not found: {decoded_path}")
            # Try to find alternative paths (maybe MIDI file exists but audio doesn't)
            if decoded_path.endswith(('.wav', '.mp3', '.ogg')):
                midi_path = decoded_path.rsplit('.', 1)[0] + '.mid'
                if Path(midi_path).exists():
                    logging.warning(
                        f"Audio file not found, but MIDI exists: {midi_path}. "
                        "Audio conversion may be needed."
                    )
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Audio file not found: {decoded_path}. "
                    "The file may not have been generated yet or may have been cleaned up."
                )
            )

        if not audio_file.is_file():
            logging.error(f"Path is not a file: {decoded_path}")
            raise HTTPException(status_code=400, detail=f"Path is not a file: {decoded_path}")

        # Determine media type based on file extension
        media_type = "application/octet-stream"  # Default fallback
        ext = audio_file.suffix.lower()
        if ext == ".wav":
            media_type = "audio/wav"  # or "audio/x-wav" for some browsers
        elif ext == ".mp3":
            media_type = "audio/mpeg"
        elif ext == ".ogg" or ext == ".oga":
            media_type = "audio/ogg"
        elif ext == ".m4a":
            media_type = "audio/mp4"
        elif ext == ".flac":
            media_type = "audio/flac"
        elif ext == ".aac":
            media_type = "audio/aac"

        logging.info(f"Serving {decoded_path} as {media_type}")

        return FileResponse(
            path=str(audio_file),
            media_type=media_type,
            filename=audio_file.name,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(audio_file.stat().st_size)
            }
        )

    @app.get("/emotions", summary="Read the Room",
             description="Get the list of emotions the AI currently understands — "
                         "from tender intimacy to fierce energy.")
    @api_error_handler(log_message="Failed to list emotions", detail=_HTTP_500_DETAIL)
    async def list_emotions():
        return sorted(EMOTIONAL_PRESETS.keys())

    def _normalize_humanizer_config(data: Dict[str, Any]) -> Dict[str, Any]:
        default_analysis = {
            "flam_threshold_ms": 30.0,
            "buzz_threshold_ms": 50.0,
            "drag_threshold_ms": 80.0,
            "alternation_window_ms": 200.0,
        }
        default_config = {
            "default_style": "standard",
            "ppq": 480,
            "bpm": 120.0,
            "analysis": default_analysis,
        }
        merged = {**default_config, **(data or {})}
        merged["analysis"] = {**default_analysis, **merged.get("analysis", {})}
        return merged

    def _parse_midi_file(path: Path) -> Tuple[List[Dict[str, Any]], float]:
        """Parse a MIDI file into event dicts; requires optional mido dependency."""
        try:
            import mido  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail="mido is required to parse MIDI files; install with pip install mido",
            ) from exc

        if not path.exists():
            raise HTTPException(status_code=400, detail=f"MIDI file not found: {path}")
        try:
            mid = mido.MidiFile(str(path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Failed to read MIDI file.") from exc

        tempo = 500000  # default 120 BPM
        events: List[Dict[str, Any]] = []
        current_time = 0.0
        for msg in mid:
            current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type in {"note_on", "note_off"}:
                events.append(
                    {
                        "time": current_time,
                        "type": msg.type,
                        "note": getattr(msg, "note", None),
                        "velocity": getattr(msg, "velocity", 0),
                        "channel": getattr(msg, "channel", 0),
                    }
                )
        return events, current_time

    @app.get("/config/humanizer", summary="Feel Settings",
             description="Check the humanizer — timing drift, velocity curves, "
                         "and micro-groove that make MIDI feel like a real player.")
    async def humanizer_config():
        """
        Return current humanizer/analysis config.
        - Loads `config/humanizer.json` if present; otherwise defaults.
        - Also exposes analysis thresholds (flam/buzz/drag/alternation).
        """
        cfg = load_json_config(
            Path("config/humanizer.json"),
            _normalize_humanizer_config({}),
        )
        return _normalize_humanizer_config(cfg)

    SPECTO_PRESETS: Dict[str, Dict[str, Any]] = {
        "preview": {"anchor_density": "sparse", "n_particles": 600, "fps": 8},
        "standard": {"anchor_density": "normal", "n_particles": 1200, "fps": 15},
        "high": {"anchor_density": "dense", "n_particles": 1800, "fps": 24},
    }

    @app.get("/spectocloud/presets", summary="Visual Presets",
             description="Browse Spectocloud particle presets — colors, shapes, "
                         "and motion styles that paint your sound.")
    async def spectocloud_presets():
        """List Spectocloud rendering presets (anchor density, particle count, fps)."""
        return SPECTO_PRESETS

    @app.put("/config/humanizer", summary="Adjust Feel",
             description="Dial in the human feel — timing drift, velocity curves, "
                         "micro-groove. Like tweaking a compressor, but for soul.")
    async def update_humanizer_config(payload: Dict[str, Any]):
        """
        Persist humanizer/analysis configuration.
        - Accepts fields: default_style, ppq, bpm,
          analysis.{flam_threshold_ms,buzz_threshold_ms,drag_threshold_ms,
          alternation_window_ms}
        - Writes to config/humanizer.json and returns the normalized config.
        """
        cfg_dir = Path("config")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        normalized = _normalize_humanizer_config(payload)
        cfg_path = cfg_dir / "humanizer.json"
        
        success = save_json_config(cfg_path, normalized)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save humanizer config")
        try:
            api.reload_humanizer()
        except Exception:
            logging.exception("Failed to reload humanizer after config update")
        return normalized

    @app.post("/config/humanizer/reload", summary="Recall Feel Preset",
              description="Reset the humanizer to its saved state — like hitting "
                          "'recall' on a mixer channel strip.")
    @api_error_handler(log_message="Failed to reload humanizer", detail=_HTTP_500_DETAIL)
    async def reload_humanizer():
        """Force reload of the in-memory humanizer/analyzer from config/humanizer.json."""
        api.reload_humanizer()
        return {"status": "ok"}

    class SpectocloudRenderRequest(BaseModel):
        midi_events: Optional[List[Dict[str, Any]]] = None
        midi_file_path: Optional[str] = None
        audio_file_path: Optional[str] = None  # MP3 or WAV file path
        duration: Optional[float] = None
        emotion_trajectory: Optional[List[Dict[str, Any]]] = None
        mode: str = "static"  # "static" or "animation"
        frame_idx: int = 0
        output_path: Optional[str] = None
        fps: int = 15
        rotate: bool = True
        anchor_density: str = "normal"
        n_particles: int = 1200

    @app.post("/spectocloud/render", summary="Paint Your Sound",
              description="Render a Spectocloud visualization — see your music "
                          "as a living particle cloud of light and color.")
    @api_error_handler(log_message="spectocloud render failed", detail=_HTTP_500_DETAIL)
    async def render_spectocloud(payload: SpectocloudRenderRequest):
        """
        Render Spectocloud output (static frame or animation).
        - For static: mode="static", frame_idx sets which frame to render.
        - For animation: mode="animation", fps/rotate control output.
        """
        try:
            from music_brain.visualization.spectocloud import Spectocloud  # Lazy import
        except Exception:  # pragma: no cover
            logging.exception("Failed to import Spectocloud")
            raise HTTPException(status_code=500, detail=_HTTP_500_DETAIL)

            events: Optional[List[Dict[str, Any]]] = payload.midi_events
            duration = payload.duration

            # Handle audio file input (convert to MIDI events if needed)
            if payload.audio_file_path:
                audio_path = Path(payload.audio_file_path)
                if not audio_path.exists():
                    raise HTTPException(
                        status_code=400, detail=f"Audio file not found: {payload.audio_file_path}")
                # For now, if audio file provided, we'd need to extract MIDI from it
                # This is a placeholder - actual implementation would analyze audio and extract MIDI
                # For now, raise an error suggesting MIDI file instead
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Audio file analysis not yet implemented. "
                        "Please provide midi_file_path or midi_events instead."
                    )
                )

            if payload.midi_file_path:
                safe_midi_path = _resolve_audio_path_sandbox(payload.midi_file_path)
                parsed_events, parsed_duration = _parse_midi_file(safe_midi_path)
                events = parsed_events
                duration = duration or parsed_duration

            if not events:
                raise HTTPException(
                    status_code=400,
                    detail="provide audio_file_path, midi_file_path, or midi_events")
            if duration is None or duration <= 0:
                # try to infer from events time
                max_time = max((e.get("time", 0) or 0) for e in events)
                if max_time > 0:
                    duration = max_time
                else:
                    raise HTTPException(status_code=400, detail="duration must be > 0")
            if payload.n_particles <= 0:
                raise HTTPException(status_code=400, detail="n_particles must be > 0")
            if payload.fps <= 0:
                raise HTTPException(status_code=400, detail="fps must be > 0")

            specto = Spectocloud(
                anchor_density=payload.anchor_density,
                n_particles=payload.n_particles,
            )
            specto.process_midi(
                midi_events=events,
                duration=duration,
                emotion_trajectory=payload.emotion_trajectory,
            )
            if not specto.frames:
                raise HTTPException(
                    status_code=400, detail="No frames generated; check duration/window_size")
            mode = payload.mode.lower()
            if mode not in {"static", "animation"}:
                raise HTTPException(status_code=400, detail="mode must be 'static' or 'animation'")

            if mode == "static":
                if payload.frame_idx < 0:
                    raise HTTPException(status_code=400, detail="frame_idx must be >= 0")
                if payload.output_path:
                    out_path = str(_resolve_audio_path_sandbox(payload.output_path))
                else:
                    out_path = str(
                        Path(tempfile.gettempdir()) / "spectocloud_frame.png")
                specto.render_static_frame(
                    frame_idx=min(payload.frame_idx, max(0, len(specto.frames) - 1)),
                    output_path=out_path,
                    show=False,
                    use_textured=False,
                )
                return {
                    "status": "success",
                    "mode": "static",
                    "output_path": out_path,
                    "frames": len(specto.frames),
                }

            if payload.output_path:
                out_path = str(_resolve_audio_path_sandbox(payload.output_path))
            else:
                out_path = str(
                    Path(tempfile.gettempdir()) / "spectocloud_anim.gif")
            specto.render_animation(
                output_path=out_path,
                fps=payload.fps,
                duration=None,
                rotate=payload.rotate,
            )
            return {
                "status": "success",
                "mode": "animation",
                "output_path": out_path,
                "frames": len(specto.frames),
            }

    def _create_output_paths(output_format: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not output_format:
            return None, None
        output_midi = None
        output_audio = None
        stamp = int(time.time())
        tmpdir = Path(tempfile.gettempdir())
        if output_format in ['mid', 'midi']:
            output_midi = str(tmpdir / f"generated_{stamp}.mid")
        elif output_format in ['wav', 'mp3']:
            output_audio = str(tmpdir / f"generated_{stamp}.{output_format}")
            output_midi = str(tmpdir / f"generated_{stamp}.mid")
        return output_midi, output_audio

    def _extract_structure_instruments(
        tech: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """Pull timeline→structure and orchestration→instruments from technical payload.

        Dedupes the two inline TTG-adapter blocks that used to appear both in
        the MIDI-plan construction and in response assembly.
        """
        if not tech:
            return None, None
        from music_brain.api_schemas.ttg_adapter import (
            orchestration_dict_to_instruments,
            ttg_dict_to_structure_list,
        )
        t_dict = pydantic_to_dict(tech)
        structure = (
            ttg_dict_to_structure_list(t_dict.get("timeline", {}))
            if t_dict.get("timeline") else None
        )
        instruments = (
            orchestration_dict_to_instruments(t_dict.get("orchestration", {}))
            if t_dict.get("orchestration") else None
        )
        return structure, instruments

    def _build_harmony_plan(
        tech: Any,
        res_output: Dict[str, Any],
        key_root: str,
        key_mode: str,
        base_bpm: int,
        structure: Optional[List[Dict[str, Any]]],
        instruments: Optional[List[Dict[str, Any]]],
    ) -> HarmonyPlan:
        """Construct HarmonyPlan from orchestrator output + technical payload.

        Handles duration/BPM clamping and structure-bar override so the
        caller doesn't have to repeat that logic inline.
        """
        duration_minutes = tech.duration if tech and tech.duration is not None else 3.0
        try:
            duration_minutes = float(duration_minutes)
            duration_minutes = max(0.1, min(60.0, duration_minutes))
        except (ValueError, TypeError):
            duration_minutes = 3.0

        length_bars = int((duration_minutes * base_bpm) / 4)
        length_bars = max(16, min(128, length_bars))

        if structure:
            total_structure_bars = sum(
                section.get("bars", 4) * section.get("repetitions", 1)
                for section in structure
            )
            if total_structure_bars > 0:
                length_bars = total_structure_bars

        harmony = res_output.get("harmony", {})
        return HarmonyPlan(
            root_note=key_root,
            mode=key_mode,
            tempo_bpm=base_bpm,
            time_signature="4/4",
            length_bars=length_bars,
            chord_symbols=harmony.get("chords", ["C", "Am", "F", "G"]),
            harmonic_rhythm="1_chord_per_bar",
            mood_profile=res_output.get("intent_summary", {}).get("mood", "neutral"),
            complexity=0.5,
            structure=structure,
            instruments=instruments,
        )

    def _build_generate_response(
        request: GenerateRequest,
        res_output: Dict[str, Any],
        tech: Any,
        lyric_text: str,
        lyric_source: str,
        structure: Optional[List[Dict[str, Any]]],
        instruments: Optional[List[Dict[str, Any]]],
        output_midi: Optional[str],
        output_audio: Optional[str],
    ) -> Dict[str, Any]:
        """Assemble the /generate response dict, including structure/instruments
        summaries and MIDI/audio file paths. Audio rendering from the generated
        MIDI happens here when output_audio was requested."""
        response: Dict[str, Any] = {
            "status": "success",
            "intent_schema_version": request.intent_schema_version,
            "result": res_output,
            "lyrics": {"source": lyric_source, "text": lyric_text},
        }
        if tech is not None and getattr(tech, "energy_curve", None) is not None:
            response["energy_curve"] = pydantic_to_dict(tech.energy_curve)

        if structure:
            response["structure"] = {
                "sections": structure,
                "total_bars": sum(
                    s.get("bars", 4) * s.get("repetitions", 1) if isinstance(s, dict) else 4
                    for s in structure
                ),
            }

        if instruments:
            response["instruments"] = {
                "tracks": [
                    {
                        "name": (
                            inst.get("name", "instrument")
                            if isinstance(inst, dict) else "instrument"
                        ),
                        "type": (
                            inst.get("type", "chord")
                            if isinstance(inst, dict) else "chord"
                        ),
                        "channel": inst.get("channel") if isinstance(inst, dict) else None,
                    }
                    for inst in instruments
                ],
            }

        midi_path = res_output.get("midi_path")
        if output_midi and midi_path:
            response["midi_path"] = midi_path
            midi_file = Path(midi_path)
            if output_audio:
                audio_path = str(midi_file.with_suffix(".wav"))
                try:
                    render_midi_to_audio(str(midi_file), audio_path)
                except Exception as render_exc:
                    logging.exception("Audio render failed from MIDI")
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"MIDI generated but audio render failed: "
                            f"{render_exc}"
                        ),
                    )
                response["audio_path"] = audio_path
                response["output_path"] = audio_path
            else:
                response["output_path"] = midi_path

        return response

    @api_error_handler(log_message="generate failed", detail=_HTTP_500_DETAIL)
    async def _handle_generate_music(request: GenerateRequest):
        # We explicitly handle TimeoutError inside.
        try:
            # Try to use full intent pipeline if we have advanced parameters
            tech = request.intent.technical
            # Default to the full intent pipeline for all requests.
            use_full_pipeline = True

            if use_full_pipeline:
                # Use full CompleteSongIntent pipeline (normalize → validate → expand)
                logging.info("Using full intent pipeline with CompleteSongIntent")

                if tech is None:
                    raise HTTPException(
                        status_code=422,
                        detail="technical payload is required for complete intent generation",
                    )

                from music_brain.orchestrator.orchestrator import AIOrchestrator, OrchestratorConfig, Pipeline
                from music_brain.orchestrator.processors.intent import IntentProcessor as OrchestratorIntentProcessor
                from music_brain.orchestrator.processors.harmony import HarmonyProcessor as OrchestratorHarmonyProcessor
                from music_brain.orchestrator.processors.groove import GrooveProcessor as OrchestratorGrooveProcessor
                
                config = OrchestratorConfig(enable_logging=True)
                orchestrator = AIOrchestrator(config)
                
                pipeline = Pipeline("http_generate")
                pipeline.add_stage("intent", OrchestratorIntentProcessor())
                pipeline.add_stage("harmony", OrchestratorHarmonyProcessor())
                pipeline.add_stage("groove", OrchestratorGrooveProcessor())
                
                key_raw = getattr(tech, "key", "C major") if tech else "C major"
                key_parts = key_raw.split(maxsplit=1)
                key_root = key_parts[0] if key_parts else "C"
                key_mode = key_parts[1] if len(key_parts) > 1 else "major"
                base_bpm = getattr(tech, "bpm", 120) if tech else 120
                if base_bpm is None:
                    base_bpm = 120
                
                input_data = {
                    "title": "Generated Song",
                    "phase_0": {
                        "core_desire": request.intent.core_desire or "",
                    },
                    "phase_1": {
                        "mood_primary": request.intent.emotional_intent or "neutral",
                        "narrative_arc": getattr(tech, "narrative_arc", "") if tech else "",
                    },
                    "phase_2": {
                        "technical_genre": getattr(tech, "genre", "") if tech else "",
                        "technical_key": key_root,
                        "technical_mode": key_mode,
                        "technical_tempo_range": [base_bpm, base_bpm],
                    }
                }
                
                pipeline_result = await orchestrator.execute(pipeline, input_data)
                if not pipeline_result.success:
                    raise HTTPException(
                        status_code=500,
                        detail=pipeline_result.error or "Pipeline execution failed",
                    )

                # Reconstruct the output dictionary expected by the endpoint
                intent_res = pipeline_result.context.get_shared("intent") or {}
                harmony = pipeline_result.context.get_shared("harmony") or {}
                groove = pipeline_result.context.get_shared("groove") or {}

                res_output: Dict[str, Any] = {
                    "intent_summary": {
                        "mood": intent_res.get("intent", {}).get("phase_1", {}).get("mood_primary", ""),
                        "rule_broken": intent_res.get("intent", {}).get("phase_2", {}).get("technical_rule_to_break", ""),
                    },
                    "harmony": {
                        "chords": harmony.get("chords", []),
                        "roman_numerals": harmony.get("roman_numerals", []),
                        "rule_broken": harmony.get("rule_broken", ""),
                        "rule_effect": harmony.get("rule_effect", ""),
                    },
                    "groove": {
                        "pattern_name": groove.get("pattern_name", ""),
                        "tempo_bpm": groove.get("tempo_bpm", base_bpm),
                        "feel": "Straight",
                    },
                    "arrangement": {"sections": []},
                    "production": {"notes": []},
                }

                output_midi, output_audio = _create_output_paths(request.output_format)
                structure, instruments = _extract_structure_instruments(tech)

                # Build harmony plan and render MIDI when output was requested.
                if output_midi and res_output.get("harmony"):
                    try:
                        plan = _build_harmony_plan(
                            tech, res_output, key_root, key_mode, base_bpm,
                            structure, instruments,
                        )
                        res_output["midi_path"] = render_plan_to_midi(plan, output_midi)
                    except Exception as midi_exc:
                        logging.exception("Failed to generate MIDI from full intent")
                        raise HTTPException(
                            status_code=500,
                            detail="MIDI generation failed",
                        ) from midi_exc

                lyric_text, lyric_source = select_lyric_payload(request.intent)
                return _build_generate_response(
                    request, res_output, tech, lyric_text, lyric_source,
                    structure, instruments, output_midi, output_audio,
                )

            # Legacy fallback retained for safety if forced externally.
            logging.info("Using legacy therapy_session fallback")
            chaos = 0.5
            motivation = 7
            if request.intent.technical and request.intent.technical.bpm:
                try:
                    # Safely convert BPM to motivation with validation
                    bpm = int(request.intent.technical.bpm)
                    bpm = max(40, min(300, bpm))  # Clamp BPM
                    motivation = max(1, min(10, int(bpm / 20)))
                except (ValueError, TypeError):
                    motivation = 7  # Default if conversion fails
            lyric_text, lyric_source = select_lyric_payload(request.intent)

            # Generate output file if format requested
            output_midi, output_audio = _create_output_paths(request.output_format)

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    api.therapy_session,
                    text=lyric_text or request.intent.emotional_intent,
                    motivation=motivation,
                    chaos_tolerance=chaos,
                    output_midi=output_midi,
                ),
                timeout=30.0,
            )

            response = {
                "status": "success",
                "intent_schema_version": request.intent_schema_version,
                "result": result,
                "lyrics": {
                    "source": lyric_source,
                    "text": lyric_text,
                },
            }

            # Add file paths to response with safe dict access
            if output_midi and result.get("midi_path"):
                response["midi_path"] = result["midi_path"]
                if output_audio:
                    response["audio_path"] = result["midi_path"].replace(
                        ".mid", f".{request.output_format}")
                    response["output_path"] = response["audio_path"]
                else:
                    response["output_path"] = result["midi_path"]

            return response
        except HTTPException:
            raise
        except asyncio.TimeoutError:
            logging.warning("generate timed out after 30s")
            raise HTTPException(
                status_code=504,
                detail="Generation timed out (30s). Try simpler parameters.",
            )

    @app.post("/generate", summary="Compose Music",
              description="Bring your musical vision to life — describe a mood, "
                          "pick instruments, set the groove, and let the engine "
                          "compose harmony, melody, and full arrangement.")
    async def generate_music(
        request: Annotated[
            GenerateRequest,
            Body(openapi_examples=GENERATE_REQUEST_OPENAPI_EXAMPLES),
        ],
    ):
        return await _handle_generate_music(request)

    v1_router = APIRouter(
        prefix="/v1",
        tags=["v1-prd"],
    )

    @v1_router.post(
        "/generate",
        summary="Compose Music (PRD v1)",
        description="Versioned generate — intent_schema_version, TTG timeline, "
                    "orchestration (docs/prd/KMIDI_PRD.md). Same handler as /generate.",
    )
    async def generate_music_v1(
        request: Annotated[
            GenerateRequest,
            Body(openapi_examples=GENERATE_REQUEST_OPENAPI_EXAMPLES),
        ],
    ):
        return await _handle_generate_music(request)

    app.include_router(v1_router)

    @app.post("/interrogate", summary="Consult the Muse",
              description="Ask the engine exactly how it interpreted your intent — "
                          "see the invisible connections it made.")
    @api_error_handler(log_message="interrogate failed", detail=_HTTP_500_DETAIL)
    async def interrogate(request: InterrogateRequest):
        """Query the current state, logic, or meaning of the session."""
        return {
            "status": "success",
            "reply": (
                f"Noted: {request.message}. "
                "Consider clarifying the desired mood or groove."
            ),
            "session_id": request.session_id,
        }

    @app.post("/lyrics", summary="Write Lyrics",
              description="Paste or write your lyrics so the engine can weave them "
                          "into the musical fabric.")
    @api_error_handler(log_message="Failed to set lyrics", detail=_HTTP_500_DETAIL)
    async def set_lyrics_endpoint(payload: LyricsRequest):
        """
        Persist user-supplied lyrics and return a summary.
        """
        return set_lyrics(payload.lyrics, source=payload.source or "user")

    @app.get("/lyrics", summary="Read Lyrics",
             description="Retrieve the current lyrics — your words, your story.")
    @api_error_handler(log_message="Failed to fetch lyrics", detail=_HTTP_500_DETAIL)
    async def get_lyrics_endpoint():
        """
        Return the current lyric payload, source, and any cached generated lyrics.
        """
        return get_lyrics()

    # ========== Audio Emotion Classification Endpoints ==========

    class AudioClassifyRequest(BaseModel):
        audio_path: str
        model_type: Optional[str] = "emotion_7"
        top_k: Optional[int] = 3

    class VoiceClassifyRequest(BaseModel):
        audio_path: str
        top_k: Optional[int] = 3

    @app.post("/audio/classify", summary="Hear the Feeling",
              description="Drop in an audio clip and discover its emotional signature — "
                          "what feelings does this sound carry?")
    @api_error_handler(log_message="audio classify failed", detail="Internal server error.")
    async def classify_audio(request: AudioClassifyRequest):
        """
        Classify emotion from audio file using trained ML models.

        Returns valence/arousal coordinates for integration with
        the emotion-to-music generation pipeline.
        """
        try:
            safe_path = _resolve_audio_path_sandbox(request.audio_path)
            from music_brain.emotion.audio_emotion_classifier import AudioEmotionClassifier

            classifier = AudioEmotionClassifier(model_type=request.model_type)
            if not classifier.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="Audio classifier model not available. Check model checkpoints."
                )

            result = classifier.classify(str(safe_path), top_k=request.top_k)
            return {
                "status": "success",
                "result": result.to_dict(),
            }
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Audio classification dependencies not installed.",
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Audio file not found.")

    @app.post("/audio/valence-arousal", summary="Mood Map",
              description="Map any sound onto the feeling grid — how positive/negative "
                          "and how energetic/calm.")
    @api_error_handler(log_message="audio valence-arousal failed", detail="Internal server error.")
    async def get_audio_valence_arousal(request: AudioClassifyRequest):
        """
        Get valence/arousal coordinates from audio.

        Primary interface for emotion-to-music mapping.
        """
        try:
            safe_path = _resolve_audio_path_sandbox(request.audio_path)
            from music_brain.emotion.audio_emotion_classifier import AudioEmotionClassifier

            classifier = AudioEmotionClassifier(model_type=request.model_type)
            if not classifier.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="Audio classifier model not available."
                )

            va = classifier.get_valence_arousal(str(safe_path))
            return {
                "status": "success",
                "valence": va["valence"],
                "arousal": va["arousal"],
                "emotion": va["emotion"],
                "confidence": va["confidence"],
            }
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="Audio classification dependencies not installed.",
            ) from exc
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Audio file not found.")

    @app.get("/audio/models", summary="Loaded AI Ears",
             description="See which AI listening models are active — emotion classifiers, "
                         "timbre analyzers, and more.")
    @api_error_handler(log_message="list audio models failed", detail=_HTTP_500_DETAIL)
    async def list_audio_models():
        """List active audio classification models."""
        models_dir = Path(__file__).parent.parent / "models" / "checkpoints"
        available = []

        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "best.pt").exists():
                    available.append({
                        "name": model_dir.name,
                        "path": str(model_dir / "best.pt"),
                    })

        return {
            "status": "success",
            "models": available,
            "supported_types": ["emotion_7", "voice_type"],
        }

    @app.post("/voice/classify", summary="Voice Range",
              description="Identify a singer's range — soprano, alto, tenor, or bass — "
                          "so the engine can write in their sweet spot.")
    @api_error_handler(log_message="voice classify failed", detail="Internal server error.")
    async def classify_voice(request: VoiceClassifyRequest):
        """Identify vocal range from audio."""
        try:
            safe_path = _resolve_audio_path_sandbox(request.audio_path)
            result = api.classify_voice_file(str(safe_path), top_k=request.top_k or 3)
            return {"status": "success", "result": result}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Audio file not found.")

    @app.get("/ai/jepa/status", summary="JEPA Models Status",
             description="Check the status of Audio-JEPA, Chord-JEPA, and "
                         "Stem-JEPA models — the AI ears that understand "
                         "musical structure and compatibility.")
    async def jepa_status():
        """Report JEPA model integration status."""
        try:
            from music_brain.jepa import (
                AudioJEPAConfig,
                ChordJEPAConfig,
            )
            audio_cfg = AudioJEPAConfig()
            chord_cfg = ChordJEPAConfig()
            return {
                "status": "available",
                "models": {
                    "audio_jepa": {
                        "tier": audio_cfg.tier,
                        "latent_dim": audio_cfg.latent_dim,
                        "n_mels": audio_cfg.n_mels,
                        "status": "ready",
                    },
                    "chord_jepa": {
                        "d_model": chord_cfg.d_model,
                        "num_layers": chord_cfg.num_layers,
                        "num_chords": chord_cfg.num_chords,
                        "status": "ready",
                    },
                    "stem_jepa": {
                        "status": "integrated",
                        "reference": "music_brain.learning.stem_compatibility",
                    },
                },
                "training": {
                    "sagemaker_config": "config/jepa_training.yaml",
                    "launcher": "scripts/launch_jepa_sagemaker.py",
                },
            }
        except ImportError as exc:
            return {
                "status": "not_available",
                "error": str(exc),
            }


def _main():
    """Entry point for `python -m music_brain.api`."""
    if not FASTAPI_AVAILABLE:
        print(
            "FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    import uvicorn

    uvicorn.run("music_brain.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    _main()
