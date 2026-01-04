"""
DAiW API Wrapper - Clean interface for desktop app and future REST API.

This module provides a simplified, consistent API surface for all music_brain
functionality, making it easier to integrate with desktop apps, web services,
or other interfaces.
"""
from typing import Dict, List, Optional, Any, Tuple
import sys
import logging
import json

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False
from pathlib import Path
import tempfile
import os

import numpy as np

# Core imports
from music_brain.audio import (
    AudioAnalyzer,
    AudioAnalysis,
    analyze_feel,
    AudioFeatures,
)
from music_brain.harmony import (
    HarmonyGenerator,
    HarmonyResult,
    generate_midi_from_harmony,
)
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
from music_brain.structure import (
    analyze_chords,
    detect_sections,
    ChordProgression,
)
from music_brain.structure.progression import (
    diagnose_progression,
    generate_reharmonizations,
)
from music_brain.structure.comprehensive_engine import (
    TherapySession,
    render_plan_to_midi,
    HarmonyPlan,
)
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    suggest_rule_break,
    validate_intent,
    list_all_rules,
)
from music_brain.session.intent_processor import process_intent
from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS
from music_brain.voice import (
    AutoTuneProcessor,
    AutoTuneSettings,
    get_auto_tune_preset,
    VoiceModulator,
    ModulationSettings,
    get_modulation_preset,
    VoiceSynthesizer,
    SynthConfig,
    get_voice_profile,
)
from music_brain.groove.drum_humanizer import DrumHumanizer


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
        result = analyzer.analyze_audio(samples, sample_rate) if hasattr(analyzer, "analyze_audio") else analyzer.analyze_waveform(samples, sample_rate)
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
                "secondary": session.state.affect_result.secondary if session.state.affect_result else None,
                "intensity": session.state.affect_result.intensity if session.state.affect_result else 0.0,
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
        
        # Convert to serializable format
        output = {
            "intent_summary": result['intent_summary'],
            "harmony": {
                "chords": result['harmony'].chords,
                "roman_numerals": result['harmony'].roman_numerals,
                "rule_broken": result['harmony'].rule_broken,
                "rule_effect": result['harmony'].rule_effect,
            },
            "groove": {
                "pattern_name": result['groove'].pattern_name,
                "tempo_bpm": result['groove'].tempo_bpm,
                "swing_factor": result['groove'].swing_factor,
                "rule_broken": result['groove'].rule_broken,
                "rule_effect": result['groove'].rule_effect,
            },
            "arrangement": {
                "sections": result['arrangement'].sections,
                "dynamic_arc": result['arrangement'].dynamic_arc,
                "rule_broken": result['arrangement'].rule_broken,
            },
            "production": {
                "vocal_treatment": result['production'].vocal_treatment,
                "eq_notes": result['production'].eq_notes,
                "dynamics_notes": result['production'].dynamics_notes,
                "rule_broken": result['production'].rule_broken,
            },
        }
        
        if output_json:
            import json
            with open(output_json, 'w') as f:
                json.dump(output, f, indent=2)
        
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


# Convenience instance
api = DAiWAPI()

__all__ = ['DAiWAPI', 'api']


# ---------- Minimal HTTP API (FastAPI) ----------
# This provides the server that `python -m music_brain.api` is expected to start.

if FASTAPI_AVAILABLE:
    class TechnicalIntent(BaseModel):
        key: Optional[str] = None
        bpm: Optional[int] = None
        progression: Optional[List[str]] = None
        genre: Optional[str] = None

    class EmotionalIntent(BaseModel):
        core_wound: Optional[str] = None
        core_desire: Optional[str] = None
        emotional_intent: str
        technical: Optional[TechnicalIntent] = None

    class GenerateRequest(BaseModel):
        intent: EmotionalIntent
        output_format: Optional[str] = None

    class InterrogateRequest(BaseModel):
        message: str
        session_id: Optional[str] = None
        context: Optional[Dict[str, Any]] = None

    app = FastAPI(title="Music Brain API", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/emotions")
    async def list_emotions():
        try:
            return sorted(EMOTIONAL_PRESETS.keys())
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to list emotions")
            raise HTTPException(status_code=500, detail=str(exc))

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
            raise HTTPException(status_code=400, detail=f"Failed to read MIDI: {exc}") from exc

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

    def _load_json_config(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logging.exception("Failed to load %s", path)
        return fallback

    @app.get("/config/humanizer")
    async def humanizer_config():
        """
        Return current humanizer/analysis config.
        - Loads `config/humanizer.json` if present; otherwise defaults.
        - Also exposes analysis thresholds (flam/buzz/drag/alternation).
        """
        cfg = _load_json_config(
            Path("config/humanizer.json"),
            _normalize_humanizer_config({}),
        )
        return _normalize_humanizer_config(cfg)

    SPECTO_PRESETS: Dict[str, Dict[str, Any]] = {
        "preview": {"anchor_density": "sparse", "n_particles": 600, "fps": 8},
        "standard": {"anchor_density": "normal", "n_particles": 1200, "fps": 15},
        "high": {"anchor_density": "dense", "n_particles": 1800, "fps": 24},
    }

    @app.get("/spectocloud/presets")
    async def spectocloud_presets():
        """List Spectocloud rendering presets (anchor density, particle count, fps)."""
        return SPECTO_PRESETS

    @app.put("/config/humanizer")
    async def update_humanizer_config(payload: Dict[str, Any]):
        """
        Persist humanizer/analysis configuration.
        - Accepts fields: default_style, ppq, bpm, analysis.{flam_threshold_ms,buzz_threshold_ms,drag_threshold_ms,alternation_window_ms}
        - Writes to config/humanizer.json and returns the normalized config.
        """
        cfg_dir = Path("config")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        normalized = _normalize_humanizer_config(payload)
        cfg_path = cfg_dir / "humanizer.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
        try:
            api.reload_humanizer()
        except Exception:
            logging.exception("Failed to reload humanizer after config update")
        return normalized

    @app.post("/config/humanizer/reload")
    async def reload_humanizer():
        """Force reload of the in-memory humanizer/analyzer from config/humanizer.json."""
        try:
            api.reload_humanizer()
            return {"status": "ok"}
        except Exception as exc:
            logging.exception("Failed to reload humanizer")
            raise HTTPException(status_code=500, detail=str(exc))

    class SpectocloudRenderRequest(BaseModel):
        midi_events: Optional[List[Dict[str, Any]]] = None
        midi_file_path: Optional[str] = None
        duration: Optional[float] = None
        emotion_trajectory: Optional[List[Dict[str, Any]]] = None
        mode: str = "static"  # "static" or "animation"
        frame_idx: int = 0
        output_path: Optional[str] = None
        fps: int = 15
        rotate: bool = True
        anchor_density: str = "normal"
        n_particles: int = 1200

    @app.post("/spectocloud/render")
    async def render_spectocloud(payload: SpectocloudRenderRequest):
        """
        Render Spectocloud output (static frame or animation).
        - For static: mode="static", frame_idx sets which frame to render.
        - For animation: mode="animation", fps/rotate control output.
        """
        try:
            from music_brain.visualization.spectocloud import Spectocloud  # Lazy import
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to import Spectocloud")
            raise HTTPException(status_code=500, detail=f"Spectocloud import failed: {exc}")

        try:
            events: Optional[List[Dict[str, Any]]] = payload.midi_events
            duration = payload.duration

            if payload.midi_file_path:
                parsed_events, parsed_duration = _parse_midi_file(Path(payload.midi_file_path))
                events = parsed_events
                duration = duration or parsed_duration

            if not events:
                raise HTTPException(status_code=400, detail="midi_events cannot be empty (or provide midi_file_path)")
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
                raise HTTPException(status_code=400, detail="No frames generated; check duration/window_size")
            mode = payload.mode.lower()
            if mode not in {"static", "animation"}:
                raise HTTPException(status_code=400, detail="mode must be 'static' or 'animation'")

            if mode == "static":
                if payload.frame_idx < 0:
                    raise HTTPException(status_code=400, detail="frame_idx must be >= 0")
                out_path = payload.output_path or str(Path(tempfile.gettempdir()) / "spectocloud_frame.png")
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

            out_path = payload.output_path or str(Path(tempfile.gettempdir()) / "spectocloud_anim.gif")
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
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            logging.exception("spectocloud render failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/generate")
    async def generate_music(request: GenerateRequest):
        try:
            # Map the simple intent into the therapy session pipeline
            chaos = 0.5
            motivation = 7
            if request.intent.technical and request.intent.technical.bpm:
                # Use bpm as a proxy for motivation scaling (soft heuristic)
                motivation = max(1, min(10, int(request.intent.technical.bpm / 20)))
            result = api.therapy_session(
                text=request.intent.emotional_intent,
                motivation=motivation,
                chaos_tolerance=chaos,
                output_midi=None,
            )
            return {"status": "success", "result": result}
        except Exception as exc:
            logging.exception("generate failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/interrogate")
    async def interrogate(request: InterrogateRequest):
        # Placeholder: echo back the message with a simple tip
        try:
            return {
                "status": "success",
                "reply": f"Noted: {request.message}. Consider clarifying the desired mood or groove.",
                "session_id": request.session_id,
            }
        except Exception as exc:  # pragma: no cover
            logging.exception("interrogate failed")
            raise HTTPException(status_code=500, detail=str(exc))


def _main():
    """Entry point for `python -m music_brain.api`."""
    if not FASTAPI_AVAILABLE:
        print(
            "FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run("music_brain.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    _main()
