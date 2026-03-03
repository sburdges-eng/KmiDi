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
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, ValidationError
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False
from pathlib import Path
import tempfile
import os

import numpy as np

# Constants for input validation
VALID_MUSICAL_MODES = {"major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"}

# Core imports
from music_brain.audio import (
    AudioAnalyzer,
    AudioAnalysis,
    analyze_feel,
    AudioFeatures,
    render_midi_to_audio,
)
try:
    from music_brain.harmony import (
        HarmonyGenerator,
        HarmonyResult,
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
from music_brain.engine_api.schema import CompleteSongIntentRequest
try:
    from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS
except ImportError:  # pragma: no cover - compatibility shim for partial installs
    EMOTIONAL_PRESETS = {}
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
    VoiceClassifier,
)
try:
    from music_brain.groove.drum_humanizer import DrumHumanizer
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
        self.user_lyrics: Optional[str] = None
        self.user_lyrics_source: str = "none"
        self.last_generated_lyrics: Optional[str] = None

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

    # ========== Lyrics Handling ==========

    def set_lyrics(self, lyrics: str, source: str = "user") -> Dict[str, Any]:
        """
        Persist user-provided lyrics (or clear when empty) and return a lightweight summary.
        """
        cleaned = (lyrics or "").strip()
        if not cleaned:
            self.user_lyrics = None
            self.user_lyrics_source = "none"
            return {
                "status": "cleared",
                "source": self.user_lyrics_source,
                "lines": 0,
                "word_count": 0,
            }

        self.user_lyrics = cleaned
        self.user_lyrics_source = source or "user"
        self.last_generated_lyrics = None

        return {
            "status": "stored",
            "source": self.user_lyrics_source,
            "lines": len(cleaned.splitlines()),
            "word_count": len(cleaned.split()),
            "preview": cleaned[:140],
        }

    def get_lyrics(self) -> Dict[str, Any]:
        """Return the active lyric payload and provenance."""
        return {
            "lyrics": self.user_lyrics,
            "source": self.user_lyrics_source,
            "generated": self.last_generated_lyrics,
        }

    def _intent_field(self, intent: Any, name: str, default: str = "") -> str:
        """Safely pull a field from either a dict-like intent or pydantic model."""
        if intent is None:
            return default
        if isinstance(intent, dict):
            return str(intent.get(name, default) or default)
        return str(getattr(intent, name, default) or default)

    def generate_structured_lyrics(self, intent: Any) -> str:
        """
        Lightweight, dependency-free fallback lyric generator.

        Produces a simple verse/chorus layout conditioned on the emotional intent fields
        without pulling heavyweight models into the runtime.
        """
        wound = self._intent_field(intent, "core_wound", "unspecified wound")
        desire = self._intent_field(intent, "core_desire", "longing")
        emotion = self._intent_field(intent, "emotional_intent", "unspecified emotion")

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
            ["[Verse 1]"]
            + verse1
            + ["", "[Chorus]"]
            + chorus
            + ["", "[Verse 2]"]
            + verse2
            + ["", "[Chorus]", *chorus]
        )
        self.last_generated_lyrics = generated
        return generated

    def _select_lyric_payload(self, intent: Any) -> Tuple[str, str]:
        """
        Decide which text should drive generation:
        - User lyrics when present (highest priority)
        - Generated fallback lyrics when no user payload exists
        - Raw emotional intent as a final fallback
        """
        if self.user_lyrics:
            return self.user_lyrics, self.user_lyrics_source
        if intent:
            generated = self.generate_structured_lyrics(intent)
            return generated, "generated"
        return "emotional intent", "intent"
    
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
    
    def _convert_request_to_complete_intent(self, request: Any) -> CompleteSongIntent:
        """
        Convert UI request payload to CompleteSongIntent.
        
        Maps the simplified UI parameters to the full CompleteSongIntent schema.
        """
        import time
        
        tech = request.intent.technical or {}
        emotional = request.intent.emotional_intent or ""
        
        # Extract emotion/mood from emotional_intent string
        mood_primary = emotional
        if "(" in emotional:
            mood_primary = emotional.split("(")[0].strip()
        
        # Map common emotions to mood_primary
        emotion_map = {
            "grief": "grief",
            "sadness": "grief",
            "joy": "tenderness",
            "happiness": "tenderness",
            "anger": "rage",
            "rage": "rage",
            "fear": "fear",
            "love": "tenderness",
            "nostalgia": "nostalgia",
            "awe": "awe",
        }
        for key, value in emotion_map.items():
            if key.lower() in emotional.lower():
                mood_primary = value
                break
        
        # Extract key and mode from technical.key (format: "F major" or "C minor")
        technical_key = "C"
        technical_mode = "major"
        if tech.get("key"):
            key_parts = tech["key"].split()
            technical_key = key_parts[0] if key_parts else "C"
            if len(key_parts) > 1:
                # Validate mode against known modes
                mode_candidate = key_parts[1].lower()
                technical_mode = mode_candidate if mode_candidate in VALID_MUSICAL_MODES else "major"
        
        # Calculate tempo range from BPM with validation
        bpm = tech.get("bpm") or 82
        try:
            bpm = int(bpm)
            bpm = max(40, min(300, bpm))  # Clamp to valid range
        except (ValueError, TypeError):
            bpm = 82
        tempo_range = (max(60, bpm - 20), min(140, bpm + 20))
        
        # Create CompleteSongIntent
        intent = CompleteSongIntent(
            core_event=request.intent.core_wound or emotional,
            core_longing=request.intent.core_desire or "",
            mood_primary=mood_primary,
            technical_genre=tech.get("genre") or "",
            technical_tempo_range=tempo_range,
            technical_key=technical_key,
            technical_mode=technical_mode,
            vulnerability_scale=0.5,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        return intent


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
        duration: Optional[float] = None  # Duration in minutes
        structure: Optional[List[Dict[str, Any]]] = None  # Song sections with repetitions
        instruments: Optional[List[Dict[str, Any]]] = None  # Instruments with techniques
        techniques: Optional[List[str]] = None  # Production techniques
        groove_feel: Optional[str] = None  # Rhythmic feel (e.g. "Straight/Driving")
        rule_to_break: Optional[str] = None  # Intentional theory violation
        rule_justification: Optional[str] = None  # Narrative reason for rule break

    class EmotionalIntent(BaseModel):
        core_wound: Optional[str] = None
        core_desire: Optional[str] = None
        emotional_intent: str
        technical: Optional[TechnicalIntent] = None
        vulnerability_scale: Optional[float] = None  # 0.0 - 1.0 emotional openness
        narrative_arc: Optional[str] = None  # Energetic trajectory (e.g. "Climb-to-Climax")

    class GenerateRequest(BaseModel):
        intent: EmotionalIntent
        output_format: Optional[str] = None

    class InterrogateRequest(BaseModel):
        message: str
        session_id: Optional[str] = None
        context: Optional[Dict[str, Any]] = None

    class LyricsRequest(BaseModel):
        lyrics: str
        source: Optional[str] = "user"

    app = FastAPI(title="Music Brain API", version="0.1.0")

    # Add CORS middleware to allow requests from frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:1420",  # Vite dev server
            "http://127.0.0.1:1420",  # Vite dev server (alternative)
            "tauri://localhost",      # Tauri app
            "http://localhost:5173",  # Alternative Vite port
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Music Brain API",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "generate": "/generate (POST)",
                "emotions": "/emotions",
                "lyrics": "/lyrics (GET/POST)",
                "interrogate": "/interrogate (POST)",
                "docs": "/docs",
                "openapi": "/openapi.json"
            },
            "documentation": "Visit /docs for interactive API documentation"
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/audio/{file_path:path}")
    async def serve_audio(file_path: str):
        """Serve audio files via HTTP. file_path should be URL-encoded."""
        import urllib.parse
        decoded_path = urllib.parse.unquote(file_path)
        audio_file = Path(decoded_path)
        
        logging.info(f"Serving audio file: {decoded_path}")
        logging.info(f"File exists: {audio_file.exists()}, is_file: {audio_file.is_file() if audio_file.exists() else 'N/A'}")
        
        if not audio_file.exists():
            logging.error(f"Audio file not found: {decoded_path}")
            # Try to find alternative paths (maybe MIDI file exists but audio doesn't)
            if decoded_path.endswith(('.wav', '.mp3', '.ogg')):
                midi_path = decoded_path.rsplit('.', 1)[0] + '.mid'
                if Path(midi_path).exists():
                    logging.warning(f"Audio file not found, but MIDI exists: {midi_path}. Audio conversion may be needed.")
            raise HTTPException(
                status_code=404, 
                detail=f"Audio file not found: {decoded_path}. The file may not have been generated yet or may have been cleaned up."
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

            # Handle audio file input (convert to MIDI events if needed)
            if payload.audio_file_path:
                audio_path = Path(payload.audio_file_path)
                if not audio_path.exists():
                    raise HTTPException(status_code=400, detail=f"Audio file not found: {payload.audio_file_path}")
                # For now, if audio file provided, we'd need to extract MIDI from it
                # This is a placeholder - actual implementation would analyze audio and extract MIDI
                # For now, raise an error suggesting MIDI file instead
                raise HTTPException(
                    status_code=400, 
                    detail="Audio file analysis not yet implemented. Please provide midi_file_path or midi_events instead."
                )

            if payload.midi_file_path:
                parsed_events, parsed_duration = _parse_midi_file(Path(payload.midi_file_path))
                events = parsed_events
                duration = duration or parsed_duration

            if not events:
                raise HTTPException(status_code=400, detail="provide audio_file_path, midi_file_path, or midi_events")
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
            # Try to use full intent pipeline if we have advanced parameters
            tech = request.intent.technical
            # Default to the full intent pipeline for all requests.
            use_full_pipeline = True
            strict_intent = None
            
            if use_full_pipeline:
                # Use full CompleteSongIntent pipeline
                logging.info("Using full intent pipeline with CompleteSongIntent")

                if tech is None:
                    raise HTTPException(
                        status_code=422,
                        detail="technical payload is required for complete intent generation",
                    )

                # Strict boundary validation for UI->engine payload.
                structure_payload = tech.structure or []
                instruments_payload = []
                for inst in (tech.instruments or []):
                    if isinstance(inst, dict):
                        instruments_payload.append(
                            {
                                "instrument": inst.get("instrument")
                                or inst.get("name")
                                or inst.get("type")
                                or "",
                                "techniques": inst.get("techniques", []) or [],
                            }
                        )
                    else:
                        instruments_payload.append({"instrument": str(inst), "techniques": []})

                strict_payload = {
                    "core_desire": request.intent.core_desire or "",
                    "mood_primary": request.intent.emotional_intent or "",
                    "genre": tech.genre or "",
                    "tempo": tech.bpm if tech.bpm is not None else 120,
                    "key_mode": tech.key or "",
                    "structure": structure_payload,
                    "instruments": instruments_payload,
                    "allow_legacy_fallback": False,
                    "groove_feel": tech.groove_feel or "Straight/Driving",
                    "narrative_arc": request.intent.narrative_arc or "Climb-to-Climax",
                    "rule_to_break": tech.rule_to_break,
                    "rule_justification": tech.rule_justification,
                }
                try:
                    if hasattr(CompleteSongIntentRequest, "model_validate"):
                        strict_intent = CompleteSongIntentRequest.model_validate(strict_payload)
                    else:  # pydantic v1 compatibility
                        strict_intent = CompleteSongIntentRequest.parse_obj(strict_payload)
                except ValidationError as validation_error:
                    raise HTTPException(status_code=422, detail=validation_error.errors()) from validation_error
                
                # Convert request to CompleteSongIntent
                def _convert_to_intent(req: GenerateRequest, validated: CompleteSongIntentRequest) -> CompleteSongIntent:
                    """Helper to convert request to CompleteSongIntent."""
                    import time
                    key_parts = validated.key_mode.split()
                    technical_key = key_parts[0]
                    technical_mode = key_parts[1].lower()
                    tempo_range = (max(60, validated.tempo - 20), min(140, validated.tempo + 20))
                    
                    return CompleteSongIntent(
                        core_event=req.intent.core_wound or validated.core_desire,
                        core_longing=validated.core_desire,
                        mood_primary=validated.mood_primary,
                        narrative_arc=validated.narrative_arc,
                        vulnerability_scale=req.intent.vulnerability_scale if req.intent.vulnerability_scale is not None else 0.5,
                        technical_genre=validated.genre,
                        technical_tempo_range=tempo_range,
                        technical_key=technical_key,
                        technical_mode=technical_mode,
                        technical_groove_feel=validated.groove_feel,
                        technical_rule_to_break=validated.rule_to_break or "",
                        rule_breaking_justification=validated.rule_justification or "",
                        created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                
                complete_intent = _convert_to_intent(request, strict_intent)
                
                # Process full intent
                result = api.process_song_intent(complete_intent, output_json=None)
                
                # Generate output file if format requested
                output_midi = None
                output_audio = None
                if request.output_format:
                    import tempfile
                    import time
                    if request.output_format in ['mid', 'midi']:
                        output_midi = str(Path(tempfile.gettempdir()) / f"generated_{int(time.time())}.mid")
                    elif request.output_format in ['wav', 'mp3']:
                        output_audio = str(Path(tempfile.gettempdir()) / f"generated_{int(time.time())}.{request.output_format}")
                        output_midi = str(Path(tempfile.gettempdir()) / f"generated_{int(time.time())}.mid")
                
                # Generate MIDI from harmony result
                if output_midi and result.get("harmony"):
                    try:
                        # Extract harmony info
                        harmony = result["harmony"]
                        groove = result.get("groove", {})
                        # Validate and clamp duration to positive value (0.1 - 60 minutes)
                        duration_minutes = tech.duration if tech and tech.duration is not None else 3.0
                        try:
                            duration_minutes = float(duration_minutes)
                            duration_minutes = max(0.1, min(60.0, duration_minutes))  # Clamp to reasonable range
                        except (ValueError, TypeError):
                            duration_minutes = 3.0
                        
                        # Validate and clamp BPM
                        bpm = strict_intent.tempo if strict_intent else 82
                        
                        length_bars = int((duration_minutes * bpm) / 4)
                        length_bars = max(16, min(128, length_bars))
                        
                        # Extract key and mode with validation
                        key_str = strict_intent.key_mode if strict_intent else "C major"
                        key_parts = key_str.split()
                        root_note = key_parts[0]
                        mode = key_parts[1].lower()
                        
                        # Extract structure and instruments from request
                        structure = [
                            section.model_dump() if hasattr(section, "model_dump") else section.dict()
                            for section in strict_intent.structure
                        ] if strict_intent else None
                        instruments = [
                            track.model_dump() if hasattr(track, "model_dump") else track.dict()
                            for track in strict_intent.instruments
                        ] if strict_intent else None
                        
                        # If structure is provided, calculate total bars from structure
                        # Otherwise use calculated length_bars
                        if structure:
                            total_structure_bars = sum(
                                section.get("bars", 4) * section.get("repetitions", 1)
                                for section in structure
                            )
                            # Use structure bars if it's reasonable, otherwise keep calculated
                            if total_structure_bars > 0:
                                length_bars = total_structure_bars
                        
                        # Create HarmonyPlan from result
                        plan = HarmonyPlan(
                            root_note=root_note,
                            mode=mode,
                            tempo_bpm=bpm,
                            time_signature="4/4",
                            length_bars=length_bars,
                            chord_symbols=harmony.get("chords", ["C", "Am", "F", "G"]),
                            harmonic_rhythm="1_chord_per_bar",
                            mood_profile=result.get("intent_summary", {}).get("mood", "neutral"),
                            complexity=0.5,
                            structure=structure,
                            instruments=instruments
                        )
                        
                        # Render MIDI
                        midi_path = render_plan_to_midi(plan, output_midi)
                        result["midi_path"] = midi_path
                    except Exception as midi_exc:
                        logging.exception("Failed to generate MIDI from full intent, falling back")
                        if strict_intent and strict_intent.allow_legacy_fallback:
                            use_full_pipeline = False
                        else:
                            raise HTTPException(
                                status_code=500,
                                detail=f"MIDI generation failed and legacy fallback is disabled: {midi_exc}",
                            ) from midi_exc
                
                lyric_text, lyric_source = api._select_lyric_payload(request.intent)
                
                # Build response with structure and instruments info
                response = {
                    "status": "success",
                    "result": result,
                    "lyrics": {
                        "source": lyric_source,
                        "text": lyric_text,
                    },
                }
                
                # Add structure and instruments information if provided
                structure = [
                    section.model_dump() if hasattr(section, "model_dump") else section.dict()
                    for section in strict_intent.structure
                ] if strict_intent else None
                instruments = [
                    track.model_dump() if hasattr(track, "model_dump") else track.dict()
                    for track in strict_intent.instruments
                ] if strict_intent else None
                
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
                                "name": inst.get("name", "instrument") if isinstance(inst, dict) else "instrument",
                                "type": inst.get("type", "chord") if isinstance(inst, dict) else "chord",
                                "channel": inst.get("channel") if isinstance(inst, dict) else None,
                            }
                            for inst in instruments
                        ],
                    }
                
                # Add file paths to response
                if output_midi and result.get("midi_path"):
                    response["midi_path"] = result["midi_path"]
                    midi_file = Path(result["midi_path"])
                    
                    if output_audio:
                        # Render audio from generated MIDI.
                        audio_path = str(midi_file.with_suffix(".wav"))
                        try:
                            render_midi_to_audio(str(midi_file), audio_path)
                        except Exception as render_exc:
                            logging.exception("Audio render failed from MIDI")
                            raise HTTPException(
                                status_code=500,
                                detail=f"MIDI generated but audio render failed: {render_exc}",
                            )
                        response["audio_path"] = audio_path
                        response["output_path"] = audio_path
                    else:
                        response["output_path"] = result["midi_path"]
                
                return response
            
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
            lyric_text, lyric_source = api._select_lyric_payload(request.intent)
            
            # Generate output file if format requested
            output_midi = None
            output_audio = None
            if request.output_format:
                import tempfile
                import time
                if request.output_format in ['mid', 'midi']:
                    output_midi = str(Path(tempfile.gettempdir()) / f"generated_{int(time.time())}.mid")
                elif request.output_format in ['wav', 'mp3']:
                    output_audio = str(Path(tempfile.gettempdir()) / f"generated_{int(time.time())}.{request.output_format}")
                    output_midi = str(Path(tempfile.gettempdir()) / f"generated_{int(time.time())}.mid")
            
            result = api.therapy_session(
                text=lyric_text or request.intent.emotional_intent,
                motivation=motivation,
                chaos_tolerance=chaos,
                output_midi=output_midi,
            )
            
            response = {
                "status": "success",
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
                    response["audio_path"] = result["midi_path"].replace(".mid", f".{request.output_format}")
                    response["output_path"] = response["audio_path"]
                else:
                    response["output_path"] = result["midi_path"]
            
            return response
        except HTTPException:
            raise
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

    @app.post("/lyrics")
    async def set_lyrics(payload: LyricsRequest):
        """
        Persist user-supplied lyrics and return a summary.
        """
        try:
            return api.set_lyrics(payload.lyrics, source=payload.source or "user")
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to set lyrics")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/lyrics")
    async def get_lyrics():
        """
        Return the current lyric payload, source, and any cached generated lyrics.
        """
        try:
            return api.get_lyrics()
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to fetch lyrics")
            raise HTTPException(status_code=500, detail=str(exc))

    # ========== Audio Emotion Classification Endpoints ==========

    class AudioClassifyRequest(BaseModel):
        audio_path: str
        model_type: Optional[str] = "emotion_7"
        top_k: Optional[int] = 3

    class VoiceClassifyRequest(BaseModel):
        audio_path: str
        top_k: Optional[int] = 3

    @app.post("/audio/classify")
    async def classify_audio(request: AudioClassifyRequest):
        """
        Classify emotion from audio file using trained ML models.

        Returns valence/arousal coordinates for integration with
        the emotion-to-music generation pipeline.
        """
        try:
            from music_brain.emotion.audio_emotion_classifier import AudioEmotionClassifier

            classifier = AudioEmotionClassifier(model_type=request.model_type)
            if not classifier.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="Audio classifier model not available. Check model checkpoints."
                )

            result = classifier.classify(request.audio_path, top_k=request.top_k)
            return {
                "status": "success",
                "result": result.to_dict(),
            }
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Audio classification dependencies not installed: {exc}"
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logging.exception("audio classify failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/audio/valence-arousal")
    async def get_audio_valence_arousal(request: AudioClassifyRequest):
        """
        Get valence/arousal coordinates from audio.

        Primary interface for emotion-to-music mapping.
        """
        try:
            from music_brain.emotion.audio_emotion_classifier import AudioEmotionClassifier

            classifier = AudioEmotionClassifier(model_type=request.model_type)
            if not classifier.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="Audio classifier model not available."
                )

            va = classifier.get_valence_arousal(request.audio_path)
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
                detail=f"Audio classification dependencies not installed: {exc}"
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logging.exception("audio valence-arousal failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/audio/models")
    async def list_audio_models():
        """List available audio classification models."""
        try:
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
        except Exception as exc:
            logging.exception("list audio models failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/voice/classify")
    async def classify_voice(request: VoiceClassifyRequest):
        """Classify voice type (alto/bass/soprano/tenor) from audio."""
        try:
            result = api.classify_voice_file(request.audio_path, top_k=request.top_k or 3)
            return {"status": "success", "result": result}
        except Exception as exc:
            logging.exception("voice classify failed")
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
