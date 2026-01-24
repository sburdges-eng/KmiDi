"""
Instrument Affinity Mapper - Map Articulation Profiles to Instrument Suggestions

Maps articulation characteristics to ranked instrument suggestions with confidence scores.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from .articulation_inference import ArticulationProfile

logger = logging.getLogger(__name__)


@dataclass
class InstrumentSuggestion:
    """Instrument suggestion with confidence score"""

    instrument_name: str
    confidence: float  # [0-1]
    reasoning: str  # Why this instrument was suggested
    alternative_names: List[str] = None  # Alternative names/variants

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        if self.alternative_names is None:
            result["alternative_names"] = []
        return result

    def __post_init__(self):
        if self.alternative_names is None:
            self.alternative_names = []


@dataclass
class InstrumentAffinity:
    """Instrument affinity results"""

    suggestions: List[InstrumentSuggestion]
    primary_suggestion: Optional[InstrumentSuggestion] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "primary_suggestion": (
                self.primary_suggestion.to_dict() if self.primary_suggestion else None
            ),
        }


class InstrumentAffinityMapper:
    """Map articulation profiles to instrument suggestions"""

    def __init__(self):
        """Initialize instrument affinity mapper"""
        # Mapping rules: excitation_type -> instrument suggestions
        self.instrument_mapping = {
            "lips_pressed_buzzing": [
                ("trumpet", 0.9, "Brass instrument with lip-buzzing excitation"),
                ("trombone", 0.85, "Brass instrument with similar excitation"),
                ("muted_brass", 0.8, "Muted brass instrument"),
                ("synth_brass", 0.7, "Synthesized brass-like sound"),
                ("vocal_formant_hybrid", 0.6, "Vocal synthesis with brass-like formants"),
            ],
            "reed_vibration": [
                ("clarinet", 0.9, "Single-reed instrument"),
                ("saxophone", 0.9, "Single-reed instrument"),
                ("oboe", 0.85, "Double-reed instrument"),
                ("bassoon", 0.85, "Double-reed instrument"),
                ("synthesized_reed", 0.7, "Synthesized reed-like sound"),
            ],
            "string_pluck": [
                ("guitar", 0.9, "Plucked string instrument"),
                ("harp", 0.85, "Plucked string instrument"),
                ("banjo", 0.8, "Plucked string instrument"),
                ("pizzicato_strings", 0.75, "Plucked orchestral strings"),
            ],
            "string_bow": [
                ("violin", 0.9, "Bowed string instrument"),
                ("viola", 0.9, "Bowed string instrument"),
                ("cello", 0.9, "Bowed string instrument"),
                ("double_bass", 0.85, "Bowed string instrument"),
                ("synthesized_strings", 0.7, "Synthesized string-like sound"),
            ],
            "vocal_formant": [
                ("vocal_synthesizer", 0.9, "Voice synthesizer"),
                ("formant_synth", 0.85, "Formant-based synthesizer"),
                ("vocal_formant_hybrid", 0.8, "Hybrid vocal synthesis"),
                ("choir", 0.7, "Choir/ensemble voice"),
            ],
            "breath_noise": [
                ("flute", 0.9, "Breath-driven instrument"),
                ("recorder", 0.85, "Breath-driven instrument"),
                ("whistle", 0.8, "Breath-driven instrument"),
                ("wind_synth", 0.7, "Synthesized wind instrument"),
            ],
        }

    def map_articulation(self, articulation_profile: ArticulationProfile) -> InstrumentAffinity:
        """
        Map articulation profile to instrument suggestions

        Args:
            articulation_profile: Articulation profile to map

        Returns:
            InstrumentAffinity with ranked suggestions
        """
        logger.info(f"Mapping articulation to instruments: {articulation_profile.excitation_type}")

        suggestions = []

        # Get base suggestions from excitation type
        base_suggestions = self.instrument_mapping.get(
            articulation_profile.excitation_type,
            [("custom", 0.5, "Custom instrument based on articulation characteristics")],
        )

        # Adjust confidence based on airflow continuity and articulation method
        for instrument_name, base_confidence, reasoning in base_suggestions:
            confidence = base_confidence

            # Adjust confidence based on airflow continuity
            # Continuous airflow: higher confidence for sustained instruments
            # Pulsed airflow: higher confidence for plucked/struck instruments
            if articulation_profile.airflow_continuity > 0.7:
                # Continuous airflow preferred
                if instrument_name in ["flute", "clarinet", "violin", "trumpet"]:
                    confidence += 0.1
                elif instrument_name in ["guitar", "harp", "pluck"]:
                    confidence -= 0.1
            elif articulation_profile.airflow_continuity < 0.3:
                # Pulsed airflow preferred
                if instrument_name in ["guitar", "harp", "pizzicato"]:
                    confidence += 0.1
                elif instrument_name in ["flute", "clarinet", "violin"]:
                    confidence -= 0.1

            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            suggestion = InstrumentSuggestion(
                instrument_name=instrument_name, confidence=confidence, reasoning=reasoning
            )
            suggestions.append(suggestion)

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda x: x.confidence, reverse=True)

        # Primary suggestion is the highest confidence one
        primary_suggestion = suggestions[0] if suggestions else None

        return InstrumentAffinity(suggestions=suggestions, primary_suggestion=primary_suggestion)

    def get_alternative_names(self, instrument_name: str) -> List[str]:
        """Get alternative names for an instrument"""
        alternatives_map = {
            "trumpet": ["trumpet", "cornet", "bugle"],
            "violin": ["violin", "fiddle"],
            "guitar": ["guitar", "acoustic_guitar", "electric_guitar"],
            "vocal_synthesizer": ["vocal_synth", "voice_synthesizer", "vocoder"],
            "formant_synth": ["formant_synthesizer", "formant_based_synth"],
        }

        return alternatives_map.get(instrument_name, [instrument_name])
