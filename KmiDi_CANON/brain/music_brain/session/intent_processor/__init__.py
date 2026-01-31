"""
Intent Processor - Refactored package with modular processors.

This package provides the same API as the original intent_processor.py module,
but with improved organization into specialized processor modules.

100% backward compatibility maintained - all existing imports continue to work.
"""

from typing import Dict

# Import from base
from .base import (
    # Data classes
    GeneratedProgression,
    GeneratedGroove,
    GeneratedArrangement,
    GeneratedProduction,
    GeneratedMelody,
    GeneratedTexture,
    GeneratedTemporal,
    # Constants (re-export for any direct usage)
    CHROMATIC,
    CHROMATIC_FLAT,
    MAJOR_DIATONIC,
    BORROWED_FROM_MINOR,
    MODAL_INTERCHANGE,
    # Helper functions (private, but available if needed)
    _get_note_index,
    _transpose_chord,
    _romans_to_chords,
    _roman_to_chord,
)

# Import from harmony_processor
from .harmony_processor import (
    generate_progression_avoid_tonic,
    generate_progression_modal_interchange,
    generate_progression_parallel_motion,
    generate_progression_unresolved_dissonance,
    generate_progression_tritone_substitution,
    generate_progression_polytonality,
)

# Import from groove_processor
from .groove_processor import (
    generate_groove_constant_displacement,
    generate_groove_tempo_fluctuation,
    generate_groove_metric_modulation,
    generate_groove_dropped_beats,
    generate_groove_polyrhythmic_layers,
)

# Import from arrangement_processor
from .arrangement_processor import (
    generate_arrangement_structural_mismatch,
    generate_arrangement_extreme_dynamics,
    generate_arrangement_unbalanced_dynamics,
    generate_arrangement_buried_vocals,
    generate_arrangement_premature_climax,
    generate_production_guidelines,
)

# Import from melody_processor
from .melody_processor import (
    generate_melody_avoid_resolution,
    generate_melody_excessive_repetition,
    generate_melody_angular_intervals,
    generate_melody_anti_climax,
    generate_melody_monotone_drone,
    generate_melody_fragmented_phrases,
)

# Import from texture_processor
from .texture_processor import (
    generate_texture_frequency_masking,
    generate_texture_sparse_emptiness,
    generate_texture_dense_wall,
    generate_texture_conflicting_timbres,
    generate_texture_single_element_focus,
    generate_texture_timbral_drift,
)

# Import from temporal_processor
from .temporal_processor import (
    generate_temporal_extended_intro,
    generate_temporal_abrupt_ending,
    generate_temporal_time_stretch,
    generate_temporal_loop_hypnosis,
    generate_temporal_breath_pauses,
    generate_temporal_accelerando_decay,
)

# Import intent_schema for IntentProcessor
from music_brain.session.intent_schema import CompleteSongIntent


# =================================================================
# INTENT PROCESSOR CLASS
# =================================================================

class IntentProcessor:
    """
    Processes a CompleteSongIntent to generate musical elements.

    Usage:
        processor = IntentProcessor(intent)
        progression = processor.generate_harmony()
        groove = processor.generate_groove()
        arrangement = processor.generate_arrangement()
        production = processor.generate_production()
        melody = processor.generate_melody()
        texture = processor.generate_texture()
        temporal = processor.generate_temporal()
    """

    def __init__(self, intent: CompleteSongIntent):
        self.intent = intent
        self._parse_intent()

    def _parse_intent(self):
        """Extract key parameters from intent."""
        self.key = self.intent.technical_constraints.technical_key or "F"
        self.mode = self.intent.technical_constraints.technical_mode or "major"
        self.tempo_range = self.intent.technical_constraints.technical_tempo_range
        self.tempo = sum(self.tempo_range) // 2  # Middle of range
        self.rule_to_break = self.intent.technical_constraints.technical_rule_to_break
        self.narrative_arc = self.intent.song_intent.narrative_arc
        self.vulnerability = self.intent.song_intent.vulnerability_scale
        self.imagery = self.intent.song_intent.imagery_texture

    def generate_harmony(self) -> GeneratedProgression:
        """Generate chord progression based on harmony rule to break."""
        rule = self.rule_to_break

        if rule == "HARMONY_AvoidTonicResolution":
            return generate_progression_avoid_tonic(self.key, self.mode)
        elif rule == "HARMONY_ModalInterchange":
            return generate_progression_modal_interchange(self.key, self.mode)
        elif rule == "HARMONY_ParallelMotion":
            return generate_progression_parallel_motion(self.key, self.mode)
        elif rule == "HARMONY_UnresolvedDissonance":
            return generate_progression_unresolved_dissonance(self.key, self.mode)
        elif rule == "HARMONY_TritoneSubstitution":
            return generate_progression_tritone_substitution(self.key, self.mode)
        elif rule == "HARMONY_Polytonality":
            return generate_progression_polytonality(self.key, self.mode)
        else:
            # Default to modal interchange for most emotional contexts
            return generate_progression_modal_interchange(self.key, self.mode)

    def generate_groove(self) -> GeneratedGroove:
        """Generate groove pattern based on rhythm rule to break."""
        rule = self.rule_to_break

        if rule == "RHYTHM_ConstantDisplacement":
            return generate_groove_constant_displacement(self.tempo)
        elif rule == "RHYTHM_TempoFluctuation":
            return generate_groove_tempo_fluctuation(self.tempo)
        elif rule == "RHYTHM_MetricModulation":
            return generate_groove_metric_modulation(self.tempo)
        elif rule == "RHYTHM_DroppedBeats":
            return generate_groove_dropped_beats(self.tempo)
        elif rule == "RHYTHM_PolyrhythmicLayers":
            return generate_groove_polyrhythmic_layers(self.tempo)
        else:
            # Default groove based on genre feel
            feel = self.intent.technical_constraints.technical_groove_feel or ""
            if "laid back" in feel.lower():
                return generate_groove_constant_displacement(self.tempo)
            else:
                return generate_groove_tempo_fluctuation(self.tempo)

    def generate_arrangement(self) -> GeneratedArrangement:
        """Generate arrangement based on narrative arc and arrangement rules."""
        rule = self.rule_to_break

        if rule == "ARRANGEMENT_StructuralMismatch":
            return generate_arrangement_structural_mismatch(self.narrative_arc)
        elif rule == "ARRANGEMENT_ExtremeDynamicRange":
            return generate_arrangement_extreme_dynamics()
        elif rule == "ARRANGEMENT_UnbalancedDynamics":
            return generate_arrangement_unbalanced_dynamics()
        elif rule == "ARRANGEMENT_BuriedVocals":
            return generate_arrangement_buried_vocals()
        elif rule == "ARRANGEMENT_PrematureClimax":
            return generate_arrangement_premature_climax()
        else:
            return generate_arrangement_structural_mismatch(self.narrative_arc)

    def generate_production(self) -> GeneratedProduction:
        """Generate production guidelines."""
        return generate_production_guidelines(
            self.rule_to_break,
            self.vulnerability,
            self.imagery
        )

    def generate_melody(self) -> GeneratedMelody:
        """Generate melody guidelines based on melody rule to break."""
        rule = self.rule_to_break

        if rule == "MELODY_AvoidResolution":
            return generate_melody_avoid_resolution(self.key, self.mode)
        elif rule == "MELODY_ExcessiveRepetition":
            return generate_melody_excessive_repetition(self.key, self.mode)
        elif rule == "MELODY_AngularIntervals":
            return generate_melody_angular_intervals(self.key, self.mode)
        elif rule == "MELODY_AntiClimax":
            return generate_melody_anti_climax(self.key, self.mode)
        elif rule == "MELODY_MonotoneDrone":
            return generate_melody_monotone_drone(self.key, self.mode)
        elif rule == "MELODY_FragmentedPhrases":
            return generate_melody_fragmented_phrases(self.key, self.mode)
        else:
            # Default based on vulnerability - high vulnerability suggests avoiding resolution
            if self.vulnerability == "High":
                return generate_melody_avoid_resolution(self.key, self.mode)
            else:
                return generate_melody_avoid_resolution(self.key, self.mode)

    def generate_texture(self) -> GeneratedTexture:
        """Generate texture guidelines based on texture rule to break."""
        rule = self.rule_to_break

        if rule == "TEXTURE_FrequencyMasking":
            return generate_texture_frequency_masking()
        elif rule == "TEXTURE_SparseEmptiness":
            return generate_texture_sparse_emptiness()
        elif rule == "TEXTURE_DenseWall":
            return generate_texture_dense_wall()
        elif rule == "TEXTURE_ConflictingTimbres":
            return generate_texture_conflicting_timbres()
        elif rule == "TEXTURE_SingleElementFocus":
            return generate_texture_single_element_focus()
        elif rule == "TEXTURE_TimbralDrift":
            return generate_texture_timbral_drift()
        else:
            # Default based on imagery texture
            imagery_lower = self.imagery.lower() if self.imagery else ""
            if "sparse" in imagery_lower or "empty" in imagery_lower:
                return generate_texture_sparse_emptiness()
            elif "dense" in imagery_lower or "heavy" in imagery_lower:
                return generate_texture_dense_wall()
            else:
                return generate_texture_sparse_emptiness()

    def generate_temporal(self) -> GeneratedTemporal:
        """Generate temporal guidelines based on temporal rule to break."""
        rule = self.rule_to_break

        if rule == "TEMPORAL_ExtendedIntro":
            return generate_temporal_extended_intro()
        elif rule == "TEMPORAL_AbruptEnding":
            return generate_temporal_abrupt_ending()
        elif rule == "TEMPORAL_TimeStretch":
            return generate_temporal_time_stretch()
        elif rule == "TEMPORAL_LoopHypnosis":
            return generate_temporal_loop_hypnosis()
        elif rule == "TEMPORAL_BreathPauses":
            return generate_temporal_breath_pauses()
        elif rule == "TEMPORAL_AccelerandoDecay":
            return generate_temporal_accelerando_decay()
        else:
            # Default based on narrative arc
            if self.narrative_arc == "Repetitive Despair":
                return generate_temporal_loop_hypnosis()
            elif self.narrative_arc == "Sudden Shift":
                return generate_temporal_breath_pauses()
            else:
                return generate_temporal_breath_pauses()

    def generate_all(self) -> Dict:
        """Generate all elements and return as dict."""
        return {
            "harmony": self.generate_harmony(),
            "groove": self.generate_groove(),
            "arrangement": self.generate_arrangement(),
            "production": self.generate_production(),
            "melody": self.generate_melody(),
            "texture": self.generate_texture(),
            "temporal": self.generate_temporal(),
            "intent_summary": {
                "mood": self.intent.song_intent.mood_primary,
                "tension": self.intent.song_intent.mood_secondary_tension,
                "narrative": self.narrative_arc,
                "rule_broken": self.rule_to_break,
                "justification": self.intent.technical_constraints.rule_breaking_justification,
            },
        }


def process_intent(intent: CompleteSongIntent) -> Dict:
    """
    Convenience function to process an intent and return all generated elements.

    Args:
        intent: Complete song intent

    Returns:
        Dict with harmony, groove, arrangement, production, melody, texture,
        temporal, and intent_summary.
    """
    processor = IntentProcessor(intent)
    return processor.generate_all()


# Public API - maintain 100% backward compatibility
__all__ = [
    # Main API
    "IntentProcessor",
    "process_intent",
    # Data classes
    "GeneratedProgression",
    "GeneratedGroove",
    "GeneratedArrangement",
    "GeneratedProduction",
    "GeneratedMelody",
    "GeneratedTexture",
    "GeneratedTemporal",
    # Harmony functions
    "generate_progression_avoid_tonic",
    "generate_progression_modal_interchange",
    "generate_progression_parallel_motion",
    "generate_progression_unresolved_dissonance",
    "generate_progression_tritone_substitution",
    "generate_progression_polytonality",
    # Groove functions
    "generate_groove_constant_displacement",
    "generate_groove_tempo_fluctuation",
    "generate_groove_metric_modulation",
    "generate_groove_dropped_beats",
    "generate_groove_polyrhythmic_layers",
    # Arrangement functions (TODO: move to arrangement_processor.py)
    "generate_arrangement_structural_mismatch",
    "generate_arrangement_extreme_dynamics",
    "generate_arrangement_unbalanced_dynamics",
    "generate_arrangement_buried_vocals",
    "generate_arrangement_premature_climax",
    "generate_production_guidelines",
    # Melody functions (TODO: move to melody_processor.py)
    "generate_melody_avoid_resolution",
    "generate_melody_excessive_repetition",
    "generate_melody_angular_intervals",
    "generate_melody_anti_climax",
    "generate_melody_monotone_drone",
    "generate_melody_fragmented_phrases",
    # Texture functions (TODO: move to texture_processor.py)
    "generate_texture_frequency_masking",
    "generate_texture_sparse_emptiness",
    "generate_texture_dense_wall",
    "generate_texture_conflicting_timbres",
    "generate_texture_single_element_focus",
    "generate_texture_timbral_drift",
    # Temporal functions (TODO: move to temporal_processor.py)
    "generate_temporal_extended_intro",
    "generate_temporal_abrupt_ending",
    "generate_temporal_time_stretch",
    "generate_temporal_loop_hypnosis",
    "generate_temporal_breath_pauses",
    "generate_temporal_accelerando_decay",
]
