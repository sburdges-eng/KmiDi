"""
ML Melody Generator - Neural network-based melody generation.

Uses the trained MelodyTransformer model to generate melodies from
emotional embeddings. Falls back to rule-based generation when
models are not available.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from pathlib import Path

# Note to interval mappings
SCALE_PATTERNS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

# Emotion to melodic characteristics
EMOTION_MELODIC_TRAITS = {
    "joy": {
        "range_octaves": 1.5,
        "leap_probability": 0.4,
        "ascending_bias": 0.6,
        "resolution_strength": 0.8,
        "preferred_intervals": [2, 4, 5],  # M2, M3, P4
        "avoid_intervals": [1, 6],  # m2, tritone
    },
    "grief": {
        "range_octaves": 1.0,
        "leap_probability": 0.2,
        "ascending_bias": 0.3,
        "resolution_strength": 0.2,  # Avoid resolution
        "preferred_intervals": [1, 3, 5],  # m2, m3, P4
        "avoid_intervals": [7],  # Avoid strong P5 resolution
    },
    "anger": {
        "range_octaves": 1.5,
        "leap_probability": 0.5,
        "ascending_bias": 0.5,
        "resolution_strength": 0.4,
        "preferred_intervals": [1, 6, 7],  # m2, tritone, P5
        "avoid_intervals": [],
    },
    "fear": {
        "range_octaves": 0.8,
        "leap_probability": 0.3,
        "ascending_bias": 0.4,
        "resolution_strength": 0.1,
        "preferred_intervals": [1, 2, 6],  # m2, M2, tritone
        "avoid_intervals": [7],
    },
    "hope": {
        "range_octaves": 1.2,
        "leap_probability": 0.4,
        "ascending_bias": 0.7,  # Rising melodies
        "resolution_strength": 0.5,
        "preferred_intervals": [2, 4, 7],
        "avoid_intervals": [1],
    },
    "longing": {
        "range_octaves": 1.0,
        "leap_probability": 0.35,
        "ascending_bias": 0.5,
        "resolution_strength": 0.3,  # Suspended endings
        "preferred_intervals": [2, 3, 5],
        "avoid_intervals": [],
    },
    "nostalgia": {
        "range_octaves": 1.0,
        "leap_probability": 0.3,
        "ascending_bias": 0.45,
        "resolution_strength": 0.6,
        "preferred_intervals": [2, 3, 4],
        "avoid_intervals": [6],
    },
    "defiance": {
        "range_octaves": 1.3,
        "leap_probability": 0.5,
        "ascending_bias": 0.6,
        "resolution_strength": 0.7,
        "preferred_intervals": [4, 5, 7],  # Strong intervals
        "avoid_intervals": [],
    },
}

# Default traits for unknown emotions
DEFAULT_TRAITS = {
    "range_octaves": 1.0,
    "leap_probability": 0.3,
    "ascending_bias": 0.5,
    "resolution_strength": 0.5,
    "preferred_intervals": [2, 4, 5, 7],
    "avoid_intervals": [],
}


@dataclass
class MelodyConfig:
    """Configuration for melody generation."""
    
    key: str = "C"
    mode: str = "major"
    octave: int = 4  # Base octave (middle C = C4 = MIDI 60)
    length_notes: int = 8
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    
    # Generation parameters
    use_ml_model: bool = True
    ml_model_path: Optional[str] = None
    
    def get_root_midi(self) -> int:
        """Get MIDI note number for root note."""
        note_map = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
        }
        return 12 * (self.octave + 1) + note_map.get(self.key, 0)


@dataclass
class GeneratedMelody:
    """A generated melody with metadata."""
    
    notes: List[int]  # MIDI note numbers
    durations: List[float]  # Duration in beats
    velocities: List[int]  # MIDI velocity (0-127)
    
    key: str = "C"
    mode: str = "major"
    emotion: str = "neutral"
    
    # Generation metadata
    method: str = "rule_based"  # "rule_based" or "ml_model"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "notes": self.notes,
            "durations": self.durations,
            "velocities": self.velocities,
            "key": self.key,
            "mode": self.mode,
            "emotion": self.emotion,
            "method": self.method,
            "confidence": self.confidence,
        }
    
    def transpose(self, semitones: int) -> "GeneratedMelody":
        """Return a transposed copy of this melody."""
        return GeneratedMelody(
            notes=[n + semitones for n in self.notes],
            durations=self.durations.copy(),
            velocities=self.velocities.copy(),
            key=self.key,
            mode=self.mode,
            emotion=self.emotion,
            method=self.method,
            confidence=self.confidence,
        )


class MLMelodyGenerator:
    """
    ML-based melody generator using trained models.
    
    Falls back to rule-based generation when models are unavailable.
    """
    
    def __init__(self, config: Optional[MelodyConfig] = None):
        self.config = config or MelodyConfig()
        self._ml_model = None
        self._ml_available = False
        self._load_ml_model()
    
    def _load_ml_model(self) -> None:
        """Attempt to load the ML melody model."""
        if not self.config.use_ml_model:
            return
        
        try:
            from penta_core.ml.inference import create_engine_by_name
            from penta_core.ml.model_registry import get_model
            
            # Try to load MelodyTransformer
            model_info = get_model("melodytransformer")
            if model_info and Path(model_info.path).exists():
                engine = create_engine_by_name("melodytransformer")
                if engine and engine.load():
                    self._ml_model = engine
                    self._ml_available = True
        except ImportError:
            pass
        except Exception:
            pass
    
    def is_ml_available(self) -> bool:
        """Check if ML model is available."""
        return self._ml_available
    
    def generate(
        self,
        emotion: str,
        length: Optional[int] = None,
        key: Optional[str] = None,
        mode: Optional[str] = None,
        context_notes: Optional[List[int]] = None,
    ) -> GeneratedMelody:
        """
        Generate a melody for the given emotion.
        
        Args:
            emotion: Target emotion (grief, joy, anger, etc.)
            length: Number of notes to generate
            key: Musical key (overrides config)
            mode: Scale mode (overrides config)
            context_notes: Optional previous notes for continuation
        
        Returns:
            GeneratedMelody with notes, durations, and velocities
        """
        length = length or self.config.length_notes
        key = key or self.config.key
        mode = mode or self.config.mode
        
        # Try ML model first
        if self._ml_available and self._ml_model:
            try:
                return self._generate_ml(emotion, length, key, mode, context_notes)
            except Exception:
                pass
        
        # Fall back to rule-based generation
        return self._generate_rule_based(emotion, length, key, mode, context_notes)
    
    def _generate_ml(
        self,
        emotion: str,
        length: int,
        key: str,
        mode: str,
        context_notes: Optional[List[int]],
    ) -> GeneratedMelody:
        """Generate melody using ML model."""
        import numpy as np
        
        # Create emotion embedding (simplified - real implementation would use EmotionRecognizer)
        emotion_embedding = self._create_emotion_embedding(emotion)
        
        # Prepare input
        inputs = {"emotion_embedding": emotion_embedding}
        
        # Run inference
        result = self._ml_model.infer(inputs)
        
        # Extract note probabilities from output
        note_probs = result.get_output()
        
        # Sample notes from probabilities
        notes = self._sample_notes_from_probs(note_probs, length, key, mode)
        
        # Generate durations and velocities based on emotion
        traits = EMOTION_MELODIC_TRAITS.get(emotion.lower(), DEFAULT_TRAITS)
        durations = self._generate_durations(length, traits)
        velocities = self._generate_velocities(length, emotion)
        
        return GeneratedMelody(
            notes=notes,
            durations=durations,
            velocities=velocities,
            key=key,
            mode=mode,
            emotion=emotion,
            method="ml_model",
            confidence=float(result.confidence or 0.8),
        )
    
    def _generate_rule_based(
        self,
        emotion: str,
        length: int,
        key: str,
        mode: str,
        context_notes: Optional[List[int]],
    ) -> GeneratedMelody:
        """Generate melody using rule-based approach."""
        # Get traits for this emotion
        traits = EMOTION_MELODIC_TRAITS.get(emotion.lower(), DEFAULT_TRAITS)
        
        # Get scale for this key/mode
        scale = self._get_scale(key, mode)
        
        # Calculate MIDI root
        config = MelodyConfig(key=key, mode=mode)
        root = config.get_root_midi()
        
        # Generate notes
        notes = []
        current_note = root
        
        # Use context if available
        if context_notes:
            current_note = context_notes[-1]
        
        for i in range(length):
            if i == 0 and not context_notes:
                # Start on tonic, 3rd, or 5th
                start_degrees = [0, 2, 4]  # 1st, 3rd, 5th scale degrees
                degree = np.random.choice(start_degrees)
                current_note = root + scale[degree % len(scale)]
            else:
                # Choose next note
                current_note = self._choose_next_note(
                    current_note, root, scale, traits, i, length
                )
            
            notes.append(current_note)
        
        # Apply resolution if needed
        if traits["resolution_strength"] > 0.5:
            notes = self._apply_resolution(notes, root, scale, traits)
        
        # Generate durations and velocities
        durations = self._generate_durations(length, traits)
        velocities = self._generate_velocities(length, emotion)
        
        return GeneratedMelody(
            notes=notes,
            durations=durations,
            velocities=velocities,
            key=key,
            mode=mode,
            emotion=emotion,
            method="rule_based",
            confidence=0.6,
        )
    
    def _get_scale(self, key: str, mode: str) -> List[int]:
        """Get scale intervals for key and mode."""
        return SCALE_PATTERNS.get(mode.lower(), SCALE_PATTERNS["major"])
    
    def _choose_next_note(
        self,
        current: int,
        root: int,
        scale: List[int],
        traits: Dict,
        position: int,
        total_length: int,
    ) -> int:
        """Choose the next note based on traits."""
        # Build list of possible scale notes within range
        range_semitones = int(traits["range_octaves"] * 12)
        possible_notes = []
        
        for octave_offset in range(-1, 2):
            for interval in scale:
                note = root + (octave_offset * 12) + interval
                if abs(note - current) <= range_semitones:
                    possible_notes.append(note)
        
        if not possible_notes:
            possible_notes = [current]
        
        # Calculate probabilities based on traits
        probs = []
        for note in possible_notes:
            interval = abs(note - current)
            
            # Base probability
            prob = 1.0
            
            # Prefer or avoid certain intervals
            if interval in traits.get("preferred_intervals", []):
                prob *= 1.5
            if interval in traits.get("avoid_intervals", []):
                prob *= 0.3
            
            # Step vs leap preference
            if interval > 4:  # Leap
                prob *= traits["leap_probability"]
            else:  # Step
                prob *= (1 - traits["leap_probability"])
            
            # Ascending vs descending bias
            if note > current:
                prob *= traits["ascending_bias"]
            elif note < current:
                prob *= (1 - traits["ascending_bias"])
            
            probs.append(max(prob, 0.1))
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return np.random.choice(possible_notes, p=probs)
    
    def _apply_resolution(
        self,
        notes: List[int],
        root: int,
        scale: List[int],
        traits: Dict,
    ) -> List[int]:
        """Apply melodic resolution at the end if appropriate."""
        if len(notes) < 2:
            return notes
        
        resolution_strength = traits["resolution_strength"]
        
        if np.random.random() < resolution_strength:
            # Resolve to tonic, 3rd, or 5th
            resolution_notes = [
                root,
                root + scale[2] if len(scale) > 2 else root,
                root + scale[4] if len(scale) > 4 else root,
            ]
            
            # Choose closest resolution note
            last_note = notes[-1]
            closest = min(resolution_notes, key=lambda x: abs(x - last_note))
            notes[-1] = closest
        
        return notes
    
    def _generate_durations(self, length: int, traits: Dict) -> List[float]:
        """Generate note durations in beats."""
        durations = []
        
        # Common duration patterns
        patterns = [
            [1.0] * length,  # All quarter notes
            [0.5] * length,  # All eighth notes
            [1.0, 0.5, 0.5] * (length // 3 + 1),  # Quarter, eighth, eighth
            [0.5, 0.5, 1.0] * (length // 3 + 1),  # Eighth, eighth, quarter
        ]
        
        pattern = patterns[np.random.randint(len(patterns))]
        return pattern[:length]
    
    def _generate_velocities(self, length: int, emotion: str) -> List[int]:
        """Generate MIDI velocities based on emotion."""
        # Base velocity by emotion
        base_velocities = {
            "joy": 85,
            "grief": 60,
            "anger": 100,
            "fear": 55,
            "hope": 75,
            "longing": 65,
            "nostalgia": 70,
            "defiance": 95,
        }
        
        base = base_velocities.get(emotion.lower(), 75)
        
        # Add some variation
        velocities = []
        for i in range(length):
            # Slight accent on beat 1 and 3
            accent = 10 if i % 4 in [0, 2] else 0
            variation = np.random.randint(-8, 8)
            velocity = max(40, min(127, base + accent + variation))
            velocities.append(velocity)
        
        return velocities
    
    def _create_emotion_embedding(self, emotion: str) -> np.ndarray:
        """Create a simple emotion embedding vector."""
        # This is a placeholder - real implementation would use EmotionRecognizer
        embedding = np.zeros(64, dtype=np.float32)
        
        # Simple encoding based on emotion
        emotion_indices = {
            "joy": 0, "grief": 1, "anger": 2, "fear": 3,
            "hope": 4, "longing": 5, "nostalgia": 6, "defiance": 7,
        }
        
        idx = emotion_indices.get(emotion.lower(), 0)
        embedding[idx * 8:(idx + 1) * 8] = 1.0
        
        return embedding.reshape(1, -1)
    
    def _sample_notes_from_probs(
        self,
        probs: np.ndarray,
        length: int,
        key: str,
        mode: str,
    ) -> List[int]:
        """Sample notes from probability distribution."""
        # Get scale
        scale = self._get_scale(key, mode)
        config = MelodyConfig(key=key, mode=mode)
        root = config.get_root_midi()
        
        notes = []
        for i in range(length):
            # Get probabilities for this position
            if probs.ndim > 1 and i < probs.shape[1]:
                note_probs = probs[0, i]
            else:
                note_probs = probs.flatten()
            
            # Sample from probabilities
            note_idx = np.random.choice(len(note_probs), p=note_probs / note_probs.sum())
            
            # Map to scale note
            scale_degree = note_idx % len(scale)
            octave_offset = note_idx // len(scale) - 2
            note = root + (octave_offset * 12) + scale[scale_degree]
            notes.append(note)
        
        return notes


def create_melody_generator(config: Optional[MelodyConfig] = None) -> MLMelodyGenerator:
    """Factory function to create melody generator."""
    return MLMelodyGenerator(config)


# Convenience function
def generate_melody(
    emotion: str,
    key: str = "C",
    mode: str = "major",
    length: int = 8,
) -> GeneratedMelody:
    """Quick melody generation."""
    generator = MLMelodyGenerator()
    return generator.generate(emotion, length, key, mode)
