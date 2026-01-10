"""
Emotion Conditioned Generator - Generate audio/MIDI from emotional embeddings.

Uses emotion embeddings to condition generation of:
- Full audio tracks
- MIDI sequences
- Waveform segments

Combines the emotion pipeline with generative models for
end-to-end emotion-to-music generation.

Usage:
    from music_brain.generative import EmotionConditionedGenerator
    
    gen = EmotionConditionedGenerator(device="mps")
    audio = gen.generate(emotion="grief", intensity=0.8, duration=30)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

from .base import GenerativeModel, GenerativeConfig, GenerationResult


@dataclass
class EmotionConditionedConfig(GenerativeConfig):
    """Configuration for emotion-conditioned generator."""
    
    # Emotion embedding
    emotion_dim: int = 128
    num_emotions: int = 8
    
    # Generation
    default_duration: float = 30.0
    output_type: str = "audio"  # "audio", "midi", "both"
    
    # Audio settings
    sample_rate: int = 44100
    channels: int = 2
    
    # MIDI settings
    midi_ticks_per_beat: int = 480
    default_tempo: int = 120


# Emotion to musical parameter mappings
EMOTION_MAPPINGS = {
    "joy": {
        "valence": 0.9,
        "arousal": 0.8,
        "tempo_range": (110, 140),
        "key_preference": "major",
        "dynamics": "forte",
        "articulation": "staccato",
        "timbre": "bright",
    },
    "grief": {
        "valence": 0.1,
        "arousal": 0.3,
        "tempo_range": (50, 80),
        "key_preference": "minor",
        "dynamics": "piano",
        "articulation": "legato",
        "timbre": "dark",
    },
    "anger": {
        "valence": 0.2,
        "arousal": 0.95,
        "tempo_range": (120, 160),
        "key_preference": "minor",
        "dynamics": "fortissimo",
        "articulation": "marcato",
        "timbre": "aggressive",
    },
    "peace": {
        "valence": 0.7,
        "arousal": 0.2,
        "tempo_range": (60, 90),
        "key_preference": "major",
        "dynamics": "pianissimo",
        "articulation": "legato",
        "timbre": "warm",
    },
    "fear": {
        "valence": 0.2,
        "arousal": 0.85,
        "tempo_range": (80, 120),
        "key_preference": "diminished",
        "dynamics": "variable",
        "articulation": "tremolo",
        "timbre": "tense",
    },
    "love": {
        "valence": 0.85,
        "arousal": 0.5,
        "tempo_range": (70, 100),
        "key_preference": "major",
        "dynamics": "mezzo-piano",
        "articulation": "espressivo",
        "timbre": "warm",
    },
    "hope": {
        "valence": 0.8,
        "arousal": 0.6,
        "tempo_range": (90, 120),
        "key_preference": "major",
        "dynamics": "crescendo",
        "articulation": "building",
        "timbre": "bright",
    },
    "nostalgia": {
        "valence": 0.5,
        "arousal": 0.4,
        "tempo_range": (70, 95),
        "key_preference": "major_with_minor",
        "dynamics": "piano",
        "articulation": "rubato",
        "timbre": "vintage",
    },
}


class EmotionConditionedGenerator(GenerativeModel):
    """
    Generate music conditioned on emotional intent.
    
    Integrates:
    - Emotion recognition/embedding
    - Audio/MIDI generation
    - Production parameter mapping
    
    Example:
        gen = EmotionConditionedGenerator(device="mps")
        
        # Generate from emotion name
        result = gen.generate(emotion="grief", intensity=0.8)
        
        # Generate from embedding
        embedding = np.random.randn(128)
        result = gen.generate_from_embedding(embedding)
    """
    
    def __init__(
        self,
        device: str = "auto",
        config: Optional[EmotionConditionedConfig] = None,
    ):
        """Initialize emotion-conditioned generator."""
        if config is None:
            config = EmotionConditionedConfig(device=device)
        super().__init__(config)
        
        self.config: EmotionConditionedConfig = config
        self._emotion_encoder = None
        self._decoder = None
        self._midi_generator = None
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load model components.
        
        Args:
            path: Path to model directory or checkpoint
        """
        # Try to load emotion encoder
        self._load_emotion_encoder(path)
        
        # Try to load audio decoder
        self._load_decoder(path)
        
        self._is_loaded = True
    
    def _load_emotion_encoder(self, path: Optional[str]) -> None:
        """Load emotion embedding model."""
        try:
            # Try to import from penta_core
            from penta_core.ml.inference import load_model
            
            encoder_path = Path(path or "models/onnx") / "emotion_recognizer.onnx"
            if encoder_path.exists():
                self._emotion_encoder = load_model(str(encoder_path))
        except ImportError:
            pass
    
    def _load_decoder(self, path: Optional[str]) -> None:
        """Load audio decoder model."""
        try:
            from .audio_diffusion import AudioDiffusion
            self._decoder = AudioDiffusion(
                model_type="audioldm",
                device=self.config.get_device(),
            )
        except Exception:
            pass
    
    def generate(
        self,
        emotion: str = "peace",
        intensity: float = 0.5,
        duration: Optional[float] = None,
        tempo: Optional[int] = None,
        key: Optional[str] = None,
        output_type: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate music from emotional specification.
        
        Args:
            emotion: Emotion name (joy, grief, anger, etc.)
            intensity: Emotion intensity 0-1
            duration: Duration in seconds
            tempo: Override tempo
            key: Override key
            output_type: "audio", "midi", or "both"
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated content
        """
        if not self._is_loaded:
            self.load()
        
        # Get emotion parameters
        params = self._get_emotion_parameters(emotion, intensity)
        
        # Apply overrides
        if tempo:
            params["tempo"] = tempo
        if key:
            params["key"] = key
        
        duration = duration or self.config.default_duration
        output_type = output_type or self.config.output_type
        
        # Generate based on output type
        if output_type == "midi":
            output = self._generate_midi(params, duration)
        elif output_type == "audio":
            output = self._generate_audio(emotion, params, duration, **kwargs)
        else:  # "both"
            audio = self._generate_audio(emotion, params, duration, **kwargs)
            midi = self._generate_midi(params, duration)
            output = {"audio": audio, "midi": midi}
        
        return GenerationResult(
            output=output,
            duration_seconds=duration,
            sample_rate=self.config.sample_rate,
            model_name="EmotionConditionedGenerator",
            generation_params={
                "emotion": emotion,
                "intensity": intensity,
                "tempo": params.get("tempo"),
                "key": params.get("key"),
                "output_type": output_type,
            }
        )
    
    def generate_from_embedding(
        self,
        embedding: np.ndarray,
        duration: Optional[float] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate from a pre-computed emotion embedding.
        
        Args:
            embedding: Emotion embedding vector
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult
        """
        # Infer emotion from embedding (simplified - uses max component)
        emotion_names = list(EMOTION_MAPPINGS.keys())
        if len(embedding) >= len(emotion_names):
            emotion_idx = int(np.argmax(embedding[:len(emotion_names)]))
            emotion = emotion_names[emotion_idx]
            intensity = float(np.max(embedding[:len(emotion_names)]))
        else:
            emotion = "peace"
            intensity = 0.5
        
        return self.generate(
            emotion=emotion,
            intensity=intensity,
            duration=duration,
            **kwargs,
        )
    
    def _get_emotion_parameters(
        self,
        emotion: str,
        intensity: float,
    ) -> Dict[str, Any]:
        """Get musical parameters from emotion."""
        base = EMOTION_MAPPINGS.get(emotion.lower(), EMOTION_MAPPINGS["peace"])
        
        # Scale parameters by intensity
        tempo_min, tempo_max = base["tempo_range"]
        tempo = int(tempo_min + (tempo_max - tempo_min) * intensity)
        
        # Determine key
        key_pref = base["key_preference"]
        if key_pref == "major":
            key = np.random.choice(["C", "G", "D", "A", "F", "Bb", "Eb"])
        elif key_pref == "minor":
            key = np.random.choice(["Am", "Em", "Dm", "Gm", "Cm", "Fm"])
        elif key_pref == "major_with_minor":
            key = np.random.choice(["C", "Am", "G", "Em", "D", "Bm"])
        else:
            key = "Am"
        
        return {
            "emotion": emotion,
            "intensity": intensity,
            "valence": base["valence"],
            "arousal": base["arousal"],
            "tempo": tempo,
            "key": key,
            "dynamics": base["dynamics"],
            "articulation": base["articulation"],
            "timbre": base["timbre"],
        }
    
    def _generate_audio(
        self,
        emotion: str,
        params: Dict[str, Any],
        duration: float,
        **kwargs,
    ) -> np.ndarray:
        """Generate audio from emotion parameters."""
        # Build prompt for diffusion model
        prompt = self._build_audio_prompt(emotion, params)
        
        # Use audio diffusion if available
        if self._decoder is not None:
            try:
                result = self._decoder.generate(
                    prompt=prompt,
                    duration=duration,
                    emotion=emotion,
                    **kwargs,
                )
                return result.output
            except Exception:
                pass
        
        # Fallback: generate simple waveform
        return self._generate_fallback_audio(params, duration)
    
    def _build_audio_prompt(
        self,
        emotion: str,
        params: Dict[str, Any],
    ) -> str:
        """Build text prompt for audio generation."""
        tempo = params.get("tempo", 100)
        key = params.get("key", "C")
        dynamics = params.get("dynamics", "piano")
        timbre = params.get("timbre", "warm")
        
        # Map to descriptive terms
        tempo_desc = "slow" if tempo < 80 else "medium" if tempo < 120 else "fast"
        mode = "minor key" if "m" in key and len(key) <= 3 else "major key"
        
        return f"{emotion} music, {tempo_desc} tempo, {mode}, {dynamics}, {timbre} tone"
    
    def _generate_fallback_audio(
        self,
        params: Dict[str, Any],
        duration: float,
    ) -> np.ndarray:
        """Generate simple audio as fallback."""
        sr = self.config.sample_rate
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples)
        
        # Simple sine wave based on emotion
        base_freq = 220 if "m" in params.get("key", "C") else 261.63
        valence = params.get("valence", 0.5)
        
        # Generate harmonics
        audio = np.sin(2 * np.pi * base_freq * t)
        audio += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)  # Octave
        audio += 0.3 * np.sin(2 * np.pi * base_freq * 1.5 * t)  # Fifth
        
        # Apply envelope
        attack = int(0.1 * sr)
        release = int(0.2 * sr)
        envelope = np.ones(num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio *= envelope
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        
        # Stereo
        return np.stack([audio, audio])
    
    def _generate_midi(
        self,
        params: Dict[str, Any],
        duration: float,
    ) -> Dict:
        """Generate MIDI data from parameters."""
        tempo = params.get("tempo", 120)
        key = params.get("key", "C")
        
        # Calculate number of bars
        beats_per_bar = 4
        bars = int(duration * tempo / 60 / beats_per_bar)
        
        # Generate chord progression
        try:
            from .chord_generator import ChordProgressionGenerator
            chord_gen = ChordProgressionGenerator(device=self.config.get_device())
            chords = chord_gen.generate(
                emotion=params.get("emotion", "peace"),
                key=key,
                length=bars,
            )
        except ImportError:
            chords = ["C", "Am", "F", "G"] * (bars // 4 + 1)
            chords = chords[:bars]
        
        # Generate melody
        try:
            from music_brain.session.ml_melody_generator import MLMelodyGenerator
            melody_gen = MLMelodyGenerator(device=self.config.get_device())
            melody = melody_gen.generate(
                emotion=params.get("emotion", "peace"),
                key=key,
                length=bars * 4,
            )
        except ImportError:
            melody = self._generate_simple_melody(key, bars * 4)
        
        return {
            "tempo": tempo,
            "key": key,
            "time_signature": (4, 4),
            "chords": chords,
            "melody": melody,
            "duration_bars": bars,
        }
    
    def _generate_simple_melody(self, key: str, length: int) -> List[int]:
        """Generate simple melody as fallback."""
        from .music_utils import note_to_midi, get_scale_notes, is_minor_key, find_note_in_scale
        
        # Get root note and scale
        root_char = key[0] if key else "C"
        root = note_to_midi(root_char, octave=4)
        scale_type = "minor" if is_minor_key(key) else "major"
        notes = get_scale_notes(root_char, scale_type, octave=4, num_octaves=1)
        
        if not notes:
            notes = [root]
        
        # Generate random melody
        melody = []
        current = notes[0]
        for _ in range(length):
            # Mostly step motion, occasional leap
            if np.random.random() < 0.7:
                step = np.random.choice([-1, 1])
                current_idx = find_note_in_scale(current, notes)
                new_idx = max(0, min(len(notes) - 1, current_idx + step))
                current = notes[new_idx]
            else:
                current = int(np.random.choice(notes))
            melody.append(current)
        
        return melody
    
    def get_emotion_info(self, emotion: str) -> Dict[str, Any]:
        """Get information about an emotion's musical mappings."""
        if emotion.lower() in EMOTION_MAPPINGS:
            return EMOTION_MAPPINGS[emotion.lower()]
        return {"error": f"Unknown emotion: {emotion}"}
    
    def list_emotions(self) -> List[str]:
        """List all supported emotions."""
        return list(EMOTION_MAPPINGS.keys())
