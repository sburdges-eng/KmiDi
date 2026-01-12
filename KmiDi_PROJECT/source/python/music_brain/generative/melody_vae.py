"""
Melody VAE - Variational Autoencoder for melody generation.

Uses a VAE architecture to:
- Encode melodies into a latent space
- Sample new melodies from the latent space
- Interpolate between melodies
- Generate variations of a seed melody

Usage:
    from music_brain.generative import MelodyVAE
    
    vae = MelodyVAE(device="mps")
    melody = vae.generate(style="classical", length=32)
    variations = vae.generate_variations(seed_melody, num=4)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from .base import GenerativeModel, GenerativeConfig, GenerationResult


@dataclass
class MelodyVAEConfig(GenerativeConfig):
    """Configuration for Melody VAE."""
    
    # Model architecture
    input_dim: int = 128  # MIDI note range
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 2
    
    # Sequence settings
    max_sequence_length: int = 128
    min_note: int = 36  # C2
    max_note: int = 96  # C7
    
    # VAE parameters
    beta: float = 1.0  # KL divergence weight
    
    # Generation
    default_length: int = 32
    default_tempo: int = 120


# Style presets for melody generation
STYLE_PRESETS = {
    "classical": {
        "note_density": 0.6,
        "leap_probability": 0.3,
        "rest_probability": 0.1,
        "octave_range": 2,
        "rhythm_variety": 0.4,
    },
    "pop": {
        "note_density": 0.8,
        "leap_probability": 0.4,
        "rest_probability": 0.15,
        "octave_range": 1.5,
        "rhythm_variety": 0.3,
    },
    "jazz": {
        "note_density": 0.7,
        "leap_probability": 0.5,
        "rest_probability": 0.2,
        "octave_range": 2.5,
        "rhythm_variety": 0.8,
    },
    "ambient": {
        "note_density": 0.3,
        "leap_probability": 0.6,
        "rest_probability": 0.4,
        "octave_range": 3,
        "rhythm_variety": 0.6,
    },
    "folk": {
        "note_density": 0.65,
        "leap_probability": 0.25,
        "rest_probability": 0.12,
        "octave_range": 1.5,
        "rhythm_variety": 0.35,
    },
}


class MelodyVAE(GenerativeModel):
    """
    Variational Autoencoder for melody generation.
    
    Features:
    - Encode existing melodies to latent space
    - Generate new melodies from random samples
    - Interpolate between melodies
    - Generate variations
    - Style-conditioned generation
    
    Example:
        vae = MelodyVAE(device="mps")
        
        # Generate new melody
        melody = vae.generate(style="jazz", length=32)
        
        # Create variations
        variations = vae.generate_variations(melody, num=4)
        
        # Interpolate between two melodies
        interpolated = vae.interpolate(melody1, melody2, steps=5)
    """
    
    def __init__(
        self,
        device: str = "auto",
        config: Optional[MelodyVAEConfig] = None,
    ):
        """Initialize Melody VAE."""
        if config is None:
            config = MelodyVAEConfig(device=device)
        super().__init__(config)
        
        self.config: MelodyVAEConfig = config
        self._encoder = None
        self._decoder = None
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load VAE model weights.
        
        Args:
            path: Path to model checkpoint
        """
        if path is None:
            default_paths = [
                Path("models/melody_vae.onnx"),
                Path("models/melody_vae.pt"),
                Path("checkpoints/melody_vae/model.pt"),
            ]
            for p in default_paths:
                if p.exists():
                    path = str(p)
                    break
        
        if path and Path(path).exists():
            self._load_model(path)
        
        # Always mark as loaded (can use fallback generation)
        self._is_loaded = True
    
    def _load_model(self, path: str) -> None:
        """Load PyTorch or ONNX model."""
        try:
            import torch
            
            if path.endswith(".onnx"):
                import onnxruntime as ort
                self._model = ort.InferenceSession(path)
            else:
                checkpoint = torch.load(path, map_location=self.config.get_device())
                if isinstance(checkpoint, dict):
                    self._encoder = checkpoint.get("encoder")
                    self._decoder = checkpoint.get("decoder")
                else:
                    self._model = checkpoint
                    
                if self._model and hasattr(self._model, "eval"):
                    self._model.eval()
        except Exception as e:
            print(f"Warning: Could not load VAE model: {e}")
    
    def generate(
        self,
        style: Optional[str] = None,
        key: str = "C",
        scale: str = "major",
        length: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate a new melody.
        
        Args:
            style: Style preset (classical, pop, jazz, ambient, folk)
            key: Key signature
            scale: Scale type (major, minor, dorian, etc.)
            length: Number of notes
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            List of note dictionaries with pitch, duration, velocity
        """
        if not self._is_loaded:
            self.load()
        
        self._set_seed(seed)
        
        length = length or self.config.default_length
        style_params = STYLE_PRESETS.get(style, STYLE_PRESETS["pop"])
        
        # Get scale notes
        scale_notes = self._get_scale_notes(key, scale)
        
        # Try neural generation first
        if self._model is not None or (self._encoder and self._decoder):
            return self._generate_neural(scale_notes, length, style_params)
        
        # Fallback to algorithmic generation
        return self._generate_algorithmic(scale_notes, length, style_params)
    
    def _generate_neural(
        self,
        scale_notes: List[int],
        length: int,
        style_params: Dict,
    ) -> List[Dict]:
        """Generate using neural network."""
        import torch
        
        # Sample from latent space
        z = torch.randn(1, self.config.latent_dim, device=self.config.get_device())
        
        # Decode
        if self._decoder:
            with torch.no_grad():
                output = self._decoder(z)
        elif self._model:
            with torch.no_grad():
                if hasattr(self._model, "decode"):
                    output = self._model.decode(z)
                else:
                    output = self._model(z)
        
        # Convert to note list
        notes = self._tensor_to_notes(output, scale_notes, length)
        return notes
    
    def _generate_algorithmic(
        self,
        scale_notes: List[int],
        length: int,
        style_params: Dict,
    ) -> List[Dict]:
        """Generate using algorithmic approach."""
        notes = []
        
        # Start on a scale degree
        current_note = scale_notes[0]
        
        for i in range(length):
            # Decide if rest
            if np.random.random() < style_params["rest_probability"]:
                notes.append({
                    "pitch": -1,  # Rest
                    "duration": self._get_random_duration(style_params),
                    "velocity": 0,
                    "position": i,
                })
                continue
            
            # Decide motion type
            if np.random.random() < style_params["leap_probability"]:
                # Leap
                interval = np.random.choice([-5, -4, -3, 3, 4, 5, 7, 8])
                idx = scale_notes.index(current_note) if current_note in scale_notes else 0
                new_idx = (idx + interval) % len(scale_notes)
                
                # Adjust octave
                octave_shift = interval // len(scale_notes) * 12
                current_note = scale_notes[new_idx] + octave_shift
            else:
                # Step motion
                idx = scale_notes.index(current_note) if current_note in scale_notes else 0
                step = np.random.choice([-2, -1, 0, 1, 1, 2])  # Bias upward
                new_idx = max(0, min(len(scale_notes) - 1, idx + step))
                current_note = scale_notes[new_idx]
            
            # Ensure within range
            while current_note < self.config.min_note:
                current_note += 12
            while current_note > self.config.max_note:
                current_note -= 12
            
            # Add note
            notes.append({
                "pitch": int(current_note),
                "duration": self._get_random_duration(style_params),
                "velocity": int(64 + np.random.randint(-20, 30)),
                "position": i,
            })
        
        return notes
    
    def _get_random_duration(self, style_params: Dict) -> float:
        """Get a random duration based on style."""
        variety = style_params.get("rhythm_variety", 0.5)
        
        if variety < 0.3:
            # Simple rhythms
            durations = [0.5, 1.0]
            weights = [0.3, 0.7]
        elif variety < 0.6:
            # Medium variety
            durations = [0.25, 0.5, 1.0, 2.0]
            weights = [0.2, 0.4, 0.3, 0.1]
        else:
            # High variety
            durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            weights = [0.15, 0.25, 0.1, 0.25, 0.1, 0.15]
        
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        return float(np.random.choice(durations, p=weights))
    
    def _get_scale_notes(self, key: str, scale: str) -> List[int]:
        """Get MIDI notes for a scale."""
        # Base note
        note_map = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
        root = key[0].upper()
        base = note_map.get(root, 60)
        
        # Adjust for sharps/flats
        if len(key) > 1:
            if key[1] == "#":
                base += 1
            elif key[1] == "b":
                base -= 1
        
        # Scale intervals
        scale_intervals = {
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
        
        intervals = scale_intervals.get(scale.lower(), scale_intervals["major"])
        
        # Generate notes across octaves
        notes = []
        for octave in range(-2, 3):
            for interval in intervals:
                note = base + octave * 12 + interval
                if self.config.min_note <= note <= self.config.max_note:
                    notes.append(note)
        
        return sorted(set(notes))
    
    def _tensor_to_notes(
        self,
        tensor,
        scale_notes: List[int],
        length: int,
    ) -> List[Dict]:
        """Convert model output tensor to note list."""
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = np.array(tensor)
        
        arr = arr.flatten()[:length * 3]  # pitch, duration, velocity
        
        notes = []
        for i in range(min(length, len(arr) // 3)):
            pitch_idx = int(arr[i * 3] * len(scale_notes)) % len(scale_notes)
            duration = max(0.25, float(arr[i * 3 + 1]) * 2)
            velocity = int(40 + abs(arr[i * 3 + 2]) * 80)
            
            notes.append({
                "pitch": scale_notes[pitch_idx],
                "duration": duration,
                "velocity": min(127, velocity),
                "position": i,
            })
        
        return notes
    
    def encode(self, melody: List[Dict]) -> np.ndarray:
        """
        Encode a melody to latent space.
        
        Args:
            melody: List of note dictionaries
            
        Returns:
            Latent vector
        """
        if not self._is_loaded:
            self.load()
        
        # Convert melody to tensor
        tensor = self._notes_to_tensor(melody)
        
        if self._encoder:
            import torch
            with torch.no_grad():
                z = self._encoder(tensor)
                return z.numpy()
        
        # Fallback: simple feature extraction
        pitches = [n["pitch"] for n in melody if n["pitch"] > 0]
        if not pitches:
            return np.zeros(self.config.latent_dim)
        
        features = np.array([
            np.mean(pitches) / 127,
            np.std(pitches) / 12,
            len(pitches) / self.config.max_sequence_length,
            np.mean([n.get("velocity", 64) for n in melody]) / 127,
        ])
        
        # Pad to latent dim
        z = np.zeros(self.config.latent_dim)
        z[:len(features)] = features
        return z
    
    def decode(self, z: np.ndarray, length: Optional[int] = None) -> List[Dict]:
        """
        Decode a latent vector to melody.
        
        Args:
            z: Latent vector
            length: Output length
            
        Returns:
            List of note dictionaries
        """
        if not self._is_loaded:
            self.load()
        
        length = length or self.config.default_length
        
        if self._decoder:
            import torch
            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = self._decoder(z_tensor)
            return self._tensor_to_notes(
                output, 
                self._get_scale_notes("C", "major"),
                length
            )
        
        # Fallback: use z to seed generation
        seed = int(abs(np.sum(z) * 1000)) % (2**31)
        return self.generate(seed=seed, length=length)
    
    def _notes_to_tensor(self, melody: List[Dict]):
        """Convert note list to tensor."""
        import torch
        
        # Create feature matrix
        features = np.zeros((len(melody), 3))
        for i, note in enumerate(melody):
            features[i, 0] = note.get("pitch", 60) / 127
            features[i, 1] = note.get("duration", 1.0) / 4
            features[i, 2] = note.get("velocity", 64) / 127
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def interpolate(
        self,
        melody1: List[Dict],
        melody2: List[Dict],
        steps: int = 5,
    ) -> List[List[Dict]]:
        """
        Interpolate between two melodies in latent space.
        
        Args:
            melody1: First melody
            melody2: Second melody
            steps: Number of interpolation steps
            
        Returns:
            List of interpolated melodies
        """
        z1 = self.encode(melody1)
        z2 = self.encode(melody2)
        
        melodies = []
        for i in range(steps):
            alpha = i / (steps - 1) if steps > 1 else 0
            z_interp = (1 - alpha) * z1 + alpha * z2
            melodies.append(self.decode(z_interp))
        
        return melodies
    
    def generate_variations(
        self,
        melody: List[Dict],
        num: int = 4,
        variation_amount: float = 0.3,
    ) -> List[List[Dict]]:
        """
        Generate variations of a melody.
        
        Args:
            melody: Seed melody
            num: Number of variations
            variation_amount: Amount of variation (0-1)
            
        Returns:
            List of melody variations
        """
        z = self.encode(melody)
        length = len(melody)
        
        variations = []
        for _ in range(num):
            # Add noise to latent
            noise = np.random.randn(len(z)) * variation_amount
            z_varied = z + noise
            variations.append(self.decode(z_varied, length))
        
        return variations
