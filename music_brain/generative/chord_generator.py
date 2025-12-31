"""
Chord Progression Generator - Transformer-based chord sequence generation.

Uses a GPT-like architecture to generate chord progressions that match
emotional intent and musical context. Can be conditioned on:
- Emotion embeddings
- Genre/style
- Key/scale constraints
- Seed chords

Usage:
    from music_brain.generative import ChordProgressionGenerator
    
    gen = ChordProgressionGenerator(device="mps")
    chords = gen.generate(
        emotion="grief",
        key="Am",
        length=8,
        seed_chords=["Am", "F"]
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import json

from .base import GenerativeModel, GenerativeConfig, GenerationResult


# Chord vocabulary
CHORD_VOCAB = [
    # Major chords
    "C", "D", "E", "F", "G", "A", "B",
    "Db", "Eb", "Gb", "Ab", "Bb",
    # Minor chords
    "Cm", "Dm", "Em", "Fm", "Gm", "Am", "Bm",
    "Dbm", "Ebm", "Gbm", "Abm", "Bbm",
    # 7th chords
    "C7", "D7", "E7", "F7", "G7", "A7", "B7",
    "Cmaj7", "Dmaj7", "Emaj7", "Fmaj7", "Gmaj7", "Amaj7", "Bmaj7",
    "Cm7", "Dm7", "Em7", "Fm7", "Gm7", "Am7", "Bm7",
    # Diminished/Augmented
    "Cdim", "Ddim", "Edim", "Fdim", "Gdim", "Adim", "Bdim",
    "Caug", "Daug", "Eaug", "Faug", "Gaug", "Aaug", "Baug",
    # Suspended
    "Csus2", "Dsus2", "Esus2", "Fsus2", "Gsus2", "Asus2", "Bsus2",
    "Csus4", "Dsus4", "Esus4", "Fsus4", "Gsus4", "Asus4", "Bsus4",
    # Special tokens
    "<PAD>", "<START>", "<END>", "<REST>",
]

CHORD_TO_ID = {chord: i for i, chord in enumerate(CHORD_VOCAB)}
ID_TO_CHORD = {i: chord for i, chord in enumerate(CHORD_VOCAB)}


# Common progressions by emotion (for fallback/initialization)
EMOTION_PROGRESSIONS = {
    "joy": [
        ["C", "G", "Am", "F"],  # Classic pop
        ["G", "D", "Em", "C"],  # Uplifting
        ["A", "E", "F#m", "D"],  # Bright
    ],
    "grief": [
        ["Am", "F", "C", "G"],  # Melancholic
        ["Em", "C", "G", "D"],  # Wistful
        ["Dm", "Bb", "F", "C"],  # Dark
        ["Am", "Em", "Dm", "Am"],  # Unresolved
    ],
    "anger": [
        ["Am", "Am", "F", "E"],  # Aggressive
        ["Em", "C", "D", "D"],  # Building tension
        ["Dm", "Dm", "A", "Dm"],  # Intense
    ],
    "peace": [
        ["Cmaj7", "Fmaj7", "Cmaj7", "Fmaj7"],  # Calm
        ["Gmaj7", "Cmaj7", "Dm7", "Gmaj7"],  # Serene
        ["Fmaj7", "Am7", "Dm7", "Gmaj7"],  # Ambient
    ],
    "love": [
        ["C", "Am", "F", "G"],  # Romantic
        ["D", "Bm", "G", "A"],  # Tender
        ["F", "Dm", "Bb", "C"],  # Warm
    ],
    "hope": [
        ["C", "G", "F", "C"],  # Building
        ["Am", "F", "C", "G"],  # Rising
        ["G", "C", "D", "G"],  # Triumphant
    ],
    "nostalgia": [
        ["Am", "C", "G", "F"],  # Bittersweet
        ["Em", "G", "D", "C"],  # Wistful
        ["Dm7", "G7", "Cmaj7", "Am7"],  # Jazz-tinged
    ],
}


@dataclass
class ChordGeneratorConfig(GenerativeConfig):
    """Configuration for chord progression generator."""
    
    # Model architecture
    vocab_size: int = len(CHORD_VOCAB)
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    hidden_dim: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 64
    
    # Generation
    default_length: int = 8
    use_rule_based_fallback: bool = True


class ChordProgressionGenerator(GenerativeModel):
    """
    Generate chord progressions using a transformer model.
    
    Supports:
    - Emotion-conditioned generation
    - Genre/style conditioning
    - Key constraints
    - Seed chord continuation
    - Rule-based fallback when no model available
    
    Example:
        gen = ChordProgressionGenerator(device="mps")
        chords = gen.generate(
            emotion="grief",
            key="Am",
            length=8,
            seed_chords=["Am", "F"]
        )
    """
    
    def __init__(
        self,
        device: str = "auto",
        config: Optional[ChordGeneratorConfig] = None,
    ):
        """Initialize chord progression generator."""
        if config is None:
            config = ChordGeneratorConfig(device=device)
        super().__init__(config)
        
        self.config: ChordGeneratorConfig = config
        self._tokenizer = ChordTokenizer()
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load trained chord progression model.
        
        Args:
            path: Path to model checkpoint
        """
        if path is None:
            # Check default locations
            default_paths = [
                Path("models/chord_generator.onnx"),
                Path("models/chord_generator.pt"),
                Path("checkpoints/chord_generator/model.pt"),
            ]
            for p in default_paths:
                if p.exists():
                    path = str(p)
                    break
        
        if path and Path(path).exists():
            self._load_model(path)
            self._is_loaded = True
        else:
            # Use rule-based fallback
            self._is_loaded = True  # Mark as "loaded" with fallback
    
    def _load_model(self, path: str) -> None:
        """Load PyTorch or ONNX model."""
        try:
            import torch
            
            if path.endswith(".onnx"):
                import onnxruntime as ort
                self._model = ort.InferenceSession(path)
            else:
                self._model = torch.load(path, map_location=self.config.get_device())
                if hasattr(self._model, "eval"):
                    self._model.eval()
        except Exception as e:
            print(f"Warning: Could not load model: {e}. Using rule-based fallback.")
    
    def generate(
        self,
        emotion: Optional[str] = None,
        key: Optional[str] = None,
        length: int = 8,
        seed_chords: Optional[List[str]] = None,
        genre: Optional[str] = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate a chord progression.
        
        Args:
            emotion: Emotional character (joy, grief, anger, etc.)
            key: Key constraint (C, Am, etc.)
            length: Number of chords to generate
            seed_chords: Initial chords to continue from
            genre: Genre/style modifier
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            List of chord symbols
        """
        if not self._is_loaded:
            self.load()
        
        # Use neural model if available
        if self._model is not None and not self.config.use_rule_based_fallback:
            return self._generate_neural(
                emotion=emotion,
                key=key,
                length=length,
                seed_chords=seed_chords,
                temperature=temperature,
            )
        
        # Rule-based generation
        return self._generate_rule_based(
            emotion=emotion,
            key=key,
            length=length,
            seed_chords=seed_chords,
            genre=genre,
        )
    
    def _generate_neural(
        self,
        emotion: Optional[str],
        key: Optional[str],
        length: int,
        seed_chords: Optional[List[str]],
        temperature: float,
    ) -> List[str]:
        """Generate using the neural network model."""
        import torch
        
        # Tokenize seed
        if seed_chords:
            tokens = [self._tokenizer.encode(c) for c in seed_chords]
        else:
            tokens = [CHORD_TO_ID["<START>"]]
        
        # Get emotion embedding if available
        emotion_id = self._get_emotion_id(emotion) if emotion else 0
        
        # Auto-regressive generation
        with torch.no_grad():
            for _ in range(length - len(tokens) + 1):
                input_tensor = torch.tensor([tokens], device=self.config.get_device())
                
                if hasattr(self._model, "forward"):
                    # PyTorch model
                    logits = self._model(input_tensor, emotion_id=emotion_id)
                else:
                    # ONNX model
                    outputs = self._model.run(
                        None, 
                        {"input_ids": input_tensor.numpy()}
                    )
                    logits = torch.tensor(outputs[0])
                
                # Sample next token
                logits = logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == CHORD_TO_ID["<END>"]:
                    break
                    
                tokens.append(next_token)
        
        # Decode tokens
        chords = [self._tokenizer.decode(t) for t in tokens]
        chords = [c for c in chords if c not in ["<START>", "<END>", "<PAD>", "<REST>"]]
        
        # Transpose to key if specified
        if key:
            chords = self._transpose_to_key(chords, key)
        
        return chords[:length]
    
    def _generate_rule_based(
        self,
        emotion: Optional[str],
        key: Optional[str],
        length: int,
        seed_chords: Optional[List[str]],
        genre: Optional[str],
    ) -> List[str]:
        """Generate using rule-based approach."""
        # Get base progression from emotion
        emotion = emotion or "peace"
        base_progressions = EMOTION_PROGRESSIONS.get(
            emotion.lower(), 
            EMOTION_PROGRESSIONS["peace"]
        )
        
        # Select a random base progression
        base_idx = int(np.random.choice(np.arange(len(base_progressions))))
        progression = list(base_progressions[base_idx])
        
        # If seed chords provided, try to incorporate them
        if seed_chords:
            progression = list(seed_chords) + progression[len(seed_chords):]
        
        # Extend or truncate to desired length
        while len(progression) < length:
            progression.extend(base_progressions[np.random.randint(len(base_progressions))])
        progression = progression[:length]
        
        # Transpose to key if specified
        if key:
            progression = self._transpose_to_key(progression, key)
        
        return progression
    
    def _transpose_to_key(self, chords: List[str], target_key: str) -> List[str]:
        """Transpose chord progression to target key."""
        # Return early if chords is empty
        if not chords:
            return chords
        
        # Simple transposition logic
        # Detect current key (assume first chord)
        current_root = chords[0][0]
        if len(chords[0]) > 1 and chords[0][1] in "b#":
            current_root = chords[0][:2]
        
        # Get target root
        target_root = target_key[0]
        if len(target_key) > 1 and target_key[1] in "b#":
            target_root = target_key[:2]
        
        # Calculate semitone difference
        notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
        try:
            current_idx = notes.index(current_root)
            target_idx = notes.index(target_root)
            semitones = (target_idx - current_idx) % 12
        except ValueError:
            return chords  # Can't transpose, return as-is
        
        if semitones == 0:
            return chords
        
        # Transpose each chord
        transposed = []
        for chord in chords:
            transposed.append(self._transpose_chord(chord, semitones, notes))
        
        return transposed
    
    def _transpose_chord(self, chord: str, semitones: int, notes: List[str]) -> str:
        """Transpose a single chord by semitones."""
        # Extract root
        if len(chord) > 1 and chord[1] in "b#":
            root = chord[:2]
            suffix = chord[2:]
        else:
            root = chord[0]
            suffix = chord[1:]
        
        # Handle enharmonic equivalents
        enharmonic = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}
        root = enharmonic.get(root, root)
        
        try:
            idx = notes.index(root)
            new_idx = (idx + semitones) % 12
            new_root = notes[new_idx]
            return new_root + suffix
        except ValueError:
            return chord
    
    def _get_emotion_id(self, emotion: str) -> int:
        """Map emotion to ID for conditioning."""
        emotions = ["joy", "grief", "anger", "peace", "fear", "love", "hope", "nostalgia"]
        try:
            return emotions.index(emotion.lower())
        except ValueError:
            return 0
    
    def analyze_progression(self, chords: List[str]) -> Dict:
        """
        Analyze a chord progression.
        
        Args:
            chords: List of chord symbols
            
        Returns:
            Analysis dictionary with key, mode, tension, etc.
        """
        if not chords:
            return {"error": "Empty progression"}
        
        # Simple analysis
        has_minor = any("m" in c and "maj" not in c for c in chords)
        has_7th = any("7" in c for c in chords)
        has_dim = any("dim" in c for c in chords)
        has_aug = any("aug" in c for c in chords)
        
        # Estimate key from first chord
        first_chord = chords[0]
        root = first_chord[0]
        if len(first_chord) > 1 and first_chord[1] in "b#":
            root = first_chord[:2]
        
        is_minor = "m" in first_chord and "maj" not in first_chord
        
        return {
            "estimated_key": f"{root}{'m' if is_minor else ''}",
            "mode": "minor" if is_minor else "major",
            "length": len(chords),
            "has_7th_chords": has_7th,
            "has_diminished": has_dim,
            "has_augmented": has_aug,
            "tension_level": "high" if (has_dim or has_aug) else ("medium" if has_7th else "low"),
            "chord_diversity": len(set(chords)) / len(chords) if chords else 0,
        }


class ChordTokenizer:
    """Simple tokenizer for chord symbols."""
    
    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab or CHORD_TO_ID
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, chord: str) -> int:
        """Encode chord to token ID."""
        return self.vocab.get(chord, self.vocab.get("<PAD>", 0))
    
    def decode(self, token_id: int) -> str:
        """Decode token ID to chord."""
        return self.reverse_vocab.get(token_id, "<PAD>")
    
    def encode_sequence(self, chords: List[str]) -> List[int]:
        """Encode sequence of chords."""
        return [self.encode(c) for c in chords]
    
    def decode_sequence(self, token_ids: List[int]) -> List[str]:
        """Decode sequence of token IDs."""
        return [self.decode(t) for t in token_ids]
