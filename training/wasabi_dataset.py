"""
WASABI Dataset Loader for Music Training.

WASABI (Web Audio Semantic Annotation and Browsing Interface) is a large-scale
music dataset containing:
- 2 million popular commercially released songs
- Lyrics with emotional content, topics, structure segmentation
- Audio analysis: chords and sound characteristics
- Cultural metadata: artists, albums, release dates, producers

Dataset access:
- REST API: https://wasabi.i3s.unice.fr/api/
- SPARQL endpoint: https://wasabi.i3s.unice.fr/sparql
- GitHub: https://github.com/micbuffa/WasabiDataset

This loader supports:
- Loading from local WASABI JSON/JSONL files
- Filtering by emotion, genre, year, artist
- Extracting lyrics, audio features, and metadata
- Integration with emotion-conditioned music generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class WasabiSample:
    """Represents a single WASABI song sample."""
    
    # Core identifiers
    song_id: str
    title: str
    artist: Optional[str] = None
    album: Optional[str] = None
    
    # Lyrics and text
    lyrics: Optional[str] = None
    lyrics_structure: Optional[Dict[str, Any]] = None  # verse, chorus, bridge, etc.
    lyrics_topics: Optional[List[str]] = None
    lyrics_emotion: Optional[Dict[str, float]] = None  # emotion scores
    explicitness: Optional[str] = None  # "explicit", "clean", etc.
    salient_passages: Optional[List[str]] = None
    
    # Audio features
    audio_features: Optional[Dict[str, Any]] = None  # chords, tempo, key, etc.
    chords: Optional[List[str]] = None
    tempo: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None  # major/minor
    
    # Cultural metadata
    release_date: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[List[str]] = None
    producer: Optional[List[str]] = None
    
    # File paths (if audio files are available locally)
    audio_path: Optional[Path] = None
    midi_path: Optional[Path] = None
    
    # Computed features (cached)
    _emotion_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    _chord_embedding: Optional[np.ndarray] = field(default=None, repr=False)


class WasabiDataset(Dataset):
    """
    PyTorch Dataset for WASABI music corpus.
    
    Supports loading from:
    1. Local JSON/JSONL files (one song per line or array)
    2. WASABI REST API (if configured)
    3. Preprocessed manifest files
    
    Usage:
        dataset = WasabiDataset(
            manifest_path="data/wasabi/train.jsonl",
            emotion_filter=["happy", "sad"],
            min_year=2000,
        )
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        emotion_filter: Optional[List[str]] = None,
        genre_filter: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        artist_filter: Optional[List[str]] = None,
        require_lyrics: bool = True,
        require_audio_features: bool = False,
        max_samples: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize WASABI dataset.
        
        Args:
            manifest_path: Path to JSON/JSONL file with WASABI data
            emotion_filter: Only include songs with these emotions
            genre_filter: Only include songs from these genres
            year_range: (min_year, max_year) to filter by
            artist_filter: Only include songs from these artists
            require_lyrics: Skip songs without lyrics
            require_audio_features: Skip songs without audio features
            max_samples: Maximum number of samples to load
            cache_dir: Directory to cache processed features
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        self.emotion_filter = set(emotion_filter) if emotion_filter else None
        self.genre_filter = set(genre_filter) if genre_filter else None
        self.year_range = year_range
        self.artist_filter = set(artist_filter) if artist_filter else None
        self.require_lyrics = require_lyrics
        self.require_audio_features = require_audio_features
        self.max_samples = max_samples
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and filter samples
        self.samples: List[WasabiSample] = []
        self._load_samples()
        
        logger.info(
            f"WasabiDataset initialized: {len(self.samples)} samples "
            f"(filtered from manifest)"
        )
    
    def _load_samples(self) -> None:
        """Load samples from manifest file."""
        samples_raw = []
        
        # Determine file format
        if self.manifest_path.suffix == ".jsonl":
            # JSONL: one JSON object per line
            with open(self.manifest_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        samples_raw.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSONL line {line_num}: {e}")
                        continue
        else:
            # JSON: array or single object
            with open(self.manifest_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples_raw = data
                elif isinstance(data, dict) and "samples" in data:
                    samples_raw = data["samples"]
                elif isinstance(data, dict):
                    samples_raw = [data]
                else:
                    raise ValueError(f"Unexpected JSON structure in {self.manifest_path}")
        
        # Convert to WasabiSample objects and filter
        for obj in samples_raw:
            sample = self._parse_sample(obj)
            if sample and self._should_include(sample):
                self.samples.append(sample)
                
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
    
    def _parse_sample(self, obj: Dict[str, Any]) -> Optional[WasabiSample]:
        """Parse a raw JSON object into a WasabiSample."""
        try:
            # Extract core fields
            song_id = obj.get("id") or obj.get("song_id") or obj.get("track_id", "")
            title = obj.get("title") or obj.get("name", "")
            
            # Lyrics
            lyrics = obj.get("lyrics")
            lyrics_structure = obj.get("lyrics_structure") or obj.get("structure")
            lyrics_topics = obj.get("lyrics_topics") or obj.get("topics", [])
            lyrics_emotion = obj.get("lyrics_emotion") or obj.get("emotion", {})
            explicitness = obj.get("explicitness") or obj.get("explicit")
            salient_passages = obj.get("salient_passages") or obj.get("salient", [])
            
            # Audio features
            audio_features = obj.get("audio_features") or obj.get("audio", {})
            chords = obj.get("chords") or audio_features.get("chords", [])
            tempo = obj.get("tempo") or audio_features.get("tempo")
            key = obj.get("key") or audio_features.get("key")
            mode = obj.get("mode") or audio_features.get("mode")
            
            # Cultural metadata
            artist = obj.get("artist") or obj.get("artist_name")
            album = obj.get("album") or obj.get("album_name")
            release_date = obj.get("release_date") or obj.get("date")
            year = obj.get("year") or obj.get("release_year")
            if year and isinstance(year, str):
                try:
                    year = int(year[:4])  # Extract year from date string
                except ValueError:
                    year = None
            
            genre = obj.get("genre") or obj.get("genres", [])
            if isinstance(genre, str):
                genre = [genre]
            
            producer = obj.get("producer") or obj.get("producers", [])
            if isinstance(producer, str):
                producer = [producer]
            
            # File paths
            audio_path = obj.get("audio_path") or obj.get("audio_file")
            midi_path = obj.get("midi_path") or obj.get("midi_file")
            
            return WasabiSample(
                song_id=str(song_id),
                title=str(title),
                artist=artist,
                album=album,
                lyrics=lyrics,
                lyrics_structure=lyrics_structure,
                lyrics_topics=lyrics_topics,
                lyrics_emotion=lyrics_emotion if isinstance(lyrics_emotion, dict) else {},
                explicitness=explicitness,
                salient_passages=salient_passages,
                audio_features=audio_features,
                chords=chords if isinstance(chords, list) else [],
                tempo=float(tempo) if tempo else None,
                key=key,
                mode=mode,
                release_date=release_date,
                year=year,
                genre=genre if isinstance(genre, list) else [],
                producer=producer if isinstance(producer, list) else [],
                audio_path=Path(audio_path) if audio_path else None,
                midi_path=Path(midi_path) if midi_path else None,
            )
        except Exception as e:
            logger.debug(f"Failed to parse sample: {e}")
            return None
    
    def _should_include(self, sample: WasabiSample) -> bool:
        """Check if sample should be included based on filters."""
        # Require lyrics if specified
        if self.require_lyrics and not sample.lyrics:
            return False
        
        # Require audio features if specified
        if self.require_audio_features and not sample.audio_features:
            return False
        
        # Emotion filter
        if self.emotion_filter:
            sample_emotions = set(sample.lyrics_emotion.keys()) if sample.lyrics_emotion else set()
            if not self.emotion_filter.intersection(sample_emotions):
                return False
        
        # Genre filter
        if self.genre_filter and sample.genre:
            if not self.genre_filter.intersection(set(sample.genre)):
                return False
        
        # Year range filter
        if self.year_range and sample.year:
            min_year, max_year = self.year_range
            if not (min_year <= sample.year <= max_year):
                return False
        
        # Artist filter
        if self.artist_filter and sample.artist:
            if sample.artist not in self.artist_filter:
                return False
        
        return True
    
    def _compute_emotion_embedding(self, sample: WasabiSample) -> np.ndarray:
        """Compute emotion embedding from lyrics_emotion dict."""
        # Standard emotion vocabulary (can be extended)
        emotion_vocab = [
            "happy", "sad", "angry", "fear", "surprise", "disgust",
            "joy", "sorrow", "love", "hate", "excitement", "calm",
            "anxiety", "peace", "energy", "melancholy"
        ]
        
        embedding = np.zeros(len(emotion_vocab), dtype=np.float32)
        
        if sample.lyrics_emotion:
            for i, emotion in enumerate(emotion_vocab):
                # Check for variations
                for key, value in sample.lyrics_emotion.items():
                    key_lower = key.lower()
                    if emotion in key_lower or key_lower in emotion:
                        embedding[i] = float(value)
                        break
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_chord_embedding(self, sample: WasabiSample) -> np.ndarray:
        """Compute chord progression embedding."""
        if not sample.chords:
            return np.zeros(64, dtype=np.float32)  # Default size
        
        # Simple one-hot encoding of chord roots and qualities
        chord_roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        chord_qualities = ["major", "minor", "diminished", "augmented", "sus2", "sus4"]
        
        embedding = np.zeros(len(chord_roots) * len(chord_qualities), dtype=np.float32)
        
        for chord_str in sample.chords[:16]:  # Limit to first 16 chords
            chord_str = chord_str.upper().strip()
            
            # Parse chord (simplified)
            root = None
            quality = "major"  # Default
            
            for i, root_name in enumerate(chord_roots):
                if chord_str.startswith(root_name):
                    root = i
                    break
            
            if root is None:
                continue
            
            # Detect quality
            if "m" in chord_str or "MIN" in chord_str:
                quality = "minor"
            elif "dim" in chord_str:
                quality = "diminished"
            elif "aug" in chord_str or "+" in chord_str:
                quality = "augmented"
            elif "sus2" in chord_str:
                quality = "sus2"
            elif "sus4" in chord_str or "sus" in chord_str:
                quality = "sus4"
            
            quality_idx = chord_qualities.index(quality) if quality in chord_qualities else 0
            embedding[root * len(chord_qualities) + quality_idx] = 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Returns:
            Dictionary with:
            - emotion_embedding: torch.Tensor (emotion_dim,)
            - chord_embedding: torch.Tensor (chord_dim,)
            - lyrics: str (original lyrics text)
            - metadata: dict (sample metadata)
        """
        sample = self.samples[idx]
        
        # Compute or cache embeddings
        if sample._emotion_embedding is None:
            sample._emotion_embedding = self._compute_emotion_embedding(sample)
        
        if sample._chord_embedding is None:
            sample._chord_embedding = self._compute_chord_embedding(sample)
        
        return {
            "emotion_embedding": torch.from_numpy(sample._emotion_embedding).float(),
            "chord_embedding": torch.from_numpy(sample._chord_embedding).float(),
            "lyrics": sample.lyrics or "",
            "title": sample.title,
            "artist": sample.artist or "",
            "year": sample.year or 0,
            "genre": sample.genre or [],
            "tempo": torch.tensor(sample.tempo or 120.0, dtype=torch.float32),
            "key": sample.key or "",
            "chords": sample.chords or [],
            "song_id": sample.song_id,
        }
    
    def get_sample_info(self, idx: int) -> WasabiSample:
        """Get full sample information."""
        return self.samples[idx]


__all__ = ["WasabiDataset", "WasabiSample"]
