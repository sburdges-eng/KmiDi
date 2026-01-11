"""
Cross-cultural music support for KmiDi.

Integrates non-Western musical systems (Raga, Maqam, East Asian Pentatonic)
with the emotion mapping system for culturally-aware music generation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from music_brain.emotion.emotion_thesaurus import EmotionMatch


class MusicSystem(Enum):
    """Musical system types."""
    WESTERN = "western"
    RAGA = "raga"
    MAQAM = "maqam"
    PENTATONIC = "pentatonic"


@dataclass
class CulturalScale:
    """Represents a scale from a non-Western musical system."""
    name: str
    culture: str
    system: MusicSystem
    intervals_semitones: List[int]
    emotional_qualities: List[str]
    emotion_mapping: Dict[str, float]
    intensity_range: Tuple[float, float]
    therapeutic_use: List[str]
    metadata: Dict[str, Any]


class RagaSystem:
    """Indian Classical Raga system."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize Raga system with emotion mappings."""
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "cultural" / "raga_emotion_map.json"
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.ragas: Dict[str, Dict[str, Any]] = {
            raga["name"]: raga for raga in data["ragas"]
        }
        self.metadata = data["metadata"]
    
    def find_raga_for_emotion(self, emotion: str, intensity: float = 0.5) -> Optional[str]:
        """
        Find best Raga for given emotion and intensity.
        
        Args:
            emotion: Emotion name (e.g., "sad", "happy", "calm")
            intensity: Emotion intensity (0.0-1.0)
            
        Returns:
            Raga name or None
        """
        best_raga = None
        best_score = 0.0
        
        for raga_name, raga_data in self.ragas.items():
            emotion_map = raga_data.get("emotion_mapping", {})
            score = emotion_map.get(emotion.lower(), 0.0)
            
            # Check if intensity is within range
            intensity_range = raga_data.get("intensity_range", [0.0, 1.0])
            if intensity < intensity_range[0] or intensity > intensity_range[1]:
                score *= 0.5  # Penalize if out of range
            
            if score > best_score:
                best_score = score
                best_raga = raga_name
        
        return best_raga
    
    def get_raga(self, name: str) -> Optional[CulturalScale]:
        """Get Raga by name."""
        if name not in self.ragas:
            return None
        
        raga_data = self.ragas[name]
        return CulturalScale(
            name=raga_data["name"],
            culture="Indian Classical",
            system=MusicSystem.RAGA,
            intervals_semitones=raga_data["intervals_semitones"],
            emotional_qualities=raga_data["emotional_qualities"],
            emotion_mapping=raga_data["emotion_mapping"],
            intensity_range=tuple(raga_data["intensity_range"]),
            therapeutic_use=raga_data["therapeutic_use"],
            metadata={
                "alternate_names": raga_data.get("alternate_names", []),
                "time_association": raga_data.get("time_association", ""),
                "ascending": raga_data.get("ascending", []),
                "descending": raga_data.get("descending", []),
                "notes": raga_data.get("notes", ""),
            }
        )


class MaqamSystem:
    """Arabic/Middle Eastern Maqam system."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize Maqam system with emotion mappings."""
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "cultural" / "maqam_emotion_map.json"
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.maqamat: Dict[str, Dict[str, Any]] = {
            maqam["name"]: maqam for maqam in data["maqamat"]
        }
        self.metadata = data["metadata"]
    
    def find_maqam_for_emotion(self, emotion: str, intensity: float = 0.5) -> Optional[str]:
        """
        Find best Maqam for given emotion and intensity.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity (0.0-1.0)
            
        Returns:
            Maqam name or None
        """
        best_maqam = None
        best_score = 0.0
        
        for maqam_name, maqam_data in self.maqamat.items():
            emotion_map = maqam_data.get("emotion_mapping", {})
            score = emotion_map.get(emotion.lower(), 0.0)
            
            # Check if intensity is within range
            intensity_range = maqam_data.get("intensity_range", [0.0, 1.0])
            if intensity < intensity_range[0] or intensity > intensity_range[1]:
                score *= 0.5
            
            if score > best_score:
                best_score = score
                best_maqam = maqam_name
        
        return best_maqam
    
    def get_maqam(self, name: str) -> Optional[CulturalScale]:
        """Get Maqam by name."""
        if name not in self.maqamat:
            return None
        
        maqam_data = self.maqamat[name]
        return CulturalScale(
            name=maqam_data["name"],
            culture="Arabic/Middle Eastern",
            system=MusicSystem.MAQAM,
            intervals_semitones=maqam_data["intervals_semitones"],
            emotional_qualities=maqam_data["emotional_qualities"],
            emotion_mapping=maqam_data["emotion_mapping"],
            intensity_range=tuple(maqam_data["intensity_range"]),
            therapeutic_use=maqam_data["therapeutic_use"],
            metadata={
                "tetrachords": maqam_data.get("tetrachords", []),
                "notes": maqam_data.get("notes", ""),
            }
        )


class EastAsianPentatonicSystem:
    """East Asian pentatonic scales (Chinese, Japanese, Korean)."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize pentatonic system with emotion mappings."""
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "cultural" / "pentatonic_emotion_map.json"
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.scales: Dict[str, Dict[str, Any]] = {
            scale["name"]: scale for scale in data["pentatonic_systems"]
        }
        self.metadata = data["metadata"]
    
    def find_scale_for_emotion(self, emotion: str, intensity: float = 0.5, culture: Optional[str] = None) -> Optional[str]:
        """
        Find best pentatonic scale for given emotion.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity (0.0-1.0)
            culture: Optional culture filter ("Chinese", "Japanese", "Korean", "Mongolian")
            
        Returns:
            Scale name or None
        """
        best_scale = None
        best_score = 0.0
        
        for scale_name, scale_data in self.scales.items():
            # Filter by culture if specified
            if culture and scale_data.get("culture", "").lower() != culture.lower():
                continue
            
            emotion_map = scale_data.get("emotion_mapping", {})
            score = emotion_map.get(emotion.lower(), 0.0)
            
            # Check if intensity is within range
            intensity_range = scale_data.get("intensity_range", [0.0, 1.0])
            if intensity < intensity_range[0] or intensity > intensity_range[1]:
                score *= 0.5
            
            if score > best_score:
                best_score = score
                best_scale = scale_name
        
        return best_scale
    
    def get_scale(self, name: str) -> Optional[CulturalScale]:
        """Get pentatonic scale by name."""
        if name not in self.scales:
            return None
        
        scale_data = self.scales[name]
        return CulturalScale(
            name=scale_data["name"],
            culture=scale_data["culture"],
            system=MusicSystem.PENTATONIC,
            intervals_semitones=scale_data["intervals_semitones"],
            emotional_qualities=scale_data["emotional_qualities"],
            emotion_mapping=scale_data["emotion_mapping"],
            intensity_range=tuple(scale_data["intensity_range"]),
            therapeutic_use=scale_data["therapeutic_use"],
            metadata={
                "scale_degrees": scale_data.get("scale_degrees", []),
                "notes": scale_data.get("notes", ""),
            }
        )


class CrossCulturalMusicMapper:
    """Main mapper for cross-cultural music systems."""
    
    def __init__(self):
        """Initialize all cultural music systems."""
        self.raga_system = RagaSystem()
        self.maqam_system = MaqamSystem()
        self.pentatonic_system = EastAsianPentatonicSystem()
    
    def get_cultural_scale_for_emotion(
        self,
        emotion: str,
        system: Optional[MusicSystem] = None,
        intensity: float = 0.5,
        culture: Optional[str] = None
    ) -> Optional[CulturalScale]:
        """
        Get cultural scale for emotion.
        
        Args:
            emotion: Emotion name
            system: Optional music system filter
            intensity: Emotion intensity (0.0-1.0)
            culture: Optional culture filter (for pentatonic)
            
        Returns:
            CulturalScale or None
        """
        if system == MusicSystem.RAGA:
            raga_name = self.raga_system.find_raga_for_emotion(emotion, intensity)
            if raga_name:
                return self.raga_system.get_raga(raga_name)
        elif system == MusicSystem.MAQAM:
            maqam_name = self.maqam_system.find_maqam_for_emotion(emotion, intensity)
            if maqam_name:
                return self.maqam_system.get_maqam(maqam_name)
        elif system == MusicSystem.PENTATONIC:
            scale_name = self.pentatonic_system.find_scale_for_emotion(emotion, intensity, culture)
            if scale_name:
                return self.pentatonic_system.get_scale(scale_name)
        else:
            # Try all systems and return best match
            best_scale = None
            best_score = 0.0
            
            # Try Raga
            raga_name = self.raga_system.find_raga_for_emotion(emotion, intensity)
            if raga_name:
                raga = self.raga_system.get_raga(raga_name)
                if raga:
                    score = raga.emotion_mapping.get(emotion.lower(), 0.0)
                    if score > best_score:
                        best_score = score
                        best_scale = raga
            
            # Try Maqam
            maqam_name = self.maqam_system.find_maqam_for_emotion(emotion, intensity)
            if maqam_name:
                maqam = self.maqam_system.get_maqam(maqam_name)
                if maqam:
                    score = maqam.emotion_mapping.get(emotion.lower(), 0.0)
                    if score > best_score:
                        best_score = score
                        best_scale = maqam
            
            # Try Pentatonic
            scale_name = self.pentatonic_system.find_scale_for_emotion(emotion, intensity, culture)
            if scale_name:
                scale = self.pentatonic_system.get_scale(scale_name)
                if scale:
                    score = scale.emotion_mapping.get(emotion.lower(), 0.0)
                    if score > best_score:
                        best_score = score
                        best_scale = scale
            
            return best_scale
        
        return None
    
    def get_all_systems_for_emotion(self, emotion: str, intensity: float = 0.5) -> Dict[str, Optional[CulturalScale]]:
        """
        Get all cultural scales for emotion across all systems.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity
            
        Returns:
            Dictionary mapping system names to scales
        """
        return {
            "raga": self.get_cultural_scale_for_emotion(emotion, MusicSystem.RAGA, intensity),
            "maqam": self.get_cultural_scale_for_emotion(emotion, MusicSystem.MAQAM, intensity),
            "pentatonic": self.get_cultural_scale_for_emotion(emotion, MusicSystem.PENTATONIC, intensity),
        }


# Convenience function
def get_cultural_scale_for_emotion(
    emotion: str,
    system: Optional[MusicSystem] = None,
    intensity: float = 0.5,
    culture: Optional[str] = None
) -> Optional[CulturalScale]:
    """
    Get cultural scale for emotion (convenience function).
    
    Args:
        emotion: Emotion name
        system: Optional music system filter
        intensity: Emotion intensity
        culture: Optional culture filter
        
    Returns:
        CulturalScale or None
    """
    mapper = CrossCulturalMusicMapper()
    return mapper.get_cultural_scale_for_emotion(emotion, system, intensity, culture)

