# Integration Architecture Design

**Date**: 2024-12-19  
**Purpose**: Detailed architecture for integrating production guides with tools

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Intent Layer                          │
│  (Emotion: "melancholy", Genre: "jazz", Section: "verse")   │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Emotion Processing Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  EmotionThesaurus                                      │  │
│  │  - find_by_synonym()                                   │  │
│  │  - get_intensity_synonyms()                            │  │
│  │  - find_blend()                                        │  │
│  └──────────────────┬─────────────────────────────────────┘  │
│                     │                                         │
│  ┌──────────────────▼─────────────────────────────────────┐  │
│  │  EmotionProductionMapper                               │  │
│  │  - get_production_preset(emotion, genre)                │  │
│  │  - get_drum_style(emotion)                             │  │
│  │  - get_dynamics_level(emotion, section)                 │  │
│  └──────────────────┬─────────────────────────────────────┘  │
└─────────────────────┼─────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Production Guide Layer                            │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Drum Programming  │  │ Dynamics &       │                │
│  │ Guide Rules       │  │ Arrangement      │                │
│  │ - Velocity patterns│  │ Guide Rules      │                │
│  │ - Ghost notes     │  │ - Section levels │                │
│  │ - Timing variation│  │ - Build techniques│              │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────┬─────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Analysis & Application Layer                     │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ DrumAnalyzer      │  │ DrumHumanizer    │                │
│  │ - Analyze MIDI    │  │ - Apply guide    │                │
│  │ - Detect techniques│  │   rules          │                │
│  │ - Create profile  │  │ - Humanize MIDI  │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                      │                            │
│           └──────────┬───────────┘                            │
│                      ▼                                         │
│           ┌─────────────────────┐                             │
│           │ DynamicsEngine       │                             │
│           │ - Apply section      │                             │
│           │   dynamics           │                             │
│           │ - Create automation  │                             │
│           └─────────────────────┘                             │
└─────────────────────┬─────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Layer                                     │
│  - Humanized MIDI                                             │
│  - Section dynamics automation                                │
│  - Production preset                                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
1. User Input: "melancholy jazz verse"
   ↓
2. EmotionThesaurus.find_by_synonym("melancholy")
   → EmotionMatch {
       base_emotion: "SAD",
       intensity_tier: 4,
       sub_emotion: "GRIEF"
     }
   ↓
3. EmotionProductionMapper.get_production_preset(emotion, "jazz")
   → ProductionPreset {
       drum_style: "jazzy",          # Heavy ghost notes
       dynamics_level: "mp",         # Moderately quiet
       arrangement_density: 0.4,     # Sparse
       humanization: {
         complexity: 0.6,            # Loose timing
         vulnerability: 0.7,         # Expressive dynamics
         ghost_notes: True
       }
     }
   ↓
4. [Optional] DrumAnalyzer.analyze(existing_midi)
   → DrumTechniqueProfile {
       ghost_note_density: 0.15,     # Already has some
       tightness: 0.8,               # Pretty tight
       primary_technique: "standard"
     }
   ↓
5. DrumHumanizer.apply_guide_rules(midi, profile, preset)
   → Applies:
     - Velocity patterns from guide
     - Ghost notes (sparse, vel 25-45)
     - Timing variation (±10-20ms for hi-hats)
     - Section-aware adjustments
   ↓
6. DynamicsEngine.apply_section_dynamics(structure, emotion)
   → {
       "intro": "pp",
       "verse": "mp",
       "chorus": "f",
       "bridge": "p"
     }
   ↓
7. Output: Humanized MIDI + Dynamics automation
```

---

## 2. Module Design

### 2.1 `music_brain/emotion/emotion_production.py`

```python
"""
Emotion → Production Mapping

Maps emotions from EmotionThesaurus to production techniques
from production guides.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path

from music_brain.emotion.emotion_thesaurus import EmotionThesaurus, EmotionMatch


@dataclass
class ProductionPreset:
    """Production preset derived from emotion and guides."""
    drum_style: str              # "jazzy", "rock", "hip-hop", "standard", "heavy", "technical"
    dynamics_level: str           # "pp", "p", "mp", "mf", "f", "ff", "fff"
    arrangement_density: float    # 0.0 (sparse) to 1.0 (full)
    humanization: Dict[str, float]  # complexity, vulnerability, etc.
    genre_hints: Optional[Dict[str, str]] = None


class EmotionProductionMapper:
    """
    Maps emotions to production techniques from guides.
    
    Encodes knowledge from:
    - Drum Programming Guide.md
    - Dynamics and Arrangement Guide.md
    - Electronic EDM Production Guide.md
    """
    
    def __init__(self, thesaurus: Optional[EmotionThesaurus] = None):
        self.thesaurus = thesaurus or EmotionThesaurus()
        self._load_guide_mappings()
    
    def _load_guide_mappings(self):
        """Load mappings from emotion to production techniques."""
        # Encoded from production guides
        self._emotion_to_drum_style = {
            "SAD": {
                "GRIEF": "jazzy",      # Heavy ghost notes
                "LONELINESS": "standard",
                "MELANCHOLY": "jazzy",
            },
            "ANGRY": {
                "RAGE": "heavy",       # Lots of flams
                "FRUSTRATION": "standard",
            },
            "HAPPY": {
                "JOY": "standard",
                "EUPHORIA": "technical",  # Busy fills
            },
            "FEAR": {
                "ANXIETY": "standard",
                "TERROR": "heavy",
            },
        }
        
        self._intensity_to_dynamics = {
            1: "pp",   # subtle
            2: "p",    # mild
            3: "mp",   # moderate
            4: "mf",   # strong
            5: "f",    # intense
            6: "ff",   # overwhelming
        }
        
        self._emotion_to_density = {
            "SAD": 0.3,      # Sparse arrangements
            "ANGRY": 0.8,    # Full arrangements
            "HAPPY": 0.7,
            "FEAR": 0.5,
            "SURPRISE": 0.6,
            "DISGUST": 0.4,
        }
    
    def get_production_preset(
        self,
        emotion: EmotionMatch,
        genre: Optional[str] = None,
        section: Optional[str] = None
    ) -> ProductionPreset:
        """
        Get production preset from emotion.
        
        Args:
            emotion: EmotionMatch from thesaurus
            genre: Optional genre hint ("jazz", "rock", "edm", etc.)
            section: Optional section ("verse", "chorus", "bridge")
        
        Returns:
            ProductionPreset with guide-based recommendations
        """
        # Get drum style
        drum_style = self.get_drum_style(emotion, genre)
        
        # Get dynamics level
        dynamics_level = self.get_dynamics_level(emotion, section)
        
        # Get arrangement density
        density = self._emotion_to_density.get(
            emotion.base_emotion.upper(),
            0.5
        )
        
        # Adjust for intensity tier
        if emotion.intensity_tier >= 5:
            density += 0.2
        elif emotion.intensity_tier <= 2:
            density -= 0.2
        density = max(0.0, min(1.0, density))
        
        # Get humanization settings
        humanization = self._get_humanization_settings(emotion, drum_style)
        
        return ProductionPreset(
            drum_style=drum_style,
            dynamics_level=dynamics_level,
            arrangement_density=density,
            humanization=humanization,
            genre_hints={"genre": genre} if genre else None
        )
    
    def get_drum_style(
        self,
        emotion: EmotionMatch,
        genre: Optional[str] = None
    ) -> str:
        """
        Get drum style from emotion and genre.
        
        Styles from Drum Programming Guide:
        - "jazzy": Heavy ghost notes, loose timing
        - "rock": Moderate ghost notes, tight kick/snare
        - "hip-hop": Snare slightly late, heavy swing
        - "standard": Balanced approach
        - "heavy": Lots of flams, technical
        - "technical": Buzz rolls, complex fills
        """
        base = emotion.base_emotion.upper()
        sub = emotion.sub_emotion.upper()
        
        # Check emotion mapping
        if base in self._emotion_to_drum_style:
            if sub in self._emotion_to_drum_style[base]:
                style = self._emotion_to_drum_style[base][sub]
                if genre:
                    # Override with genre if specified
                    genre_styles = {
                        "jazz": "jazzy",
                        "rock": "rock",
                        "hip-hop": "hip-hop",
                        "edm": "standard",  # EDM can be tighter
                    }
                    if genre.lower() in genre_styles:
                        return genre_styles[genre.lower()]
                return style
        
        # Default based on intensity
        if emotion.intensity_tier >= 5:
            return "heavy"
        elif emotion.intensity_tier <= 2:
            return "standard"
        else:
            return "standard"
    
    def get_dynamics_level(
        self,
        emotion: EmotionMatch,
        section: Optional[str] = None
    ) -> str:
        """
        Get dynamics level from emotion and section.
        
        From Dynamics and Arrangement Guide:
        - Verse: mp
        - Chorus: f
        - Bridge: varies (often drops)
        """
        # Base level from intensity
        base_level = self._intensity_to_dynamics.get(
            emotion.intensity_tier,
            "mf"
        )
        
        # Adjust for section
        if section:
            section_adjustments = {
                "intro": -2,      # Quieter
                "verse": -1,
                "pre-chorus": 0,
                "chorus": +1,     # Louder
                "bridge": -1,      # Often drops
                "outro": -1,
            }
            adjustment = section_adjustments.get(section.lower(), 0)
            
            # Apply adjustment
            levels = ["pp", "p", "mp", "mf", "f", "ff", "fff"]
            try:
                idx = levels.index(base_level)
                new_idx = max(0, min(len(levels) - 1, idx + adjustment))
                return levels[new_idx]
            except ValueError:
                pass
        
        return base_level
    
    def _get_humanization_settings(
        self,
        emotion: EmotionMatch,
        drum_style: str
    ) -> Dict[str, float]:
        """
        Get humanization settings from emotion and drum style.
        
        From Drum Programming Guide and groove_engine.py.
        """
        # Base settings from drum style
        style_settings = {
            "jazzy": {
                "complexity": 0.7,      # Loose timing
                "vulnerability": 0.8,   # Expressive dynamics
                "ghost_notes": True,
            },
            "rock": {
                "complexity": 0.4,
                "vulnerability": 0.6,
                "ghost_notes": True,
            },
            "hip-hop": {
                "complexity": 0.5,
                "vulnerability": 0.5,
                "ghost_notes": True,
            },
            "heavy": {
                "complexity": 0.6,
                "vulnerability": 0.7,
                "ghost_notes": False,  # Less ghost notes
            },
            "technical": {
                "complexity": 0.8,
                "vulnerability": 0.6,
                "ghost_notes": True,
            },
            "standard": {
                "complexity": 0.5,
                "vulnerability": 0.5,
                "ghost_notes": True,
            },
        }
        
        settings = style_settings.get(drum_style, style_settings["standard"]).copy()
        
        # Adjust for intensity tier
        intensity_factor = (emotion.intensity_tier - 3) / 3.0  # -1 to +1
        settings["complexity"] += intensity_factor * 0.2
        settings["vulnerability"] += intensity_factor * 0.2
        
        # Clamp values
        settings["complexity"] = max(0.0, min(1.0, settings["complexity"]))
        settings["vulnerability"] = max(0.0, min(1.0, settings["vulnerability"]))
        
        return settings
```

### 2.2 `music_brain/groove/drum_humanizer.py`

```python
"""
Drum Humanizer - Applies Drum Programming Guide Rules

Encodes principles from Drum Programming Guide.md and applies them
to MIDI files using drum_analysis.py for technique detection.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import mido

from music_brain.groove.drum_analysis import (
    DrumAnalyzer,
    DrumTechniqueProfile,
    analyze_drum_technique
)
from music_brain.groove.groove_engine import (
    GrooveSettings,
    humanize_drums,
    humanize_midi_file
)
from music_brain.emotion.emotion_production import ProductionPreset


class DrumHumanizer:
    """
    Applies Drum Programming Guide rules to MIDI.
    
    Guide principles encoded:
    - Hi-hat velocity patterns (downbeat/upbeat accents)
    - Ghost notes (velocity 25-45, sparse)
    - Timing variation (hi-hats loosest, kick tightest)
    - Fill crescendos
    - Section-aware humanization
    """
    
    def __init__(self, ppq: int = 480, bpm: float = 120.0):
        self.analyzer = DrumAnalyzer(ppq=ppq, bpm=bpm)
        self.ppq = ppq
        self.bpm = bpm
    
    def apply_guide_rules(
        self,
        midi_path: str,
        technique_profile: Optional[DrumTechniqueProfile] = None,
        preset: Optional[ProductionPreset] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Apply guide rules to MIDI file.
        
        Args:
            midi_path: Input MIDI file
            technique_profile: Optional pre-computed profile
            preset: Production preset with humanization settings
            output_path: Optional output path
        
        Returns:
            Path to humanized MIDI file
        """
        # Load MIDI
        mid = mido.MidiFile(midi_path)
        
        # Analyze if profile not provided
        if technique_profile is None:
            # Extract notes from MIDI
            notes = self._extract_notes(mid)
            technique_profile = self.analyzer.analyze(notes, bpm=self.bpm)
        
        # Get settings from preset or defaults
        if preset:
            settings = self._preset_to_settings(preset, technique_profile)
        else:
            settings = GrooveSettings()
        
        # Apply humanization
        if output_path is None:
            output_path = midi_path.replace('.mid', '_humanized.mid')
        
        return humanize_midi_file(
            input_path=midi_path,
            output_path=output_path,
            complexity=settings.complexity,
            vulnerability=settings.vulnerability,
            settings=settings
        )
    
    def create_preset_from_guide(
        self,
        style: str,
        section: Optional[str] = None
    ) -> GrooveSettings:
        """
        Create humanization preset from guide recommendations.
        
        Styles from Drum Programming Guide:
        - "jazzy": Loose hi-hats, heavy ghost notes
        - "rock": Moderate variation, sparse ghosts
        - "hip-hop": Snare late, heavy swing
        - "edm": Tighter, less variation
        """
        # Guide-based presets
        guide_presets = {
            "jazzy": GrooveSettings(
                complexity=0.7,
                vulnerability=0.8,
                enable_ghost_notes=True,
                hihat_timing_mult=1.5,  # Loosest
                snare_timing_mult=0.8,
                kick_timing_mult=0.5,  # Tightest
            ),
            "rock": GrooveSettings(
                complexity=0.4,
                vulnerability=0.6,
                enable_ghost_notes=True,
                hihat_timing_mult=1.0,
                snare_timing_mult=0.7,
                kick_timing_mult=0.5,
            ),
            "hip-hop": GrooveSettings(
                complexity=0.5,
                vulnerability=0.5,
                enable_ghost_notes=True,
                hihat_timing_mult=1.2,
                snare_timing_mult=0.9,  # Slightly late
                kick_timing_mult=0.5,
            ),
            "edm": GrooveSettings(
                complexity=0.3,  # Tighter
                vulnerability=0.4,
                enable_ghost_notes=False,  # EDM less ghost notes
                hihat_timing_mult=0.8,
                snare_timing_mult=0.6,
                kick_timing_mult=0.4,
            ),
        }
        
        preset = guide_presets.get(style, GrooveSettings())
        
        # Adjust for section (from Dynamics Guide)
        if section:
            section_adjustments = {
                "verse": {"complexity": -0.1, "vulnerability": -0.1},
                "chorus": {"complexity": +0.1, "vulnerability": +0.1},
                "bridge": {"complexity": -0.2, "vulnerability": +0.1},
            }
            if section.lower() in section_adjustments:
                adj = section_adjustments[section.lower()]
                preset.complexity = max(0.0, min(1.0, preset.complexity + adj["complexity"]))
                preset.vulnerability = max(0.0, min(1.0, preset.vulnerability + adj["vulnerability"]))
        
        return preset
    
    def _preset_to_settings(
        self,
        preset: ProductionPreset,
        profile: DrumTechniqueProfile
    ) -> GrooveSettings:
        """Convert ProductionPreset to GrooveSettings."""
        humanization = preset.humanization
        
        # Adjust based on detected techniques
        if profile.ghost_note_density < 0.1 and preset.humanization.get("ghost_notes", True):
            # Add more ghost notes if missing
            pass  # Will be handled by groove_engine
        
        return GrooveSettings(
            complexity=humanization.get("complexity", 0.5),
            vulnerability=humanization.get("vulnerability", 0.5),
            enable_ghost_notes=humanization.get("ghost_notes", True),
        )
    
    def _extract_notes(self, mid: mido.MidiFile) -> List:
        """Extract notes from MIDI for analysis."""
        # Implementation depends on note format expected by DrumAnalyzer
        # This is a placeholder
        notes = []
        # ... extract notes from MIDI ...
        return notes
```

### 2.3 `music_brain/production/dynamics_engine.py`

```python
"""
Dynamics Engine - Applies Dynamics and Arrangement Guide

Encodes principles from Dynamics and Arrangement Guide.md for
section-by-section dynamics and arrangement building.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from music_brain.emotion.emotion_production import EmotionMatch, ProductionPreset


class DynamicsLevel(Enum):
    """Dynamics levels from guide."""
    PPP = "ppp"  # Pianississimo
    PP = "pp"    # Pianissimo
    P = "p"      # Piano
    MP = "mp"    # Mezzo-piano
    MF = "mf"    # Mezzo-forte
    F = "f"      # Forte
    FF = "ff"    # Fortissimo
    FFF = "fff"  # Fortississimo


@dataclass
class SongStructure:
    """Song structure with sections."""
    sections: List[Dict[str, any]]  # [{"name": "verse", "bars": 8}, ...]


@dataclass
class AutomationCurve:
    """Automation curve for dynamics."""
    points: List[Dict[str, float]]  # [{"time": 0.0, "value": 0.5}, ...]


class DynamicsEngine:
    """
    Applies Dynamics and Arrangement Guide.
    
    Guide principles encoded:
    - Section dynamics (verse: mp, chorus: f, etc.)
    - Build techniques (pre-chorus crescendo)
    - Contrast rules (quiet parts make loud parts loud)
    - Arrangement density per section
    """
    
    def __init__(self):
        self._load_guide_rules()
    
    def _load_guide_rules(self):
        """Load dynamics rules from guide."""
        # From Dynamics and Arrangement Guide
        self._section_dynamics = {
            "intro": "pp",
            "verse": "mp",
            "pre-chorus": "mf",
            "chorus": "f",
            "bridge": "p",  # Often drops
            "outro": "mf",
        }
        
        self._section_density = {
            "intro": 0.3,      # Sparse
            "verse": 0.5,
            "pre-chorus": 0.7,  # Building
            "chorus": 0.9,      # Full
            "bridge": 0.4,      # Often sparse
            "outro": 0.6,
        }
    
    def apply_section_dynamics(
        self,
        structure: SongStructure,
        emotion: Optional[EmotionMatch] = None,
        preset: Optional[ProductionPreset] = None
    ) -> Dict[str, str]:
        """
        Get dynamics levels per section.
        
        Args:
            structure: Song structure
            emotion: Optional emotion for intensity scaling
            preset: Optional production preset
        
        Returns:
            Dict mapping section names to dynamics levels
        """
        dynamics = {}
        
        for section in structure.sections:
            name = section["name"].lower()
            
            # Get base level from guide
            base_level = self._section_dynamics.get(name, "mf")
            
            # Adjust for emotion intensity
            if emotion:
                intensity_factor = (emotion.intensity_tier - 3) / 3.0
                # Higher intensity = louder
                levels = ["pp", "p", "mp", "mf", "f", "ff", "fff"]
                try:
                    idx = levels.index(base_level)
                    new_idx = max(0, min(len(levels) - 1, idx + int(intensity_factor)))
                    base_level = levels[new_idx]
                except ValueError:
                    pass
            
            # Override with preset if provided
            if preset and preset.dynamics_level:
                base_level = preset.dynamics_level
            
            dynamics[name] = base_level
        
        return dynamics
    
    def create_automation(
        self,
        structure: SongStructure,
        dynamics: Dict[str, str],
        bpm: float = 120.0
    ) -> AutomationCurve:
        """
        Create automation curve from dynamics.
        
        Args:
            structure: Song structure
            dynamics: Section dynamics levels
            bpm: Tempo for time calculation
        
        Returns:
            AutomationCurve with automation points
        """
        # Convert dynamics levels to values (0.0 to 1.0)
        level_values = {
            "ppp": 0.1,
            "pp": 0.2,
            "p": 0.3,
            "mp": 0.5,
            "mf": 0.7,
            "f": 0.85,
            "ff": 0.95,
            "fff": 1.0,
        }
        
        points = []
        current_time = 0.0
        beats_per_bar = 4.0
        seconds_per_beat = 60.0 / bpm
        
        for section in structure.sections:
            name = section["name"].lower()
            bars = section.get("bars", 8)
            duration = bars * beats_per_bar * seconds_per_beat
            
            # Get dynamics value
            level = dynamics.get(name, "mf")
            value = level_values.get(level, 0.5)
            
            # Add points (start and end)
            points.append({"time": current_time, "value": value})
            points.append({"time": current_time + duration, "value": value})
            
            current_time += duration
        
        return AutomationCurve(points=points)
    
    def get_arrangement_density(
        self,
        section: str,
        emotion: Optional[EmotionMatch] = None
    ) -> float:
        """
        Get arrangement density for section.
        
        From Dynamics Guide: Add instruments = louder, more intense.
        """
        base_density = self._section_density.get(section.lower(), 0.5)
        
        # Adjust for emotion
        if emotion:
            intensity_factor = (emotion.intensity_tier - 3) / 3.0
            base_density += intensity_factor * 0.2
        
        return max(0.0, min(1.0, base_density))
```

---

## 3. API Contracts

### 3.1 Emotion Production Mapper

**Input**: `EmotionMatch` + optional genre/section  
**Output**: `ProductionPreset`

**Guarantees**:
- Always returns valid preset
- Preset values are within valid ranges
- Preset respects guide recommendations

### 3.2 Drum Humanizer

**Input**: MIDI file + optional profile/preset  
**Output**: Humanized MIDI file

**Guarantees**:
- Output MIDI is valid
- Guide rules are applied
- Original MIDI structure preserved

### 3.3 Dynamics Engine

**Input**: Song structure + optional emotion/preset  
**Output**: Dynamics mapping + automation curve

**Guarantees**:
- All sections have dynamics levels
- Automation curve is continuous
- Levels respect guide recommendations

---

## 4. Integration Points

### 4.1 With Existing Systems

- **groove_engine.py**: Uses `GrooveSettings` from humanizer
- **emotion_api.py**: Uses `EmotionProductionMapper` for presets
- **drum_analysis.py**: Feeds into `DrumHumanizer`

### 4.2 External Interfaces

- **MIDI I/O**: Uses `mido` library
- **File System**: Reads/writes MIDI files
- **Configuration**: JSON for guide mappings (optional)

---

## 5. Error Handling

- **Missing Data**: Default to safe values
- **Invalid Input**: Raise clear exceptions
- **File Errors**: Graceful degradation
- **Analysis Failures**: Fall back to defaults

---

## 6. Performance Considerations

- **Analysis Caching**: Cache technique profiles
- **Lazy Loading**: Load guide mappings on demand
- **Batch Processing**: Support multiple files

---

## 7. Testing Strategy

1. **Unit Tests**: Each module independently
2. **Integration Tests**: Full pipeline
3. **Guide Compliance**: Verify rules match guides
4. **Edge Cases**: Invalid inputs, missing data
