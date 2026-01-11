"""
Arrangement Generator - Full song arrangement from musical seeds.

Generates complete song arrangements including:
- Song structure (intro, verse, chorus, bridge, outro)
- Instrument layering
- Dynamic progression
- Orchestration

Integrates all generative components to produce cohesive arrangements.

Usage:
    from music_brain.generative import ArrangementGenerator
    
    gen = ArrangementGenerator(device="mps")
    arrangement = gen.generate(
        emotion="hope",
        genre="pop",
        duration=180,  # 3 minutes
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

from .base import GenerativeModel, GenerativeConfig, GenerationResult


@dataclass
class ArrangementConfig(GenerativeConfig):
    """Configuration for arrangement generator."""
    
    # Structure
    default_duration: float = 180.0  # 3 minutes
    default_tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    
    # Sections
    min_section_bars: int = 4
    max_section_bars: int = 16
    
    # Instruments
    max_tracks: int = 16
    
    # Quality
    output_format: str = "midi"  # "midi", "audio", "both"


# Song structure templates
STRUCTURE_TEMPLATES = {
    "pop": [
        {"section": "intro", "bars": 4},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "bridge", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 4},
    ],
    "pop_with_solo": [
        {"section": "intro", "bars": 4},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "solo", "bars": 8},
        {"section": "bridge", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 4},
    ],
    "pop_minimal": [
        {"section": "intro", "bars": 4},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 4},
    ],
    "ballad": [
        {"section": "intro", "bars": 8},
        {"section": "verse", "bars": 16},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 16},
        {"section": "chorus", "bars": 8},
        {"section": "bridge", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 8},
    ],
    "ballad_with_solo": [
        {"section": "intro", "bars": 8},
        {"section": "verse", "bars": 16},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 16},
        {"section": "chorus", "bars": 8},
        {"section": "solo", "bars": 16},
        {"section": "bridge", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 8},
    ],
    "rock": [
        {"section": "intro", "bars": 4},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "solo", "bars": 16},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 8},
    ],
    "rock_extended": [
        {"section": "intro", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "bridge", "bars": 8},
        {"section": "solo", "bars": 16},
        {"section": "chorus", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 8},
    ],
    "edm": [
        {"section": "intro", "bars": 8},
        {"section": "buildup", "bars": 8},
        {"section": "drop", "bars": 16},
        {"section": "breakdown", "bars": 8},
        {"section": "buildup", "bars": 8},
        {"section": "drop", "bars": 16},
        {"section": "outro", "bars": 8},
    ],
    "edm_extended": [
        {"section": "intro", "bars": 8},
        {"section": "buildup", "bars": 8},
        {"section": "drop", "bars": 16},
        {"section": "breakdown", "bars": 8},
        {"section": "bridge", "bars": 8},
        {"section": "buildup", "bars": 8},
        {"section": "drop", "bars": 16},
        {"section": "breakdown", "bars": 8},
        {"section": "outro", "bars": 8},
    ],
    "ambient": [
        {"section": "intro", "bars": 16},
        {"section": "development", "bars": 32},
        {"section": "climax", "bars": 16},
        {"section": "resolution", "bars": 16},
        {"section": "outro", "bars": 16},
    ],
    "jazz": [
        {"section": "intro", "bars": 4},
        {"section": "head", "bars": 32},
        {"section": "solo", "bars": 32},
        {"section": "solo", "bars": 32},
        {"section": "head", "bars": 32},
        {"section": "outro", "bars": 8},
    ],
    "jazz_with_bridge": [
        {"section": "intro", "bars": 4},
        {"section": "head", "bars": 32},
        {"section": "solo", "bars": 32},
        {"section": "bridge", "bars": 16},
        {"section": "solo", "bars": 32},
        {"section": "head", "bars": 32},
        {"section": "outro", "bars": 8},
    ],
    "blues": [
        {"section": "intro", "bars": 4},
        {"section": "verse", "bars": 12},
        {"section": "verse", "bars": 12},
        {"section": "solo", "bars": 12},
        {"section": "verse", "bars": 12},
        {"section": "solo", "bars": 12},
        {"section": "outro", "bars": 4},
    ],
    "prog_rock": [
        {"section": "intro", "bars": 8},
        {"section": "verse", "bars": 8},
        {"section": "chorus", "bars": 8},
        {"section": "bridge", "bars": 12},
        {"section": "solo", "bars": 16},
        {"section": "verse", "bars": 8},
        {"section": "bridge", "bars": 12},
        {"section": "solo", "bars": 24},
        {"section": "chorus", "bars": 8},
        {"section": "outro", "bars": 12},
    ],
}

# Instrument palettes by genre
INSTRUMENT_PALETTES = {
    "pop": {
        "drums": ["kick", "snare", "hihat", "crash"],
        "bass": ["electric_bass"],
        "chords": ["piano", "guitar"],
        "lead": ["synth_lead", "vocal"],
        "pads": ["strings", "synth_pad"],
    },
    "rock": {
        "drums": ["kick", "snare", "hihat", "toms", "crash"],
        "bass": ["electric_bass"],
        "chords": ["electric_guitar", "power_chords"],
        "lead": ["lead_guitar", "vocal"],
        "pads": [],
    },
    "electronic": {
        "drums": ["kick", "clap", "hihat", "percussion"],
        "bass": ["synth_bass", "sub_bass"],
        "chords": ["synth_pluck", "synth_stab"],
        "lead": ["synth_lead", "arp"],
        "pads": ["synth_pad", "atmosphere"],
    },
    "orchestral": {
        "drums": ["timpani", "percussion"],
        "bass": ["contrabass", "cello"],
        "chords": ["strings_section", "brass"],
        "lead": ["violin", "flute", "oboe"],
        "pads": ["strings_sustained", "choir"],
    },
    "jazz": {
        "drums": ["brush_kit", "ride", "hihat"],
        "bass": ["upright_bass"],
        "chords": ["piano", "guitar"],
        "lead": ["saxophone", "trumpet", "piano"],
        "pads": [],
    },
}

# Section characteristics
SECTION_CHARACTERISTICS = {
    "intro": {
        "energy": 0.3,
        "complexity": 0.3,
        "instruments": ["minimal"],
        "dynamic": "building",
    },
    "verse": {
        "energy": 0.5,
        "complexity": 0.5,
        "instruments": ["foundation"],
        "dynamic": "steady",
    },
    "pre_chorus": {
        "energy": 0.6,
        "complexity": 0.6,
        "instruments": ["foundation", "extras"],
        "dynamic": "building",
    },
    "chorus": {
        "energy": 0.9,
        "complexity": 0.7,
        "instruments": ["full"],
        "dynamic": "high",
    },
    "bridge": {
        "energy": 0.6,
        "complexity": 0.8,
        "instruments": ["varied"],
        "dynamic": "contrasting",
    },
    "solo": {
        "energy": 0.85,
        "complexity": 0.9,
        "instruments": ["solo_lead", "rhythm"],
        "dynamic": "expressive",
    },
    "head": {
        "energy": 0.7,
        "complexity": 0.7,
        "instruments": ["full"],
        "dynamic": "melodic",
    },
    "breakdown": {
        "energy": 0.3,
        "complexity": 0.4,
        "instruments": ["minimal"],
        "dynamic": "low",
    },
    "buildup": {
        "energy": 0.7,
        "complexity": 0.6,
        "instruments": ["building"],
        "dynamic": "crescendo",
    },
    "drop": {
        "energy": 1.0,
        "complexity": 0.8,
        "instruments": ["full"],
        "dynamic": "maximum",
    },
    "development": {
        "energy": 0.5,
        "complexity": 0.6,
        "instruments": ["layered"],
        "dynamic": "evolving",
    },
    "climax": {
        "energy": 0.9,
        "complexity": 0.8,
        "instruments": ["full"],
        "dynamic": "peak",
    },
    "resolution": {
        "energy": 0.4,
        "complexity": 0.5,
        "instruments": ["fading"],
        "dynamic": "resolving",
    },
    "outro": {
        "energy": 0.3,
        "complexity": 0.3,
        "instruments": ["fading"],
        "dynamic": "diminishing",
    },
}


class ArrangementGenerator(GenerativeModel):
    """
    Generate complete song arrangements.
    
    Combines:
    - Structure planning
    - Chord progression generation
    - Melody generation
    - Instrument allocation
    - Dynamic mapping
    
    Example:
        gen = ArrangementGenerator(device="mps")
        
        arrangement = gen.generate(
            emotion="hope",
            genre="pop",
            duration=180,
            key="G",
            tempo=120
        )
        
        # Save to MIDI
        arrangement.save("my_song.mid")
    """
    
    def __init__(
        self,
        device: str = "auto",
        config: Optional[ArrangementConfig] = None,
    ):
        """Initialize arrangement generator."""
        if config is None:
            config = ArrangementConfig(device=device)
        super().__init__(config)
        
        self.config: ArrangementConfig = config
        self._chord_gen = None
        self._melody_gen = None
        self._vae = None
    
    def load(self, path: Optional[str] = None) -> None:
        """Load component generators."""
        try:
            from .chord_generator import ChordProgressionGenerator
            self._chord_gen = ChordProgressionGenerator(device=self.config.get_device())
            self._chord_gen.load()
        except ImportError:
            pass
        
        try:
            from .melody_vae import MelodyVAE
            self._vae = MelodyVAE(device=self.config.get_device())
            self._vae.load()
        except ImportError:
            pass
        
        try:
            from music_brain.session.ml_melody_generator import MLMelodyGenerator
            self._melody_gen = MLMelodyGenerator()
        except ImportError:
            pass
        
        self._is_loaded = True
    
    def generate(
        self,
        emotion: str = "peace",
        genre: str = "pop",
        duration: Optional[float] = None,
        key: str = "C",
        tempo: Optional[int] = None,
        structure: Optional[List[Dict]] = None,
        include_solo: bool = False,
        include_bridge: bool = True,
        solo_bars: int = 16,
        bridge_bars: int = 8,
        **kwargs,
    ) -> "Arrangement":
        """
        Generate a complete arrangement.

        Args:
            emotion: Emotional character
            genre: Musical genre/style
            duration: Target duration in seconds
            key: Key signature
            tempo: Tempo in BPM
            structure: Custom structure (list of section dicts)
            include_solo: Whether to include a solo section
            include_bridge: Whether to include a bridge section
            solo_bars: Number of bars for solo section (default 16)
            bridge_bars: Number of bars for bridge section (default 8)
            **kwargs: Additional parameters

        Returns:
            Arrangement object with all tracks and structure
        """
        if not self._is_loaded:
            self.load()

        duration = duration or self.config.default_duration
        tempo = tempo or self.config.default_tempo

        # Get or create structure
        if structure is None:
            structure = self._generate_structure(
                genre, duration, tempo,
                include_solo=include_solo,
                include_bridge=include_bridge,
                solo_bars=solo_bars,
                bridge_bars=bridge_bars,
            )
        
        # Generate chords for entire song
        chord_progression = self._generate_song_chords(
            emotion=emotion,
            key=key,
            structure=structure,
        )
        
        # Generate melodies for each section
        melodies = self._generate_section_melodies(
            emotion=emotion,
            key=key,
            structure=structure,
            chords=chord_progression,
        )
        
        # Generate instrument tracks
        tracks = self._generate_tracks(
            genre=genre,
            emotion=emotion,
            structure=structure,
            chords=chord_progression,
            melodies=melodies,
            tempo=tempo,
        )
        
        # Create arrangement object
        arrangement = Arrangement(
            tracks=tracks,
            structure=structure,
            chords=chord_progression,
            melodies=melodies,
            tempo=tempo,
            key=key,
            time_signature=self.config.time_signature,
            emotion=emotion,
            genre=genre,
        )
        
        return arrangement
    
    def _generate_structure(
        self,
        genre: str,
        duration: float,
        tempo: int,
        include_solo: bool = False,
        include_bridge: bool = True,
        solo_bars: int = 16,
        bridge_bars: int = 8,
    ) -> List[Dict]:
        """Generate song structure based on genre and duration.

        Args:
            genre: Musical genre/style
            duration: Target duration in seconds
            tempo: Tempo in BPM
            include_solo: Whether to include a solo section
            include_bridge: Whether to include a bridge section
            solo_bars: Number of bars for solo section
            bridge_bars: Number of bars for bridge section
        """
        # Check for genre variants with solo/bridge
        template_key = genre
        if include_solo and f"{genre}_with_solo" in STRUCTURE_TEMPLATES:
            template_key = f"{genre}_with_solo"
        elif include_solo and genre == "rock":
            template_key = "rock"  # Rock template already has solo

        template = STRUCTURE_TEMPLATES.get(template_key, STRUCTURE_TEMPLATES["pop"])

        # Dynamically modify template based on options
        template = list(template)  # Make a copy

        # Add solo section if requested and not already present
        if include_solo:
            has_solo = any(s["section"] == "solo" for s in template)
            if not has_solo:
                # Insert solo after second chorus
                chorus_count = 0
                insert_idx = len(template) - 1  # Default: before outro
                for i, s in enumerate(template):
                    if s["section"] == "chorus":
                        chorus_count += 1
                        if chorus_count == 2:
                            insert_idx = i + 1
                            break
                template.insert(insert_idx, {"section": "solo", "bars": solo_bars})

        # Add bridge section if requested and not already present
        if include_bridge:
            has_bridge = any(s["section"] == "bridge" for s in template)
            if not has_bridge:
                # Insert bridge before final chorus
                last_chorus_idx = None
                for i, s in enumerate(template):
                    if s["section"] == "chorus":
                        last_chorus_idx = i
                if last_chorus_idx is not None:
                    template.insert(last_chorus_idx, {"section": "bridge", "bars": bridge_bars})

        # Remove bridge if explicitly not wanted
        if not include_bridge:
            template = [s for s in template if s["section"] != "bridge"]

        # Calculate total bars in template
        template_bars = sum(s["bars"] for s in template)

        # Calculate target bars based on duration
        beats_per_bar = self.config.time_signature[0]
        seconds_per_beat = 60.0 / tempo
        seconds_per_bar = beats_per_bar * seconds_per_beat
        target_bars = int(duration / seconds_per_bar)

        # Scale structure to fit duration
        scale_factor = target_bars / template_bars if template_bars > 0 else 1.0

        structure = []
        current_bar = 0

        for section in template:
            scaled_bars = max(
                self.config.min_section_bars,
                min(
                    self.config.max_section_bars,
                    int(section["bars"] * scale_factor)
                )
            )
            
            structure.append({
                "section": section["section"],
                "bars": scaled_bars,
                "start_bar": current_bar,
                "end_bar": current_bar + scaled_bars,
                **SECTION_CHARACTERISTICS.get(section["section"], {}),
            })
            
            current_bar += scaled_bars
        
        return structure
    
    def _generate_song_chords(
        self,
        emotion: str,
        key: str,
        structure: List[Dict],
    ) -> List[Dict]:
        """Generate chord progression for entire song."""
        chord_progression = []
        
        for section in structure:
            section_length = section["bars"]
            
            # Get chords for section
            if self._chord_gen:
                chords = self._chord_gen.generate(
                    emotion=emotion,
                    key=key,
                    length=section_length,
                )
            else:
                # Fallback
                chords = self._generate_fallback_chords(key, section_length)
            
            for i, chord in enumerate(chords):
                chord_progression.append({
                    "chord": chord,
                    "bar": section["start_bar"] + i,
                    "section": section["section"],
                    "duration_bars": 1,
                })
        
        return chord_progression
    
    def _generate_fallback_chords(self, key: str, length: int) -> List[str]:
        """Generate simple chord progression as fallback."""
        is_minor = "m" in key and "maj" not in key
        
        if is_minor:
            base = [key, key.replace("m", "") + "m7", 
                   self._transpose_note(key[0], 3), 
                   self._transpose_note(key[0], 7)]
        else:
            base = [key, self._transpose_note(key, 7) + "m",
                   self._transpose_note(key, 5), 
                   self._transpose_note(key, 7)]
        
        chords = []
        while len(chords) < length:
            chords.extend(base)
        return chords[:length]
    
    def _transpose_note(self, note: str, semitones: int) -> str:
        """Transpose a note by semitones."""
        notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
        base = note[0]
        if len(note) > 1 and note[1] in "b#":
            base = note[:2]
        
        try:
            idx = notes.index(base)
            new_idx = (idx + semitones) % 12
            return notes[new_idx]
        except ValueError:
            return note
    
    def _generate_section_melodies(
        self,
        emotion: str,
        key: str,
        structure: List[Dict],
        chords: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Generate melodies for each unique section."""
        melodies = {}
        
        # Get unique sections
        unique_sections = set(s["section"] for s in structure)
        
        for section_name in unique_sections:
            # Find first occurrence of this section
            section = next(s for s in structure if s["section"] == section_name)
            notes_per_bar = 4  # Approximate
            length = section["bars"] * notes_per_bar
            
            melody = None
            if self._melody_gen:
                generated = self._melody_gen.generate(
                    emotion=emotion,
                    key=key,
                    length=length,
                )
                # Convert GeneratedMelody to list of dicts
                if hasattr(generated, 'notes'):
                    melody = [
                        {
                            "pitch": pitch,
                            "duration": dur,
                            "velocity": vel,
                            "position": i,
                        }
                        for i, (pitch, dur, vel) in enumerate(
                            zip(generated.notes, generated.durations, generated.velocities)
                        )
                    ]
            elif self._vae:
                melody = self._vae.generate(
                    key=key,
                    length=length,
                )
            
            if melody is None:
                melody = self._generate_fallback_melody(key, length)
            
            melodies[section_name] = melody
        
        return melodies
    
    def _generate_fallback_melody(self, key: str, length: int) -> List[Dict]:
        """Generate simple melody as fallback."""
        # Get scale
        root = 60 if key[0] == "C" else 62 if key[0] == "D" else 64 if key[0] == "E" else 65 if key[0] == "F" else 67 if key[0] == "G" else 69 if key[0] == "A" else 71
        
        is_minor = "m" in key and "maj" not in key
        scale = [0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11]
        notes = [root + s for s in scale]
        
        melody = []
        current = notes[0]
        
        for i in range(length):
            if np.random.random() < 0.1:  # Rest
                melody.append({"pitch": -1, "duration": 1.0, "velocity": 0, "position": i})
                continue
            
            # Step motion mostly
            step = np.random.choice([-1, 0, 1, 1])
            idx = notes.index(current) if current in notes else 0
            new_idx = max(0, min(len(notes) - 1, idx + step))
            current = notes[new_idx]
            
            melody.append({
                "pitch": current,
                "duration": np.random.choice([0.5, 1.0, 1.0, 2.0]),
                "velocity": 64 + np.random.randint(-20, 20),
                "position": i,
            })
        
        return melody
    
    def _generate_tracks(
        self,
        genre: str,
        emotion: str,
        structure: List[Dict],
        chords: List[Dict],
        melodies: Dict[str, List[Dict]],
        tempo: int,
    ) -> List[Dict]:
        """Generate all instrument tracks."""
        palette = INSTRUMENT_PALETTES.get(genre, INSTRUMENT_PALETTES["pop"])
        tracks = []
        
        # Drums track
        if palette.get("drums"):
            tracks.append({
                "name": "Drums",
                "type": "drums",
                "instruments": palette["drums"],
                "events": self._generate_drum_pattern(structure, tempo),
            })
        
        # Bass track
        if palette.get("bass"):
            tracks.append({
                "name": "Bass",
                "type": "bass",
                "instruments": palette["bass"],
                "events": self._generate_bass_line(chords, structure),
            })
        
        # Chord/harmony track
        if palette.get("chords"):
            tracks.append({
                "name": "Chords",
                "type": "chords",
                "instruments": palette["chords"],
                "events": self._generate_chord_voicings(chords, structure),
            })
        
        # Lead/melody track
        if palette.get("lead"):
            tracks.append({
                "name": "Lead",
                "type": "lead",
                "instruments": palette["lead"],
                "events": self._melody_to_events(melodies, structure),
            })
        
        # Pad/atmosphere track
        if palette.get("pads"):
            tracks.append({
                "name": "Pads",
                "type": "pads",
                "instruments": palette["pads"],
                "events": self._generate_pad_layer(chords, structure),
            })
        
        return tracks
    
    def _generate_drum_pattern(
        self,
        structure: List[Dict],
        tempo: int,
    ) -> List[Dict]:
        """Generate drum pattern events."""
        events = []
        
        for section in structure:
            energy = section.get("energy", 0.5)
            
            for bar in range(section["bars"]):
                current_bar = section["start_bar"] + bar
                
                # Basic pattern based on energy
                # Kick on 1 and 3
                events.append({"bar": current_bar, "beat": 1, "drum": "kick", "velocity": int(80 * energy)})
                events.append({"bar": current_bar, "beat": 3, "drum": "kick", "velocity": int(80 * energy)})
                
                # Snare on 2 and 4
                events.append({"bar": current_bar, "beat": 2, "drum": "snare", "velocity": int(90 * energy)})
                events.append({"bar": current_bar, "beat": 4, "drum": "snare", "velocity": int(90 * energy)})
                
                # Hihat on every eighth note if high energy
                if energy > 0.5:
                    for beat in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
                        events.append({
                            "bar": current_bar, 
                            "beat": beat, 
                            "drum": "hihat", 
                            "velocity": int(50 + 30 * energy)
                        })
        
        return events
    
    def _generate_bass_line(
        self,
        chords: List[Dict],
        structure: List[Dict],
    ) -> List[Dict]:
        """Generate bass line from chords."""
        events = []
        
        for chord_info in chords:
            chord = chord_info["chord"]
            bar = chord_info["bar"]
            
            # Get root note
            root = chord[0]
            if len(chord) > 1 and chord[1] in "b#":
                root = chord[:2]
            
            # Convert to MIDI
            note_map = {"C": 36, "D": 38, "E": 40, "F": 41, "G": 43, "A": 45, "B": 47}
            base_note = note_map.get(root, 36)
            if "b" in root:
                base_note -= 1
            elif "#" in root:
                base_note += 1
            
            # Simple pattern: root on beats 1 and 3
            events.append({"bar": bar, "beat": 1, "pitch": base_note, "duration": 1.0, "velocity": 80})
            events.append({"bar": bar, "beat": 3, "pitch": base_note, "duration": 1.0, "velocity": 70})
        
        return events
    
    def _generate_chord_voicings(
        self,
        chords: List[Dict],
        structure: List[Dict],
    ) -> List[Dict]:
        """Generate chord voicing events."""
        events = []
        
        for chord_info in chords:
            chord = chord_info["chord"]
            bar = chord_info["bar"]
            
            # Get voicing
            voicing = self._get_chord_voicing(chord)
            
            # Add event
            events.append({
                "bar": bar,
                "beat": 1,
                "pitches": voicing,
                "duration": 4.0,  # Whole note
                "velocity": 60,
            })
        
        return events
    
    def _get_chord_voicing(self, chord: str) -> List[int]:
        """Get MIDI notes for a chord voicing."""
        # Parse chord
        root = chord[0]
        if len(chord) > 1 and chord[1] in "b#":
            root = chord[:2]
            suffix = chord[2:]
        else:
            suffix = chord[1:]
        
        # Root MIDI
        note_map = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
        base = note_map.get(root, 60)
        if "b" in root:
            base -= 1
        elif "#" in root:
            base += 1
        
        # Build voicing
        if "m" in suffix and "maj" not in suffix:
            intervals = [0, 3, 7]  # Minor
        elif "dim" in suffix:
            intervals = [0, 3, 6]  # Diminished
        elif "aug" in suffix:
            intervals = [0, 4, 8]  # Augmented
        else:
            intervals = [0, 4, 7]  # Major
        
        # Add 7th if specified
        if "7" in suffix:
            if "maj7" in suffix:
                intervals.append(11)
            else:
                intervals.append(10)
        
        return [base + i for i in intervals]
    
    def _melody_to_events(
        self,
        melodies: Dict[str, List[Dict]],
        structure: List[Dict],
    ) -> List[Dict]:
        """Convert melodies to events for each section."""
        events = []
        
        for section in structure:
            section_name = section["section"]
            melody = melodies.get(section_name, [])
            
            if not melody:
                continue
            
            notes_per_bar = max(1, len(melody) / section["bars"])
            
            for i, note in enumerate(melody):
                if note.get("pitch", -1) < 0:
                    continue
                
                bar = section["start_bar"] + int(i / notes_per_bar)
                beat = 1 + (i % int(notes_per_bar)) * (4 / notes_per_bar)
                
                events.append({
                    "bar": bar,
                    "beat": beat,
                    "pitch": note["pitch"],
                    "duration": note.get("duration", 1.0),
                    "velocity": note.get("velocity", 64),
                })
        
        return events
    
    def _generate_pad_layer(
        self,
        chords: List[Dict],
        structure: List[Dict],
    ) -> List[Dict]:
        """Generate pad/atmosphere layer."""
        events = []
        
        for chord_info in chords:
            chord = chord_info["chord"]
            bar = chord_info["bar"]
            section = chord_info.get("section", "verse")
            
            # Get section info
            section_info = next(
                (s for s in structure if s["section"] == section),
                {"energy": 0.5}
            )
            
            # Only add pads for sections with medium-high energy
            if section_info.get("energy", 0.5) < 0.4:
                continue
            
            voicing = self._get_chord_voicing(chord)
            # Shift up an octave for pads
            voicing = [p + 12 for p in voicing]
            
            events.append({
                "bar": bar,
                "beat": 1,
                "pitches": voicing,
                "duration": 4.0,
                "velocity": 40,
            })
        
        return events


@dataclass
class Arrangement:
    """
    Complete song arrangement.
    
    Contains all the data needed to render a full song:
    - Track data (notes, events)
    - Structure information
    - Chord progression
    - Melodies
    - Tempo and key
    """
    
    tracks: List[Dict]
    structure: List[Dict]
    chords: List[Dict]
    melodies: Dict[str, List[Dict]]
    tempo: int
    key: str
    time_signature: Tuple[int, int]
    emotion: str
    genre: str
    
    def get_duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        total_bars = sum(s["bars"] for s in self.structure)
        beats_per_bar = self.time_signature[0]
        seconds_per_beat = 60.0 / self.tempo
        return total_bars * beats_per_bar * seconds_per_beat
    
    def get_total_bars(self) -> int:
        """Get total number of bars."""
        return sum(s["bars"] for s in self.structure)
    
    def get_section_at_bar(self, bar: int) -> Optional[Dict]:
        """Get the section containing a specific bar."""
        for section in self.structure:
            if section["start_bar"] <= bar < section["end_bar"]:
                return section
        return None
    
    def to_midi(self) -> bytes:
        """Convert arrangement to MIDI bytes."""
        try:
            import mido
            
            mid = mido.MidiFile(type=1, ticks_per_beat=480)
            
            # Tempo track
            tempo_track = mido.MidiTrack()
            mid.tracks.append(tempo_track)
            tempo_track.append(mido.MetaMessage(
                'set_tempo', 
                tempo=mido.bpm2tempo(self.tempo)
            ))
            tempo_track.append(mido.MetaMessage(
                'time_signature',
                numerator=self.time_signature[0],
                denominator=self.time_signature[1]
            ))
            
            # Add each track
            for track_data in self.tracks:
                track = mido.MidiTrack()
                mid.tracks.append(track)
                track.append(mido.MetaMessage('track_name', name=track_data["name"]))
                
                # Sort events by time
                events = sorted(
                    track_data.get("events", []),
                    key=lambda e: e.get("bar", 0) * 4 + e.get("beat", 1)
                )
                
                current_time = 0
                for event in events:
                    # Calculate absolute time
                    bar = event.get("bar", 0)
                    beat = event.get("beat", 1)
                    ticks = int((bar * 4 + (beat - 1)) * 480)
                    delta = ticks - current_time
                    
                    if delta < 0:
                        delta = 0
                    
                    # Handle different event types
                    if "pitch" in event:
                        pitch = event["pitch"]
                        velocity = event.get("velocity", 64)
                        duration = int(event.get("duration", 1.0) * 480)
                        
                        track.append(mido.Message(
                            'note_on', note=pitch, velocity=velocity, time=delta
                        ))
                        track.append(mido.Message(
                            'note_off', note=pitch, velocity=0, time=duration
                        ))
                        current_time = ticks + duration
                    
                    elif "pitches" in event:
                        pitches = event["pitches"]
                        velocity = event.get("velocity", 64)
                        duration = int(event.get("duration", 1.0) * 480)
                        
                        for i, pitch in enumerate(pitches):
                            track.append(mido.Message(
                                'note_on', note=pitch, velocity=velocity, 
                                time=delta if i == 0 else 0
                            ))
                        
                        for i, pitch in enumerate(pitches):
                            track.append(mido.Message(
                                'note_off', note=pitch, velocity=0,
                                time=duration if i == 0 else 0
                            ))
                        
                        current_time = ticks + duration
            
            # Write to bytes
            from io import BytesIO
            buffer = BytesIO()
            mid.save(file=buffer)
            return buffer.getvalue()
            
        except ImportError:
            raise ImportError("mido required for MIDI export: pip install mido")
    
    def save(self, path: str) -> str:
        """
        Save arrangement to file.
        
        Args:
            path: Output file path (.mid for MIDI)
            
        Returns:
            Path to saved file
        """
        if path.endswith(".mid") or path.endswith(".midi"):
            midi_data = self.to_midi()
            with open(path, "wb") as f:
                f.write(midi_data)
        else:
            # Save as JSON for other formats
            import json
            data = {
                "tempo": self.tempo,
                "key": self.key,
                "time_signature": self.time_signature,
                "emotion": self.emotion,
                "genre": self.genre,
                "structure": self.structure,
                "chords": self.chords,
                "tracks": self.tracks,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        
        return path
    
    def get_summary(self) -> str:
        """Get a text summary of the arrangement."""
        duration = self.get_duration_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        return f"""Arrangement Summary:
- Genre: {self.genre}
- Emotion: {self.emotion}
- Key: {self.key}
- Tempo: {self.tempo} BPM
- Duration: {minutes}:{seconds:02d}
- Bars: {self.get_total_bars()}
- Tracks: {len(self.tracks)}
- Sections: {', '.join(s['section'] for s in self.structure)}
"""
