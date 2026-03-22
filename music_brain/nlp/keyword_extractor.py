"""
Musical Keyword Extractor

Extracts musical keywords from free-text descriptions. Unlike the emotion
parser (which handles feelings), this extracts concrete musical terms:
instruments, techniques, tempo/density modifiers, genre markers.

These keywords feed into context cluster matching — "overdrive" alone
doesn't set a parameter, but "overdrive + fast + guitar" activates the
metal-shred cluster which shifts everything.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class ExtractedKeywords:
    """Result of keyword extraction from text."""
    instruments: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    tempo_words: List[str] = field(default_factory=list)
    density_words: List[str] = field(default_factory=list)
    genre_markers: List[str] = field(default_factory=list)
    feel_words: List[str] = field(default_factory=list)
    mode_words: List[str] = field(default_factory=list)
    structure_words: List[str] = field(default_factory=list)
    dynamic_words: List[str] = field(default_factory=list)
    all_matched: List[str] = field(default_factory=list)
    taxonomy_hints: List[str] = field(default_factory=list)


# Word → (category, canonical_form, taxonomy_node_id)
KEYWORD_MAP: Dict[str, Tuple[str, str, str]] = {}

_KEYWORD_GROUPS: Dict[str, Dict[str, str]] = {
    # instrument words → taxonomy node ID
    "instruments": {
        "guitar": "timbre.instruments.electric-guitar",
        "electric guitar": "timbre.instruments.electric-guitar",
        "acoustic guitar": "timbre.instruments.acoustic-guitar",
        "acoustic": "timbre.instruments.acoustic-guitar",
        "bass": "timbre.instruments.bass-guitar",
        "bass guitar": "timbre.instruments.bass-guitar",
        "upright bass": "timbre.instruments.upright-bass",
        "piano": "timbre.instruments.piano",
        "keys": "timbre.instruments.piano",
        "keyboard": "timbre.instruments.piano",
        "electric piano": "timbre.instruments.electric-piano",
        "rhodes": "timbre.instruments.electric-piano",
        "wurlitzer": "timbre.instruments.electric-piano",
        "organ": "timbre.instruments.organ",
        "synth": "timbre.instruments.synth-lead",
        "synthesizer": "timbre.instruments.synth-lead",
        "synth lead": "timbre.instruments.synth-lead",
        "synth pad": "timbre.instruments.synth-pad",
        "pad": "timbre.instruments.synth-pad",
        "synth bass": "timbre.instruments.synth-bass",
        "strings": "timbre.instruments.strings",
        "violin": "timbre.instruments.violin",
        "cello": "timbre.instruments.cello",
        "brass": "timbre.instruments.brass",
        "trumpet": "timbre.instruments.trumpet",
        "horn": "timbre.instruments.brass",
        "sax": "timbre.instruments.sax",
        "saxophone": "timbre.instruments.sax",
        "flute": "timbre.instruments.flute",
        "woodwinds": "timbre.instruments.woodwinds",
        "drums": "timbre.instruments.drums",
        "percussion": "timbre.instruments.percussion",
        "choir": "timbre.instruments.choir",
        "vocals": "timbre.instruments.choir",
        "808": "timbre.instruments.808",
        "harp": "timbre.instruments.harp",
        "marimba": "timbre.instruments.marimba",
        "vibraphone": "timbre.instruments.vibraphone",
        "vibes": "timbre.instruments.vibraphone",
        "fiddle": "timbre.instruments.violin",
        "pedal steel": "timbre.instruments.electric-guitar",
    },
    "techniques": {
        "overdrive": "timbre.effects.overdrive",
        "distortion": "timbre.effects.distortion",
        "distorted": "timbre.effects.distortion",
        "clean": "timbre.effects.clean",
        "reverb": "timbre.effects.reverb",
        "delay": "timbre.effects.delay",
        "echo": "timbre.effects.delay",
        "chorus": "timbre.effects.chorus-effect",
        "flanger": "timbre.effects.flanger",
        "phaser": "timbre.effects.phaser",
        "wah": "timbre.effects.wah",
        "compression": "timbre.effects.compression",
        "compressed": "timbre.effects.compression",
        "bitcrushed": "timbre.effects.bitcrusher",
        "lo-fi": "timbre.effects.lo-fi",
        "lofi": "timbre.effects.lo-fi",
        "staccato": "articulation.staccato",
        "legato": "articulation.legato",
        "portamento": "articulation.portamento",
        "glissando": "articulation.glissando",
        "vibrato": "articulation.vibrato",
        "tremolo": "articulation.tremolo",
        "pizzicato": "articulation.pizzicato",
        "arco": "articulation.arco",
        "muted": "articulation.muted",
        "palm mute": "articulation.palm-mute",
        "palm-muted": "articulation.palm-mute",
        "hammer-on": "articulation.hammer-on",
        "pull-off": "articulation.pull-off",
        "tapping": "articulation.tapping",
        "fingerpicking": "articulation.fingerpicking",
        "fingerpicked": "articulation.fingerpicking",
        "strumming": "articulation.strumming",
        "strummed": "articulation.strumming",
        "slap": "articulation.slap",
        "slap bass": "articulation.slap",
        "harmonics": "articulation.harmonics",
        "bend": "articulation.bend",
        "bending": "articulation.bend",
        "slide": "articulation.slide",
        "power chords": "harmony.chords.power-chord",
        "power chord": "harmony.chords.power-chord",
        "arpeggiated": "melody.contour.wave",
        "arpeggio": "melody.contour.wave",
    },
    "tempo_words": {
        "fast": "rhythm.density.dense",
        "quick": "rhythm.density.dense",
        "rapid": "rhythm.density.dense",
        "driving": "rhythm.groove-feel.straight-driving",
        "rushing": "rhythm.groove-feel.straight-driving",
        "uptempo": "rhythm.density.dense",
        "slow": "rhythm.density.sparse",
        "laid-back": "rhythm.groove-feel.laid-back",
        "laid back": "rhythm.groove-feel.laid-back",
        "crawling": "rhythm.density.sparse",
        "plodding": "rhythm.density.sparse",
        "moderate": "rhythm.density.medium",
        "mid-tempo": "rhythm.density.medium",
        "medium": "rhythm.density.medium",
        "breakneck": "rhythm.density.dense",
        "blistering": "rhythm.density.dense",
        "relaxed": "rhythm.groove-feel.laid-back",
        "chill": "rhythm.groove-feel.laid-back",
        "gentle": "rhythm.density.sparse",
    },
    "density_words": {
        "busy": "rhythm.density.dense",
        "sparse": "rhythm.density.sparse",
        "thick": "texture.density.thick",
        "thin": "texture.density.thin",
        "minimal": "texture.density.monophonic",
        "minimalist": "texture.density.monophonic",
        "dense": "rhythm.density.dense",
        "lush": "texture.density.thick",
        "full": "texture.density.thick",
        "empty": "texture.density.thin",
        "spacious": "texture.space.hall",
        "massive": "texture.density.massive",
        "wall of sound": "texture.density.massive",
        "stripped": "texture.density.thin",
        "bare": "texture.density.monophonic",
        "layered": "texture.density.thick",
    },
    "genre_markers": {
        "bluesy": "harmony.scales.blues",
        "blues": "harmony.scales.blues",
        "jazzy": "harmony.scales.dorian",
        "jazz": "harmony.scales.dorian",
        "metal": "harmony.scales.phrygian",
        "punk": "harmony.scales.minor",
        "ambient": "timbre.instruments.synth-pad",
        "funky": "rhythm.groove-feel.syncopated",
        "funk": "rhythm.groove-feel.syncopated",
        "classical": "timbre.instruments.strings",
        "orchestral": "timbre.instruments.strings",
        "cinematic": "timbre.instruments.strings",
        "electronic": "timbre.instruments.synth-lead",
        "edm": "timbre.instruments.synth-lead",
        "house": "rhythm.patterns.four-on-floor",
        "techno": "rhythm.patterns.four-on-floor",
        "hip-hop": "rhythm.patterns.boom-bap",
        "hip hop": "rhythm.patterns.boom-bap",
        "trap": "timbre.instruments.808",
        "r&b": "rhythm.groove-feel.laid-back",
        "rnb": "rhythm.groove-feel.laid-back",
        "soul": "rhythm.groove-feel.laid-back",
        "reggae": "rhythm.syncopation.offbeat",
        "country": "timbre.instruments.acoustic-guitar",
        "folk": "timbre.instruments.acoustic-guitar",
        "indie": "timbre.instruments.acoustic-guitar",
        "pop": "harmony.scales.major",
        "rock": "timbre.effects.overdrive",
        "grunge": "timbre.effects.distortion",
        "prog": "rhythm.time-signatures.7-8",
        "progressive": "rhythm.time-signatures.7-8",
        "latin": "rhythm.patterns.clave",
        "salsa": "rhythm.patterns.clave",
        "bossa nova": "rhythm.swing.light-swing",
        "bossa": "rhythm.swing.light-swing",
        "gospel": "timbre.instruments.organ",
        "synthwave": "timbre.instruments.synth-lead",
        "retrowave": "timbre.instruments.synth-lead",
        "lo-fi hip hop": "timbre.effects.lo-fi",
        "boom bap": "rhythm.patterns.boom-bap",
        "drill": "timbre.instruments.808",
        "dnb": "rhythm.density.dense",
        "drum and bass": "rhythm.density.dense",
        "dubstep": "timbre.instruments.synth-bass",
        "neo-soul": "rhythm.groove-feel.laid-back",
    },
    "feel_words": {
        "groovy": "rhythm.groove-feel.syncopated",
        "tight": "rhythm.groove-feel.mechanical",
        "loose": "rhythm.groove-feel.organic",
        "swinging": "rhythm.swing.heavy-swing",
        "swing": "rhythm.swing.heavy-swing",
        "swung": "rhythm.swing.heavy-swing",
        "straight": "rhythm.swing.straight",
        "mechanical": "rhythm.groove-feel.mechanical",
        "robotic": "rhythm.groove-feel.mechanical",
        "organic": "rhythm.groove-feel.organic",
        "human": "rhythm.groove-feel.organic",
        "natural": "rhythm.groove-feel.organic",
        "breathing": "rhythm.groove-feel.organic",
        "syncopated": "rhythm.syncopation.heavy",
        "offbeat": "rhythm.syncopation.offbeat",
        "free": "rhythm.groove-feel.rubato",
        "rubato": "rhythm.groove-feel.rubato",
        "pushing": "rhythm.groove-feel.push-pull",
        "pulling": "rhythm.groove-feel.push-pull",
        "shuffle": "rhythm.patterns.shuffle",
        "shuffling": "rhythm.patterns.shuffle",
        "bouncy": "rhythm.swing.light-swing",
    },
    "mode_words": {
        "dorian": "harmony.scales.dorian",
        "phrygian": "harmony.scales.phrygian",
        "lydian": "harmony.scales.lydian",
        "mixolydian": "harmony.scales.mixolydian",
        "aeolian": "harmony.scales.minor",
        "locrian": "harmony.scales.locrian",
        "major": "harmony.scales.major",
        "minor": "harmony.scales.minor",
        "pentatonic": "harmony.scales.pentatonic-minor",
        "minor pentatonic": "harmony.scales.pentatonic-minor",
        "major pentatonic": "harmony.scales.pentatonic-major",
        "blues scale": "harmony.scales.blues",
        "chromatic": "harmony.scales.chromatic",
        "whole tone": "harmony.scales.whole-tone",
        "harmonic minor": "harmony.scales.harmonic-minor",
        "melodic minor": "harmony.scales.melodic-minor",
        "diminished": "harmony.scales.diminished",
    },
    "dynamic_words": {
        "quiet": "dynamics.levels.p",
        "soft": "dynamics.levels.pp",
        "gentle": "dynamics.levels.pp",
        "whisper": "dynamics.levels.pp",
        "delicate": "dynamics.levels.pp",
        "moderate volume": "dynamics.levels.mf",
        "loud": "dynamics.levels.f",
        "powerful": "dynamics.levels.ff",
        "aggressive": "dynamics.levels.ff",
        "intense": "dynamics.levels.ff",
        "explosive": "dynamics.levels.ff",
        "crescendo": "dynamics.changes.crescendo",
        "building": "dynamics.changes.crescendo",
        "fading": "dynamics.changes.decrescendo",
        "dying": "dynamics.changes.decrescendo",
        "diminishing": "dynamics.changes.decrescendo",
        "swelling": "dynamics.changes.swell",
    },
    "structure_words": {
        "intro": "structure.sections.intro",
        "verse": "structure.sections.verse",
        "chorus": "structure.sections.chorus",
        "bridge": "structure.sections.bridge",
        "outro": "structure.sections.outro",
        "build": "structure.sections.build",
        "buildup": "structure.sections.build",
        "drop": "structure.sections.drop",
        "breakdown": "structure.sections.breakdown",
        "solo": "structure.sections.solo",
    },
}

# Build flat lookup: lowercase word → (category, canonical_word, taxonomy_id)
for _cat, _words in _KEYWORD_GROUPS.items():
    for _word, _tax_id in _words.items():
        KEYWORD_MAP[_word.lower()] = (_cat, _word, _tax_id)


class KeywordExtractor:
    """
    Extract musical keywords from free-text descriptions.

    Scans text for known musical terms (instruments, techniques, genres, etc.)
    and returns structured results with taxonomy node ID hints.
    """

    def __init__(self):
        # Sort keywords by length descending so multi-word matches take priority
        self._sorted_keywords = sorted(
            KEYWORD_MAP.keys(), key=len, reverse=True
        )

    def extract(self, text: str) -> ExtractedKeywords:
        """Extract all musical keywords from text."""
        result = ExtractedKeywords()
        lower_text = f" {text.lower()} "
        matched_spans: Set[Tuple[int, int]] = set()

        for keyword in self._sorted_keywords:
            # Search for keyword as a whole word
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, lower_text):
                start, end = match.start(), match.end()
                # Skip if this span overlaps with an already-matched span
                if any(
                    not (end <= ms or start >= me)
                    for ms, me in matched_spans
                ):
                    continue

                matched_spans.add((start, end))
                category, canonical, taxonomy_id = KEYWORD_MAP[keyword]

                result.all_matched.append(canonical)
                result.taxonomy_hints.append(taxonomy_id)

                if category == "instruments":
                    result.instruments.append(canonical)
                elif category == "techniques":
                    result.techniques.append(canonical)
                elif category == "tempo_words":
                    result.tempo_words.append(canonical)
                elif category == "density_words":
                    result.density_words.append(canonical)
                elif category == "genre_markers":
                    result.genre_markers.append(canonical)
                elif category == "feel_words":
                    result.feel_words.append(canonical)
                elif category == "mode_words":
                    result.mode_words.append(canonical)
                elif category == "dynamic_words":
                    result.dynamic_words.append(canonical)
                elif category == "structure_words":
                    result.structure_words.append(canonical)

        return result
