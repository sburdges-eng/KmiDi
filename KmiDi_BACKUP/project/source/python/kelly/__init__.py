"""Kelly - Therapeutic iDAW translating emotions to music.

Kelly is an Intelligent Digital Audio Workstation (iDAW) that processes
therapeutic intent through three phases:
1. Wound identification
2. Emotional mapping using a 216-node emotion thesaurus
3. Musical rule-breaking for authentic expression

The system translates emotional wounds into musical expression through
intentional rule-breaking, allowing authentic representation of difficult
emotions in musical form.

Example:
    >>> from kelly import EmotionThesaurus, IntentProcessor, MidiGenerator
    >>> 
    >>> # Process a wound
    >>> processor = IntentProcessor()
    >>> from kelly.core.intent_processor import Wound
    >>> wound = Wound("feeling of loss", 0.9, "user_input")
    >>> result = processor.process_intent(wound)
    >>> 
    >>> # Generate MIDI
    >>> generator = MidiGenerator(tempo=120)
    >>> chords = generator.generate_chord_progression(mode="minor")
    >>> midi_file = generator.create_midi_file(chords, output_path="output.mid")
"""

__version__ = "0.1.0"
__author__ = "Kelly Development Team"
__description__ = "Therapeutic iDAW translating emotions to music"

# Core imports
from kelly.core.emotion_thesaurus import (
    EmotionThesaurus,
    EmotionNode,
    EmotionCategory,
)
from kelly.core.intent_processor import (
    IntentProcessor,
    Wound,
    RuleBreak,
    IntentPhase,
)
from kelly.core.midi_generator import (
    MidiGenerator,
    GrooveTemplate,
    Chord,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    # Emotion thesaurus
    "EmotionThesaurus",
    "EmotionNode",
    "EmotionCategory",
    # Intent processing
    "IntentProcessor",
    "Wound",
    "RuleBreak",
    "IntentPhase",
    # MIDI generation
    "MidiGenerator",
    "GrooveTemplate",
    "Chord",
]
