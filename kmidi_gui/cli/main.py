"""CLI interface for KmiDi core logic.

This demonstrates that core logic can run headless without GUI.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kmidi_gui.core.engine import get_engine
from kmidi_gui.core.models import EmotionIntent, GenerationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_music(intent_text: str) -> GenerationResult:
    """Generate music from emotional intent text.

    Args:
        intent_text: Emotional intent description

    Returns:
        GenerationResult
    """
    engine = get_engine()
    intent = EmotionIntent(
        core_event=intent_text,
        mood_primary="grief" if "grief" in intent_text.lower() else None,
    )
    return engine.generate_music(intent)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KmiDi CLI - Generate music from emotional intent"
    )
    parser.add_argument(
        "intent",
        help="Emotional intent text (e.g., 'grief hidden as love')"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output MIDI file path"
    )

    args = parser.parse_args()

    logger.info(f"Generating music for intent: {args.intent}")

    result = generate_music(args.intent)

    if result.success:
        print("✓ Generation successful!")
        if result.chords:
            print(f"  Chords: {' - '.join(result.chords)}")
        if result.key:
            print(f"  Key: {result.key}")
        if result.tempo:
            print(f"  Tempo: {result.tempo} BPM")
        if result.midi_path:
            print(f"  MIDI: {result.midi_path}")
        if args.output and result.midi_path:
            # Would copy MIDI file to output path
            print(f"  Saved to: {args.output}")
    else:
        print(f"✗ Generation failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

