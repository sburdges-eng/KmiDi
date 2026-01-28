"""
Kelly Companion Data Package

Contains emotion data, chord progressions, genre maps, and intent schemas.
"""

from pathlib import Path

# Data directory paths
DATA_DIR = Path(__file__).parent
EMOTIONS_DIR = DATA_DIR / "emotions"
CHORDS_DIR = DATA_DIR / "chords"
GENRES_DIR = DATA_DIR / "genres"

__all__ = ["DATA_DIR", "EMOTIONS_DIR", "CHORDS_DIR", "GENRES_DIR"]
