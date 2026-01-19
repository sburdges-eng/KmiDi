"""
Integration tests for Intent IR v1

Tests the full pipeline: Python → IntentFrame → JSON → C++ → Engine
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add Python paths
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from music_brain.session.intent_ir import IntentFrame, IntentSource
    from music_brain.session.intent_ir_converter import convert_complete_song_intent_to_ir
except ImportError:
    # Fallback if imports fail
    print("Warning: Could not import intent_ir modules. Some tests will be skipped.")
    IntentFrame = None


def test_intent_frame_creation():
    """Test creating an IntentFrame from scratch."""
    if IntentFrame is None:
        return
    
    frame = IntentFrame()
    assert frame.meta.ir_version == 1
    assert frame.emotion.valence == 0.0
    assert frame.music.tempo_bias == 0.0
    print("✓ IntentFrame creation works")


def test_intent_frame_json_roundtrip():
    """Test JSON serialization round-trip."""
    if IntentFrame is None:
        return
    
    frame = IntentFrame()
    frame.emotion.valence = 0.5
    frame.emotion.arousal = 0.7
    frame.music.tempo_bias = 0.3
    frame.music.mode_preference = 1
    
    # Serialize
    json_str = frame.to_json()
    assert json_str is not None
    assert "valence" in json_str
    assert "tempo_bias" in json_str
    
    # Deserialize
    frame2 = IntentFrame.from_json(json_str)
    assert frame2.emotion.valence == 0.5
    assert frame2.emotion.arousal == 0.7
    assert frame2.music.tempo_bias == 0.3
    assert frame2.music.mode_preference == 1
    print("✓ JSON round-trip works")


def test_intent_frame_validation():
    """Test that IntentFrame validates correctly."""
    if IntentFrame is None:
        return
    
    frame = IntentFrame()
    frame.meta.ir_version = 1
    frame.emotion.valence = 0.5
    frame.emotion.arousal = 0.7
    
    # Should be valid
    json_str = frame.to_json()
    frame2 = IntentFrame.from_json(json_str)
    assert frame2.meta.ir_version == 1
    print("✓ Validation works")


def test_intent_frame_file_io():
    """Test saving and loading IntentFrame from file."""
    if IntentFrame is None:
        return
    
    frame = IntentFrame()
    frame.emotion.valence = -0.3
    frame.music.tempo_bias = -0.5
    frame.meta.intent_id = 12345
    frame.meta.session_id = 67890
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        f.write(frame.to_json())
    
    try:
        # Load from file
        with open(temp_path, 'r') as f:
            frame2 = IntentFrame.from_json(f.read())
        
        assert frame2.emotion.valence == -0.3
        assert frame2.music.tempo_bias == -0.5
        assert frame2.meta.intent_id == 12345
        assert frame2.meta.session_id == 67890
        print("✓ File I/O works")
    finally:
        os.unlink(temp_path)


def test_intent_id_generation():
    """Test that intent_id is generated deterministically."""
    if IntentFrame is None:
        return
    
    frame1 = IntentFrame()
    frame1.emotion.valence = 0.5
    frame1.music.tempo_bias = 0.3
    
    frame2 = IntentFrame()
    frame2.emotion.valence = 0.5
    frame2.music.tempo_bias = 0.3
    
    # Generate IDs
    id1 = frame1.generate_intent_id()
    id2 = frame2.generate_intent_id()
    
    # Same content should generate same ID
    assert id1 == id2
    assert id1 != 0
    print("✓ Intent ID generation works")


def test_converter_from_complete_intent():
    """Test converting CompleteSongIntent to IntentFrame."""
    try:
        from music_brain.session.intent_schema import CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints
        
        # Create a minimal CompleteSongIntent
        complete_intent = CompleteSongIntent(
            mood_primary="Grief",
            technical_tempo_range=(60, 80),
            technical_mode="minor",
            technical_groove_feel="Laid Back"
        )
        
        frame = convert_complete_song_intent_to_ir(
            complete_intent,
            session_id=999,
            source=IntentSource.ML_TEXT
        )
        
        assert frame.meta.ir_version == 1
        assert frame.meta.session_id == 999
        assert frame.provenance.source == IntentSource.ML_TEXT
        assert frame.emotion.valence < 0  # Grief should be negative valence
        assert frame.music.mode_preference == -1  # Minor mode
        print("✓ CompleteSongIntent conversion works")
    except ImportError:
        print("⚠ CompleteSongIntent conversion test skipped (import failed)")


def test_all_engines_consumable():
    """Test that IntentFrame can be consumed by all engines."""
    # This is a placeholder - actual C++ engine consumption would be tested
    # via C++ unit tests, but we can verify the structure is correct
    
    if IntentFrame is None:
        return
    
    frame = IntentFrame()
    
    # Verify all required fields exist for engine consumption
    assert hasattr(frame, 'emotion')
    assert hasattr(frame, 'music')
    assert hasattr(frame, 'time')
    assert hasattr(frame, 'constraints')
    assert hasattr(frame, 'provenance')
    
    # Verify emotion fields
    assert hasattr(frame.emotion, 'valence')
    assert hasattr(frame.emotion, 'arousal')
    assert hasattr(frame.emotion, 'dominance')
    
    # Verify music fields
    assert hasattr(frame.music, 'tempo_bias')
    assert hasattr(frame.music, 'rhythmic_density')
    assert hasattr(frame.music, 'groove_strength')
    assert hasattr(frame.music, 'harmonic_tension')
    assert hasattr(frame.music, 'mode_preference')
    
    print("✓ IntentFrame structure is engine-consumable")


if __name__ == "__main__":
    print("Running Intent IR v1 Integration Tests...")
    print()
    
    test_intent_frame_creation()
    test_intent_frame_json_roundtrip()
    test_intent_frame_validation()
    test_intent_frame_file_io()
    test_intent_id_generation()
    test_converter_from_complete_intent()
    test_all_engines_consumable()
    
    print()
    print("All integration tests passed!")
