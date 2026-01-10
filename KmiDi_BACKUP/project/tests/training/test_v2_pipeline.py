"""Tests for v2 training pipeline manifest builder."""

import json
import os
import tempfile
from pathlib import Path
import pytest


def test_manifest_builder_imports():
    """Test that manifest builder can be imported."""
    # Add scripts to path
    import sys
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / 'scripts'))
    
    from build_manifests import (
        find_audio_files,
        find_midi_files,
        match_audio_midi_pairs,
        extract_emotion_from_path,
        split_train_val,
    )
    
    assert callable(find_audio_files)
    assert callable(find_midi_files)
    assert callable(match_audio_midi_pairs)
    assert callable(extract_emotion_from_path)
    assert callable(split_train_val)


def test_emotion_extraction():
    """Test emotion extraction from file paths."""
    import sys
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / 'scripts'))
    
    from build_manifests import extract_emotion_from_path
    
    # Test directory-based emotion
    happy_path = Path('/data/audio/happy/track001.wav')
    emotion = extract_emotion_from_path(happy_path)
    assert len(emotion) == 3
    assert emotion[0] > 0  # Positive valence for happy
    
    # Test filename-based emotion
    sad_path = Path('/data/audio/tracks/track_sad_001.wav')
    emotion = extract_emotion_from_path(sad_path)
    assert len(emotion) == 3
    assert emotion[0] < 0  # Negative valence for sad
    
    # Test neutral default
    neutral_path = Path('/data/audio/unknown/track123.wav')
    emotion = extract_emotion_from_path(neutral_path)
    assert emotion == [0.0, 0.0, 0.5]


def test_train_val_split():
    """Test deterministic train/val splitting."""
    import sys
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / 'scripts'))
    
    from build_manifests import split_train_val
    
    items = [f'item_{i}' for i in range(100)]
    
    # Test split ratio
    train, val = split_train_val(items, val_split=0.1, seed=42)
    assert len(train) + len(val) == len(items)
    assert len(val) > 0
    assert len(val) < len(items)
    
    # Test determinism
    train2, val2 = split_train_val(items, val_split=0.1, seed=42)
    assert train == train2
    assert val == val2
    
    # Test different seed gives different split
    train3, val3 = split_train_val(items, val_split=0.1, seed=123)
    assert train != train3 or val != val3


def test_manifest_format():
    """Test manifest JSONL format."""
    # Create sample manifest entry
    entry = {
        'audio_path': '/path/to/audio.wav',
        'midi_path': '/path/to/track.mid',
        'emotion': [0.8, 0.7, 0.7]
    }
    
    # Verify it can be JSON serialized
    json_str = json.dumps(entry)
    parsed = json.loads(json_str)
    
    assert parsed['audio_path'] == entry['audio_path']
    assert parsed['midi_path'] == entry['midi_path']
    assert len(parsed['emotion']) == 3


def test_midi_tokenizer_vocab():
    """Test that MIDI tokenizer has correct vocab size."""
    import sys
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / 'training' / 'cuda_session'))
    
    from train_midi_generator import MIDITokenizer
    
    tokenizer = MIDITokenizer(vocab_size=388)
    
    # Verify vocab size
    assert tokenizer.vocab_size == 388
    
    # Verify special tokens are in correct range
    assert tokenizer.pad_token == 384
    assert tokenizer.bos_token == 385
    assert tokenizer.eos_token == 386
    assert tokenizer.bar_token == 387
    
    # Test encoding/decoding
    events = [
        {'type': 'bar'},
        {'type': 'note_on', 'note': 60, 'velocity': 80, 'time_in_bar': 0.0},
        {'type': 'note_on', 'note': 64, 'velocity': 72, 'time_in_bar': 0.25},
    ]
    
    tokens = tokenizer.encode(events)
    
    # Should have BOS, events, EOS
    assert tokens[0] == tokenizer.bos_token
    assert tokens[-1] == tokenizer.eos_token
    assert tokenizer.bar_token in tokens
    
    # All tokens should be in vocab range
    for token in tokens:
        assert 0 <= token < 388


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
