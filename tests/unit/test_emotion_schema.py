import pytest
from music_brain.engine_api.schema import EmotionStateSchema


def test_valid_neutral():
    e = EmotionStateSchema(valence=0.0, arousal=0.5, dominance=0.5, confidence=0.5)
    assert e.valence == 0.0
    assert e.arousal == 0.5
    assert e.dominance == 0.5
    assert e.tags == []
    assert e.confidence == 0.5


def test_valid_with_tags():
    e = EmotionStateSchema(
        valence=0.8, arousal=0.9, dominance=0.7,
        tags=["bright", "drive"], confidence=0.9
    )
    assert len(e.tags) == 2


def test_invalid_valence_out_of_range():
    with pytest.raises(Exception):
        EmotionStateSchema(valence=2.0, arousal=0.5, dominance=0.5, confidence=0.5)


def test_invalid_unknown_tag():
    with pytest.raises(Exception):
        EmotionStateSchema(
            valence=0.0, arousal=0.5, dominance=0.5,
            tags=["angry"], confidence=0.5
        )


def test_invalid_too_many_tags():
    with pytest.raises(Exception):
        EmotionStateSchema(
            valence=0.0, arousal=0.5, dominance=0.5,
            tags=["tension", "release", "warm", "cold"], confidence=0.5
        )


def test_invalid_extra_field():
    with pytest.raises(Exception):
        EmotionStateSchema(
            valence=0.0, arousal=0.5, dominance=0.5,
            confidence=0.5, intensity=0.5
        )
