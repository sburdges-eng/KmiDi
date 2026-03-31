import json
from pathlib import Path

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


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "intent"

VALID_FIXTURES = [
    "emotion_valid_neutral.json",
    "emotion_valid_excited.json",
    "emotion_valid_sad.json",
    "emotion_valid_max_tags.json",
    "emotion_valid_no_tags.json",
]

INVALID_FIXTURES = [
    "emotion_invalid_valence_oob.json",
    "emotion_invalid_tag_unknown.json",
    "emotion_invalid_too_many_tags.json",
    "emotion_invalid_extra_field.json",
]


@pytest.mark.parametrize("fixture_name", VALID_FIXTURES)
def test_valid_fixture_accepted(fixture_name):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text())
    e = EmotionStateSchema(**data)
    assert -1.0 <= e.valence <= 1.0
    assert 0.0 <= e.arousal <= 1.0
    assert 0.0 <= e.dominance <= 1.0
    assert 0.0 <= e.confidence <= 1.0
    assert len(e.tags) <= 3


@pytest.mark.parametrize("fixture_name", INVALID_FIXTURES)
def test_invalid_fixture_rejected(fixture_name):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text())
    with pytest.raises(Exception):
        EmotionStateSchema(**data)
