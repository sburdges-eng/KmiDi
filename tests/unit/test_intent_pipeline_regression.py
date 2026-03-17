"""
Regression tests for the Request → Intent pipeline.

Assert that emotion_map, imagery_texture, and BPM/tempo are preserved
through normalize → validate → expand. No partial intent fallback.
"""

import pytest

pytest.importorskip("fastapi", reason="FastAPI required for request models")

from music_brain.api import TechnicalIntent, EmotionalIntent, GenerateRequest  # noqa: E402
from music_brain.pipeline import IntentPipeline  # noqa: E402
from music_brain.session.intent_schema import CompleteSongIntent  # noqa: E402


def _make_request(
    emotional_intent: str = "joy",
    imagery_texture: str = "",
    bpm: int = 120,
    core_wound: str | None = None,
    core_desire: str | None = None,
    **tech_kw,
):
    tech = TechnicalIntent(
        key="C major",
        bpm=bpm,
        genre="pop",
        structure=[{"name": "verse", "bars": 8}],
        instruments=[{"instrument": "piano", "techniques": []}],
        **tech_kw,
    )
    emo = EmotionalIntent(
        core_wound=core_wound if core_wound is not None else "test event",
        core_desire=core_desire if core_desire is not None else "test desire",
        emotional_intent=emotional_intent,
        imagery_texture=imagery_texture,
        technical=tech,
    )
    return GenerateRequest(intent=emo)


class TestPipelineEmotionMap:
    """Emotion string → mood_primary via emotion_map in Stage 1."""

    def test_emotion_map_joy_to_tenderness(self):
        req = _make_request(emotional_intent="joy")
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert isinstance(intent, CompleteSongIntent)
        assert intent.song_intent.mood_primary == "tenderness"

    def test_emotion_map_grief_preserved(self):
        req = _make_request(emotional_intent="grief")
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert intent.song_intent.mood_primary == "grief"

    def test_emotion_map_anger_to_rage(self):
        req = _make_request(emotional_intent="anger and power")
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert intent.song_intent.mood_primary == "rage"


class TestPipelineImageryTexture:
    """imagery_texture from request preserved in expand (Stage 3)."""

    def test_imagery_texture_present(self):
        req = _make_request(imagery_texture="foggy and cold")
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert intent.song_intent.imagery_texture == "foggy and cold"

    def test_imagery_texture_empty_default(self):
        req = _make_request(imagery_texture="")
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert intent.song_intent.imagery_texture == ""


class TestPipelineTempoBpm:
    """BPM/tempo clamped and preserved through pipeline."""

    def test_tempo_clamped_in_range(self):
        req = _make_request(bpm=400)  # over 300
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        assert normalized["tempo"] == 300
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert validated.tempo == 300
        lo, hi = intent.technical_constraints.technical_tempo_range
        assert isinstance(lo, int) and isinstance(hi, int)

    def test_tempo_preserved_valid(self):
        req = _make_request(bpm=100)
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        lo, hi = intent.technical_constraints.technical_tempo_range
        assert 80 <= lo <= 120 <= hi
        assert validated.tempo == 100


class TestPipelineCoreEvent:
    """core_event extraction: core_wound or core_desire or emotional_intent."""

    def test_core_event_from_core_wound(self):
        req = _make_request(core_wound="loss", core_desire="peace")
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        assert intent.song_root.core_event == "loss"
        assert intent.song_root.core_longing == "peace"

    def test_core_event_fallback_to_core_desire(self):
        req = _make_request(core_wound="", core_desire="reconnection")  # empty wound → desire
        pipeline = IntentPipeline()
        normalized = pipeline.normalize(req)
        validated = pipeline.validate(normalized)
        intent = pipeline.expand(req, validated, normalized)
        # expand uses: core_wound or validated.core_desire or emotional_intent
        assert intent.song_root.core_event == "reconnection"
