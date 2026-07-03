"""Tests for music_brain.nlp: KeywordExtractor and TextToIntentService.

The NLP layer converts free-text music descriptions into probabilistic
parameter distributions. Core behavior under test: the same keyword
("dorian") yields different interpretations in different contexts
(metal shred vs slow blues), and the service degrades gracefully when
the optional emotion parser is unavailable.
"""

import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# KeywordExtractor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor():
    from music_brain.nlp.keyword_extractor import KeywordExtractor

    return KeywordExtractor()


def test_extracts_instruments(extractor):
    """Concrete instrument words are extracted with taxonomy hints."""
    result = extractor.extract("guitar and piano over drums")
    assert "guitar" in result.instruments
    assert "piano" in result.instruments
    assert "drums" in result.instruments
    assert "timbre.instruments.piano" in result.taxonomy_hints


def test_multi_word_keywords_take_priority(extractor):
    """ "electric guitar" matches as one keyword, not as bare "guitar"."""
    result = extractor.extract("electric guitar riff")
    assert "electric guitar" in result.all_matched
    assert "guitar" not in result.all_matched


def test_word_boundaries_respected(extractor):
    """Substrings inside larger words do not match (bassoon != bass)."""
    result = extractor.extract("the bassoon player")
    assert result.instruments == []
    assert result.all_matched == []


def test_categorizes_keywords(extractor):
    """Keywords land in their semantic category buckets."""
    result = extractor.extract("slow dorian blues")
    assert "slow" in result.tempo_words
    assert "dorian" in result.mode_words
    assert "blues" in result.genre_markers


def test_case_insensitive(extractor):
    """Extraction is case-insensitive."""
    result = extractor.extract("GUITAR with Overdrive")
    assert "guitar" in result.instruments
    assert "overdrive" in result.techniques


def test_empty_text(extractor):
    """Empty input produces empty results."""
    result = extractor.extract("")
    assert result.all_matched == []
    assert result.taxonomy_hints == []


# ---------------------------------------------------------------------------
# TextToIntentService
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def service():
    from music_brain.nlp.text_to_intent_service import TextToIntentService

    return TextToIntentService()


RESPONSE_KEYS = {
    "param_distributions",
    "activated_clusters",
    "activated_taxonomy_ids",
    "detected_keywords",
    "confidence",
}


def test_parse_returns_response_schema(service):
    """parse() returns the ParseTextResponse contract used by the frontend."""
    result = service.parse("slow bluesy dorian with clean bass")
    assert RESPONSE_KEYS <= set(result.keys())
    assert isinstance(result["param_distributions"], dict)
    assert isinstance(result["activated_clusters"], list)
    assert isinstance(result["activated_taxonomy_ids"], list)
    assert isinstance(result["detected_keywords"], list)
    assert 0.0 <= result["confidence"] <= 1.0


def test_parse_result_is_json_serializable(service):
    """The endpoint returns the parse dict verbatim, so it must serialize."""
    result = service.parse("fast metal guitar with distortion")
    json.dumps(result)


def test_metal_context_activates_shred_cluster(service):
    """Metal context: high-tempo cluster activates."""
    result = service.parse("fast metal shred on dorian with distorted guitar")
    cluster_ids = [c["id"] for c in result["activated_clusters"]]
    assert "metal-shred" in cluster_ids
    tempo = result["param_distributions"]["tempo"]
    assert tempo["type"] == "gaussian"
    assert tempo["center"] >= 140


def test_blues_context_activates_slow_cluster(service):
    """Blues context: laid-back low-tempo cluster activates."""
    result = service.parse("slow bluesy dorian, clean tone, laid back")
    cluster_ids = [c["id"] for c in result["activated_clusters"]]
    assert "blues-slow" in cluster_ids
    tempo = result["param_distributions"]["tempo"]
    assert tempo["center"] <= 100


def test_same_mode_word_interpreted_by_context(service):
    """Flagship behavior: "dorian" means shred or blues depending on context."""
    metal = service.parse("fast metal shred on dorian with distorted guitar")
    blues = service.parse("slow bluesy dorian, clean tone, laid back")
    metal_tempo = metal["param_distributions"]["tempo"]["center"]
    blues_tempo = blues["param_distributions"]["tempo"]["center"]
    assert metal_tempo > blues_tempo + 30


def test_mode_keywords_boost_mode_weights(service):
    """Explicitly named modes get boosted in mode_weights."""
    result = service.parse("a dorian groove")
    weights = result["param_distributions"]["mode_weights"]["weights"]
    assert weights.get("dorian", 0) > 0


def test_cluster_activation_entries_are_well_formed(service):
    """Each activated cluster carries id, label, and bounded confidence."""
    result = service.parse("slow bluesy dorian, clean tone, laid back")
    assert result["activated_clusters"], "expected at least one activated cluster"
    for cluster in result["activated_clusters"]:
        assert cluster["id"]
        assert cluster["label"]
        assert 0.0 <= cluster["confidence"] <= 1.0
    labels = [c["label"] for c in result["activated_clusters"]]
    assert "Slow Blues" in labels


def test_at_most_five_clusters(service):
    """Cluster activations are capped at the top 5."""
    text = (
        "fast slow metal blues jazz funk ambient pop punk reggae "
        "guitar bass piano synth drums distortion clean reverb "
        "dorian minor major pentatonic"
    )
    result = service.parse(text)
    assert len(result["activated_clusters"]) <= 5


def test_non_musical_text_yields_defaults(service):
    """Text with no musical keywords still returns usable default distributions."""
    result = service.parse("the quarterly report is due on thursday")
    assert result["detected_keywords"] == []
    assert result["confidence"] < 0.5
    dists = result["param_distributions"]
    for required in ("tempo", "mode_weights", "density"):
        assert required in dists


def test_degrades_without_emotion_parser(service, monkeypatch):
    """Service still fulfills the contract when optional emotion deps are absent."""
    from music_brain.nlp import text_to_intent_service as mod

    monkeypatch.setattr(mod, "_get_emotion_parser", lambda: None)
    monkeypatch.setattr(mod, "_get_emotional_mapping", lambda: None)
    result = service.parse("slow bluesy dorian, clean tone, laid back")
    assert RESPONSE_KEYS <= set(result.keys())
    assert "tempo" in result["param_distributions"]
    cluster_ids = [c["id"] for c in result["activated_clusters"]]
    assert "blues-slow" in cluster_ids


def test_param_distribution_to_dict_omits_unset_fields():
    """ParamDistribution.to_dict() emits only the fields relevant to its type."""
    from music_brain.nlp.text_to_intent_service import ParamDistribution

    gaussian = ParamDistribution("gaussian", center=120.0, spread=10.0)
    as_dict = gaussian.to_dict()
    assert as_dict == {"type": "gaussian", "center": 120.0, "spread": 10.0}
