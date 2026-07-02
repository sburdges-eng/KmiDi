"""Tests for POST /parse-text: natural language -> probabilistic music parameters."""

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from music_brain.api import app

    return TestClient(app)


RESPONSE_KEYS = {
    "param_distributions",
    "activated_clusters",
    "activated_taxonomy_ids",
    "detected_keywords",
    "confidence",
}


def test_parse_text_returns_interpretation(client):
    """A musical description yields distributions, clusters, and taxonomy ids."""
    r = client.post(
        "/parse-text", json={"text": "slow bluesy dorian, clean tone, laid back"}
    )
    assert r.status_code == 200
    body = r.json()
    assert RESPONSE_KEYS <= set(body.keys())
    assert "blues-slow" in [c["id"] for c in body["activated_clusters"]]
    assert body["param_distributions"]["tempo"]["center"] <= 100
    assert "harmony.scales.dorian" in body["activated_taxonomy_ids"]


def test_parse_text_missing_text_field(client):
    """Requests without the required text field are rejected."""
    r = client.post("/parse-text", json={"locale": "en"})
    assert r.status_code == 422


def test_parse_text_rejects_overlong_text(client):
    """Text beyond the 4096-char cap is rejected by validation."""
    r = client.post("/parse-text", json={"text": "x" * 4097})
    assert r.status_code == 422


def test_parse_text_empty_text_returns_defaults(client):
    """Empty text is not an error: defaults come back with no detections."""
    r = client.post("/parse-text", json={"text": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["detected_keywords"] == []
    assert "tempo" in body["param_distributions"]


def test_parse_text_non_musical_text_low_confidence(client):
    """Prose without musical terms yields low confidence but a usable payload."""
    r = client.post(
        "/parse-text", json={"text": "please file the expense report by friday"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] < 0.5
    assert body["detected_keywords"] == []
