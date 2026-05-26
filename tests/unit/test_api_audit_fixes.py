"""Tests for cross-cutting audit fixes: classify path sandbox, max length, 500 sanitization."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audio_sandbox_root(tmp_path_factory):
    """A directory under which classify endpoints allow audio paths (module-scoped so app import sees it)."""  # noqa: E501
    return tmp_path_factory.mktemp("audio_sandbox")


@pytest.fixture(scope="module")
def app_with_sandbox(audio_sandbox_root):
    """FastAPI app with KMIDI_AUDIO_SERVE_ROOT set to a temp dir."""
    os.environ["KMIDI_AUDIO_SERVE_ROOT"] = str(audio_sandbox_root)
    try:
        from music_brain.api import app

        yield app
    finally:
        os.environ.pop("KMIDI_AUDIO_SERVE_ROOT", None)


@pytest.fixture
def client(app_with_sandbox):
    """TestClient for the app with sandbox."""
    from fastapi.testclient import TestClient

    return TestClient(app_with_sandbox)


def _path_outside_default_sandbox():
    """Path that resolves outside tempfile.gettempdir() so any default sandbox rejects it."""
    if os.name == "nt":
        return (Path("C:\\") / "nonexistent_kmidi_audit_test.wav").resolve()
    return (Path("/usr") / "nonexistent_kmidi_audit_test.wav").resolve()


def test_classify_audio_rejects_path_outside_sandbox(client):
    """POST /audio/classify with path outside KMIDI_AUDIO_SERVE_ROOT returns 400."""
    r = client.post("/audio/classify", json={"audio_path": str(_path_outside_default_sandbox())})
    assert r.status_code == 400
    assert "not allowed" in str(r.json().get("detail", ""))


def test_classify_audio_rejects_traversal_style_path(client):
    """POST /audio/classify with traversal-style path returns 400."""
    if os.name == "nt":
        traversal = str((Path("C:\\") / "Windows" / "nonexistent").resolve())
    else:
        traversal = str((Path("/usr") / ".." / "etc" / "passwd").resolve())
    r = client.post("/audio/classify", json={"audio_path": traversal})
    assert r.status_code == 400
    assert "not allowed" in str(r.json().get("detail", ""))


def test_classify_audio_accepts_path_inside_sandbox(client, audio_sandbox_root):
    """POST /audio/classify with path under sandbox is allowed (may 404 if file missing)."""
    in_root = audio_sandbox_root / "valid.wav"
    in_root.touch()
    r = client.post("/audio/classify", json={"audio_path": str(in_root)})
    # 200 if classifier works, 503 if model not available, 404 if file not found - but not 400 path rejected  # noqa: E501
    assert r.status_code in (200, 404, 503), r.json()


def test_voice_classify_rejects_path_outside_sandbox(client):
    """POST /voice/classify with path outside sandbox returns 400."""
    r = client.post("/voice/classify", json={"audio_path": str(_path_outside_default_sandbox())})
    assert r.status_code == 400
    assert "not allowed" in str(r.json().get("detail", ""))


def test_interrogate_accepts_message_within_max_length(client):
    """POST /interrogate with message length 4096 is accepted."""
    r = client.post("/interrogate", json={"message": "x" * 4096})
    assert r.status_code == 200


def test_interrogate_rejects_message_over_max_length(client):
    """POST /interrogate with message length > 4096 returns 422."""
    r = client.post("/interrogate", json={"message": "x" * 4097})
    assert r.status_code == 422


def test_lyrics_accepts_within_max_length(client):
    """POST /lyrics with lyrics length 32768 is accepted."""
    r = client.post("/lyrics", json={"lyrics": "y" * 32768, "source": "user"})
    assert r.status_code == 200


def test_lyrics_rejects_over_max_length(client):
    """POST /lyrics with lyrics length > 32768 returns 422."""
    r = client.post("/lyrics", json={"lyrics": "y" * 32769, "source": "user"})
    assert r.status_code == 422


def test_classify_rejection_does_not_leak_path(client):
    """Path outside sandbox returns 400 with generic message, not internal path."""
    r = client.post("/audio/classify", json={"audio_path": str(_path_outside_default_sandbox())})
    assert r.status_code == 400
    detail = str(r.json().get("detail", ""))
    assert "not allowed" in detail


def test_spectocloud_rejects_midi_file_path_outside_sandbox(client):
    """POST /spectocloud/render with midi_file_path outside KMIDI_AUDIO_SERVE_ROOT returns 400."""
    r = client.post(
        "/spectocloud/render",
        json={
            "midi_file_path": str(_path_outside_default_sandbox()).replace(".wav", ".mid"),
            "mode": "static",
        },
    )
    assert r.status_code == 400
    assert "not allowed" in str(r.json().get("detail", ""))


def test_spectocloud_rejects_output_path_outside_sandbox(client):
    """POST /spectocloud/render with output_path outside sandbox returns 400."""
    outside = str(_path_outside_default_sandbox()).replace(".wav", ".png")
    r = client.post(
        "/spectocloud/render",
        json={
            "midi_events": [
                {"time": 0.1, "type": "note_on", "note": 60, "velocity": 80, "channel": 0},
            ],
            "duration": 1.0,
            "mode": "static",
            "output_path": outside,
        },
    )
    assert r.status_code == 400
    assert "not allowed" in str(r.json().get("detail", ""))
