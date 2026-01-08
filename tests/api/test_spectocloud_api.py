import json
from pathlib import Path

import pytest

# Set matplotlib backend to non-interactive for testing
import matplotlib
matplotlib.use('Agg')

api_module = pytest.importorskip("music_brain.api")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture(scope="module")
def client():
    if not hasattr(api_module, "app"):
        pytest.skip("FastAPI not available")
    return TestClient(api_module.app)


def test_humanizer_reload(client):
    resp = client.post("/config/humanizer/reload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_spectocloud_render_validation(client):
    # Missing midi_events and midi_file_path
    resp = client.post("/spectocloud/render", json={"duration": 1.0})
    assert resp.status_code == 400

    # Invalid fps
    resp = client.post(
        "/spectocloud/render",
        json={"midi_events": [{"time": 0, "type": "note_on", "note": 60, "velocity": 90}], "duration": 1.0, "fps": 0},
    )
    assert resp.status_code == 400


def test_spectocloud_render_static(client, tmp_path):
    payload = {
        "midi_events": [
            {"time": 0.0, "type": "note_on", "note": 60, "velocity": 90},
            {"time": 0.5, "type": "note_on", "note": 62, "velocity": 85},
        ],
        "duration": 1.0,
        "mode": "static",
        "frame_idx": 0,
        "output_path": str(tmp_path / "frame.png"),
    }
    resp = client.post("/spectocloud/render", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "static"
    assert Path(data["output_path"]).exists()


def test_spectocloud_render_animation(client, tmp_path):
    payload = {
        "midi_events": [
            {"time": 0.0, "type": "note_on", "note": 60, "velocity": 90},
            {"time": 0.5, "type": "note_on", "note": 62, "velocity": 85},
        ],
        "duration": 1.0,
        "mode": "animation",
        "fps": 8,
        "rotate": False,
        "output_path": str(tmp_path / "anim.gif"),
    }
    resp = client.post("/spectocloud/render", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "animation"
    assert Path(data["output_path"]).exists()
