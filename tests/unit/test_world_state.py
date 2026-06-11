"""WorldState — persistent musical session state with history (Goal:
Persistent latent world-state modeling, Longitudinal user modeling,
Arrangement continuity preservation)."""

import pytest
from pathlib import Path

from music_brain.session.world_state import WorldState


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


def test_world_state_starts_with_defaults():
    state = WorldState()
    assert state.key == ""
    assert state.tempo == 0
    assert state.current_section is None
    assert state.current_emotion is None
    assert state.history == []


def test_world_state_accepts_initial_values():
    state = WorldState(key="C major", tempo=120, current_section="intro")
    assert state.key == "C major"
    assert state.tempo == 120
    assert state.current_section == "intro"


# ---------------------------------------------------------------------------
# update_section / update_emotion record history
# ---------------------------------------------------------------------------


def test_update_section_appends_to_history():
    state = WorldState(current_section="intro")
    state.update_section("verse")
    assert state.current_section == "verse"
    assert state.history[-1] == {"kind": "section", "from": "intro", "to": "verse"}


def test_update_emotion_appends_to_history():
    state = WorldState()
    state.update_emotion("tension")
    assert state.current_emotion == "tension"
    assert state.history[-1] == {"kind": "emotion", "from": None, "to": "tension"}


def test_no_op_updates_dont_add_history():
    """Updating to the same value should not bloat the history log."""
    state = WorldState(current_section="verse")
    state.update_section("verse")
    assert state.history == []


def test_history_preserves_chronological_order():
    state = WorldState()
    state.update_section("intro")
    state.update_emotion("hope")
    state.update_section("chorus")
    kinds = [entry["kind"] for entry in state.history]
    assert kinds == ["section", "emotion", "section"]


# ---------------------------------------------------------------------------
# transition_count helpers
# ---------------------------------------------------------------------------


def test_section_transition_count_zero_initially():
    state = WorldState()
    assert state.section_transition_count() == 0


def test_section_transition_count_increments_per_change():
    state = WorldState()
    state.update_section("intro")
    state.update_section("verse")
    state.update_section("chorus")
    assert state.section_transition_count() == 3


def test_recent_sections_returns_last_n_in_order():
    state = WorldState()
    for section in ["intro", "verse", "chorus", "verse", "outro"]:
        state.update_section(section)
    assert state.recent_sections(3) == ["chorus", "verse", "outro"]


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_to_json_and_from_json_round_trip():
    state = WorldState(key="D minor", tempo=100, current_section="bridge")
    state.update_section("chorus")
    state.update_emotion("release")
    payload = state.to_json()
    restored = WorldState.from_json(payload)
    assert restored.key == "D minor"
    assert restored.tempo == 100
    assert restored.current_section == "chorus"
    assert restored.current_emotion == "release"
    assert restored.history == state.history


def test_from_json_with_invalid_payload_raises():
    with pytest.raises(ValueError):
        WorldState.from_json("not-json")


def test_save_load_round_trip(tmp_path: Path):
    state = WorldState(key="A minor", tempo=140)
    state.update_section("intro")
    file_path = tmp_path / "ws.json"
    state.save(file_path)
    assert file_path.exists()
    restored = WorldState.load(file_path)
    assert restored.key == "A minor"
    assert restored.tempo == 140
    assert restored.current_section == "intro"


# ---------------------------------------------------------------------------
# clear / reset_history
# ---------------------------------------------------------------------------


def test_reset_history_keeps_current_state_but_drops_log():
    state = WorldState(current_section="verse")
    state.update_section("chorus")
    state.update_emotion("longing")
    state.reset_history()
    assert state.current_section == "chorus"
    assert state.current_emotion == "longing"
    assert state.history == []
