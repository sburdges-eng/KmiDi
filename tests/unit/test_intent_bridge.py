"""Regression coverage for the canonical intent bridge path."""

import json

from music_brain.session.intent_bridge import process_intent


def test_process_intent_bridge_returns_cpp_result_shape():
    payload = {
        "phase_0": {
            "core_event": "loss",
            "core_resistance": "distance",
            "core_longing": "peace",
        },
        "phase_1": {
            "mood_primary": "grief",
            "vulnerability_scale": 0.8,
            "imagery_texture": "foggy",
        },
        "phase_2": {
            "technical_genre": "ambient",
            "technical_key": "C",
            "technical_mode": "minor",
            "technical_rule_to_break": [],
        },
    }

    result = json.loads(process_intent(json.dumps(payload)))

    assert {"key", "mode", "tempoBpm"} <= set(result)
