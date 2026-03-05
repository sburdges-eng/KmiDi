from music_brain.session.intent_schema import CompleteSongIntent


def test_mapping():
    # Mock payload from UI
    payload = {
        "core_desire": "find inner peace",
        "emotional_intent": "Serene",
        "vulnerability_scale": 0.8,
        "secondary_tension": 0.2,
        "imagery_texture": "Soft clouds",
        "technical": {
            "key": "G major",
            "bpm": 72,
            "genre": "Ambient",
            "groove_feel": "Ethereal",
            "rule_to_break": "HARMONY_ParallelMotion",
            "rule_justification": "Break tradition for peace",
            "structure": [{"name": "intro", "bars": 8, "repetitions": 1}],
            "instruments": [{"instrument": "pad", "techniques": ["swell"]}],
        },
    }

    # Simulate what happens in /generate endpoint
    # 1. Map to strict payload for CompleteSongIntentRequest validation
    tech = payload["technical"]
    strict_payload = {
        "core_desire": payload["core_desire"],
        "mood_primary": payload["emotional_intent"],
        "genre": tech["genre"],
        "tempo": tech["bpm"],
        "key_mode": tech["key"],
        "structure": tech["structure"],
        "instruments": tech["instruments"],
        "allow_legacy_fallback": False,
        "groove_feel": tech.get("groove_feel", "Straight/Driving"),
        "narrative_arc": tech.get("narrative_arc", "Climb-to-Climax"),
        "rule_to_break": tech.get("rule_to_break"),
        "rule_justification": tech.get("rule_justification"),
    }

    # 2. Map to domain CompleteSongIntent
    # In api.py we merge strict_intent.dict() with request.intent fields
    ui_data = strict_payload.copy()
    ui_data.update(
        {
            "core_wound": payload.get("core_wound"),
            "vulnerability_scale": payload.get("vulnerability_scale"),
            "secondary_tension": payload.get("secondary_tension"),
            "imagery_texture": payload.get("imagery_texture"),
        }
    )

    intent = CompleteSongIntent.from_ui_payload(ui_data)

    assert intent.song_intent.mood_primary == "Serene", (
        "mood_primary should map from emotional_intent"
    )
    assert intent.song_intent.vulnerability_scale == "High", (
        "vulnerability_scale 0.8 should map to 'High'"
    )
    assert intent.song_intent.mood_secondary_tension == 0.2, (
        "secondary_tension should pass through unchanged"
    )
    assert intent.song_intent.imagery_texture == "Soft clouds", (
        "imagery_texture should pass through unchanged"
    )
    assert intent.technical_constraints.technical_genre == "Ambient", (
        "genre should map to technical_genre"
    )
    assert intent.technical_constraints.technical_key == "G", (
        "key should be parsed to root note"
    )
    assert intent.technical_constraints.technical_mode == "major", (
        "mode should be parsed from key string"
    )
    assert intent.technical_constraints.technical_groove_feel == "Ethereal", (
        "groove_feel should map to technical_groove_feel"
    )
    assert intent.technical_constraints.technical_rule_to_break == "HARMONY_ParallelMotion", (
        "rule_to_break should map to technical_rule_to_break"
    )
    assert intent.technical_constraints.rule_breaking_justification == "Break tradition for peace", (
        "rule_justification should map to rule_breaking_justification"
    )
