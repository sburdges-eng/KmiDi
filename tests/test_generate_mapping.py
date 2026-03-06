from music_brain.session.intent_schema import CompleteSongIntent

def test_ui_to_engine_parameter_mapping():
    """Verify that the UI payload is strictly and correctly mapped to the internal CompleteSongIntent."""

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
            "instruments": [{"instrument": "pad", "techniques": ["swell"]}]
        }
    }

    # Simulate what happens in /generate endpoint
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

    ui_data = strict_payload.copy()
    ui_data.update({
        "core_wound": payload.get("core_wound"),
        "vulnerability_scale": payload.get("vulnerability_scale"),
        "secondary_tension": payload.get("secondary_tension"),
        "imagery_texture": payload.get("imagery_texture"),
    })

    intent = CompleteSongIntent.from_ui_payload(ui_data)

    # Assertions with descriptive messages
    assert intent.song_intent.mood_primary == "Serene", f"Expected mood_primary 'Serene', got '{intent.song_intent.mood_primary}'"
    assert intent.song_intent.vulnerability_scale == "High", f"Expected vulnerability_scale 'High' for 0.8, got '{intent.song_intent.vulnerability_scale}'"
    assert intent.song_intent.mood_secondary_tension == 0.2, f"Expected tension 0.2, got {intent.song_intent.mood_secondary_tension}"
    assert intent.song_intent.imagery_texture == "Soft clouds", f"Expected imagery 'Soft clouds', got '{intent.song_intent.imagery_texture}'"
    assert intent.technical_constraints.technical_genre == "Ambient", f"Expected genre 'Ambient', got '{intent.technical_constraints.technical_genre}'"
    assert intent.technical_constraints.technical_key == "G", f"Expected key 'G', got '{intent.technical_constraints.technical_key}'"
    assert intent.technical_constraints.technical_mode == "major", f"Expected mode 'major', got '{intent.technical_constraints.technical_mode}'"
    assert intent.technical_constraints.technical_groove_feel == "Ethereal", f"Expected groove 'Ethereal', got '{intent.technical_constraints.technical_groove_feel}'"
    assert intent.technical_constraints.technical_rule_to_break == "HARMONY_ParallelMotion", f"Expected rule 'HARMONY_ParallelMotion', got '{intent.technical_constraints.technical_rule_to_break}'"
    assert intent.technical_constraints.rule_breaking_justification == "Break tradition for peace", f"Expected justification 'Break tradition for peace', got '{intent.technical_constraints.rule_breaking_justification}'"
