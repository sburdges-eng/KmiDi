import pytest
from pydantic import ValidationError

from music_brain.engine_api.schema import CompleteSongIntentRequest


def _validate(payload):
    if hasattr(CompleteSongIntentRequest, "model_validate"):
        return CompleteSongIntentRequest.model_validate(payload)
    return CompleteSongIntentRequest.parse_obj(payload)


def _valid_payload():
    return {
        "core_desire": "Resolve internal tension with hopeful momentum",
        "mood_primary": "nostalgic",
        "genre": "synthwave",
        "tempo": 120,
        "key_mode": "C major",
        "structure": [
            {"name": "intro", "bars": 4, "repetitions": 1},
            {"name": "verse", "bars": 8, "repetitions": 2},
        ],
        "instruments": [
            {"instrument": "piano", "techniques": ["staccato"]},
            {"instrument": "synth_bass", "techniques": []},
        ],
        "allow_legacy_fallback": False,
    }


def test_schema_accepts_valid_payload():
    model = _validate(_valid_payload())
    assert model.tempo == 120
    assert model.key_mode == "C major"
    assert len(model.structure) == 2


@pytest.mark.parametrize("tempo", [39, 301])
def test_schema_rejects_out_of_bounds_bpm(tempo):
    payload = _valid_payload()
    payload["tempo"] = tempo
    with pytest.raises(ValidationError):
        _validate(payload)


def test_schema_rejects_invalid_key_mode():
    payload = _valid_payload()
    payload["key_mode"] = "H major"
    with pytest.raises(ValidationError):
        _validate(payload)


def test_schema_rejects_missing_required_field():
    payload = _valid_payload()
    payload.pop("core_desire")
    with pytest.raises(ValidationError):
        _validate(payload)


def test_schema_rejects_oversized_total_structure():
    payload = _valid_payload()
    payload["structure"] = [
        {"name": "verse", "bars": 128, "repetitions": 8},
    ]
    with pytest.raises(ValidationError):
        _validate(payload)


def test_schema_accepts_ui_constraint_fields():
    """UI-sourced groove_feel, narrative_arc, rule_to_break, rule_justification are accepted."""
    payload = _valid_payload()
    payload["groove_feel"] = "Laid-back"
    payload["narrative_arc"] = "Plateau"
    payload["rule_to_break"] = "parallel fifths"
    payload["rule_justification"] = "Intentional tension"
    model = _validate(payload)
    assert model.groove_feel == "Laid-back"
    assert model.narrative_arc == "Plateau"
    assert model.rule_to_break == "parallel fifths"
    assert model.rule_justification == "Intentional tension"


def test_schema_defaults_optional_ui_fields():
    """Missing optional UI fields get defaults."""
    payload = _valid_payload()
    model = _validate(payload)
    assert model.groove_feel == "Straight/Driving"
    assert model.narrative_arc == "Climb-to-Climax"
    assert model.rule_to_break is None
    assert model.rule_justification is None
