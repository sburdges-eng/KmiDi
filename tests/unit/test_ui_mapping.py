import pytest
from music_brain.api import TechnicalIntent, EmotionalIntent, GenerateRequest, DAiWAPI
from music_brain.session.intent_schema import CompleteSongIntent

def test_ui_mapping_completeness():
    # 1. Setup mock UI request with all fields
    tech = TechnicalIntent(
        key="F major",
        bpm=140,
        genre="cinematic",
        groove_feel="Laid Back",
        narrative_arc="Sudden Shift",
        rule_to_break="HARMONY_AvoidTonicResolution",
        rule_justification="To create a sense of longing",
        structure=[{"name": "verse", "bars": 8}],
        instruments=[{"instrument": "piano", "techniques": ["legato"]}]
    )

    emo = EmotionalIntent(
        core_wound="loss of a friend",
        core_desire="reconnection",
        emotional_intent="melancholy",
        imagery_texture="foggy and cold",
        vulnerability_scale=0.8,
        technical=tech
    )

    req = GenerateRequest(intent=emo)

    # 2. Test DAiWAPI._convert_request_to_complete_intent (used internally)
    api = DAiWAPI()
    intent = api._convert_request_to_complete_intent(req)

    assert isinstance(intent, CompleteSongIntent)
    assert intent.song_root.core_event == "loss of a friend"
    assert intent.song_root.core_longing == "reconnection"
    assert intent.song_intent.mood_primary == "melancholy"
    assert intent.song_intent.imagery_texture == "foggy and cold"
    assert intent.song_intent.vulnerability_scale == "High"  # 0.8 maps to High
    assert intent.song_intent.narrative_arc == "Sudden Shift"

    assert intent.technical_constraints.technical_key == "F"
    assert intent.technical_constraints.technical_mode == "major"
    assert intent.technical_constraints.technical_genre == "cinematic"
    assert intent.technical_constraints.technical_groove_feel == "Laid Back"
    assert intent.technical_constraints.technical_rule_to_break == "HARMONY_AvoidTonicResolution"
    assert intent.technical_constraints.rule_breaking_justification == "To create a sense of longing"

    print("UI mapping verification PASSED")

if __name__ == "__main__":
    test_ui_mapping_completeness()
