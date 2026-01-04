from music_brain.kelly import Kelly
from music_brain.dee import Dee


def test_kelly_listen():
    kelly = Kelly()
    user_feeling = "I'm feeling really overwhelmed and sad about everything happening."
    response = kelly.listen(user_feeling)

    # Check validation message contains relevant emotion words
    # Note: "overwhelmed" maps to FEAR/ANXIETY in our current logic, so we check for that or the input word
    msg_lower = response.validation_message.lower()
    assert any(w in msg_lower for w in [
               "overwhelmed", "sad", "fear", "anxiety", "grief"])

    assert response.musical_inspiration is not None
    assert response.guidance is not None
    assert response.musical_inspiration.intensity_tier is not None


def test_dee_produce():
    dee = Dee()
    tech_request = "I've had a beat 5/8 in my head that leads an ensemble of violas into an escalating intro stecatto ascending 2 steps in E minor before the synth bass booms and slows into an delay/phaser to lead into verse 1"
    spec = dee.produce(tech_request)

    assert spec.time_signature == "5/8"
    assert "viola" in spec.instruments
    assert "synth bass" in spec.instruments
    assert "delay" in spec.effects
    assert "phaser" in spec.effects

    consultation = dee.consult(spec)
    assert "5/8" in consultation
    assert "viola" in consultation
