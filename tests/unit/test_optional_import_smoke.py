"""Smoke tests for optional import handling around voice and API surfaces."""

from importlib import import_module, util


def test_voice_import_handles_optional_neural_backend():
    voice = import_module("music_brain.voice")

    assert hasattr(voice, "MODULATION_PRESETS")
    assert hasattr(voice, "AUTO_TUNE_PRESETS")
    assert hasattr(voice, "VOICE_PROFILES")

    backend_spec = util.find_spec("music_brain.voice.neural_backend")
    if backend_spec is None:
        assert not hasattr(voice, "NeuralBackend")
    else:
        assert hasattr(voice, "NeuralBackend")


def test_api_import_smoke():
    api = import_module("music_brain.api")

    assert hasattr(api, "DAiWAPI")
