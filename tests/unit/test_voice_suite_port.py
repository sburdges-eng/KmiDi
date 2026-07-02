"""
Unit tests for the voice/vocal suite ported from archive/feature/integration-finalize.

Covers: phoneme processing, pitch control, formant singing synthesis,
instrument synthesis, voice input/learning (including path-traversal
sanitization), the neural backend's degraded mode, music_brain.common
helpers, and the emotion_api facade adapted to main's module layout.

All tests run with core dependencies only (numpy); optional-dependency
paths (librosa/soundfile/torch/g2p_en/sounddevice) are exercised in their
guarded fallback form when those packages are absent.
"""

import base64
import importlib
import json
import os

import numpy as np
import pytest

from music_brain import common as mb_common
from music_brain import emotion_api
from music_brain.voice import (
    ExpressionParams,
    FormantConfig,
    InstrumentSynthesizer,
    LearnedVoiceProfile,
    PhonemeProcessor,
    PitchController,
    SingingSynthesizer,
    SingingVoice,
    SingingVoiceDev,
    VoiceLearner,
    VoiceLearningManager,
    VoiceMimic,
    create_singing_voice,
    create_singing_voice_dev,
)
from music_brain.voice import neural_backend as neural_backend_module
from music_brain.voice.neural_backend import NeuralBackend, check_neural_availability
from music_brain.voice.voice_learning import _sanitize_id

pytestmark = pytest.mark.unit

SR = 8000  # small sample rate keeps synthesis tests fast


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_voice_package_exports_ported_suite():
    import music_brain.voice as voice_pkg

    expected = {
        "PhonemeProcessor",
        "PhonemeSequence",
        "Phoneme",
        "PitchController",
        "PitchCurve",
        "ExpressionParams",
        "SingingSynthesizer",
        "FormantConfig",
        "SingingVoice",
        "create_singing_voice",
        "SingingVoiceDev",
        "create_singing_voice_dev",
        "VoiceRecorder",
        "VoiceMimic",
        "VoiceLearningManager",
        "VoiceLearner",
        "VoiceSampleStore",
        "VoiceSample",
        "LearnedVoiceProfile",
        "InstrumentSynthesizer",
        "InstrumentConfig",
        "get_instrument_preset",
        "NeuralBackend",
        "create_neural_backend",
        "check_neural_availability",
    }
    missing = expected - set(voice_pkg.__all__)
    assert not missing, f"missing from music_brain.voice.__all__: {sorted(missing)}"


# ---------------------------------------------------------------------------
# PhonemeProcessor
# ---------------------------------------------------------------------------


def test_text_to_phonemes_produces_cmu_symbols():
    processor = PhonemeProcessor()
    phonemes = processor.text_to_phonemes("la la")
    assert phonemes, "expected non-empty phoneme list"
    from music_brain.voice.phoneme_processor import CMU_PHONEMES

    assert all(p in CMU_PHONEMES or p in (" ", ".", ",") for p in phonemes)


def test_text_to_phonemes_empty_input_yields_silence():
    processor = PhonemeProcessor()
    assert processor.text_to_phonemes("") == ["SIL"] or processor.g2p is not None


def test_estimate_durations_scales_to_total():
    processor = PhonemeProcessor()
    seq = processor.estimate_durations(["L", "AE", "SP"], total_duration_ms=600.0)
    assert seq.total_duration_ms == pytest.approx(600.0, rel=1e-6)
    assert len(seq.phonemes) == 3
    # Start times are cumulative
    starts = [p.start_time_ms for p in seq.phonemes]
    assert starts == sorted(starts)


def test_align_to_melody_covers_all_notes():
    processor = PhonemeProcessor()
    phonemes = ["L", "AE", "L", "AE"]
    seq = processor.align_to_melody(phonemes, [60, 62, 64], tempo_bpm=120.0)
    # One beat per note at 120 BPM -> 500 ms per note -> 1500 ms total
    assert seq.total_duration_ms == pytest.approx(1500.0, rel=1e-6)
    assert len(seq.phonemes) >= len(phonemes)


def test_process_lyrics_roundtrip():
    from music_brain.voice.phoneme_processor import process_lyrics

    seq = process_lyrics("hi", melody_notes=[60, 64], tempo_bpm=120.0)
    assert seq.total_duration_ms > 0
    assert seq.phonemes


# ---------------------------------------------------------------------------
# PitchController
# ---------------------------------------------------------------------------


def test_midi_frequency_roundtrip():
    pc = PitchController(sample_rate=SR)
    assert pc.midi_to_frequency(69) == pytest.approx(440.0)
    assert pc.frequency_to_midi(440.0) == pytest.approx(69.0)
    for note in (0, 60, 127):
        assert pc.frequency_to_midi(pc.midi_to_frequency(note)) == pytest.approx(note)


def test_create_pitch_curve_length_and_content():
    pc = PitchController(sample_rate=SR)
    curve = pc.create_pitch_curve([60, 64], [0.1, 0.1], ExpressionParams())
    assert len(curve.frequencies) == int(0.2 * SR)
    assert curve.duration_seconds == pytest.approx(0.2)
    # First samples should sit near C4 (~261.6 Hz) modulo vibrato
    assert curve.frequencies[0] == pytest.approx(261.63, rel=0.05)
    assert np.all(np.isfinite(curve.frequencies))


def test_add_pitch_bend_applies_per_segment():
    pc = PitchController(sample_rate=SR)
    curve = pc.create_pitch_curve([69], [0.2], ExpressionParams(vibrato_depth=0.0))
    bent = pc.add_pitch_bend(curve, [(0.0, 12.0), (0.1, 0.0)])
    # First segment shifted up an octave, second unchanged
    assert bent.frequencies[10] == pytest.approx(curve.frequencies[10] * 2.0, rel=1e-6)
    assert bent.frequencies[-10] == pytest.approx(curve.frequencies[-10], rel=1e-6)


def test_audio_to_midi_notes_detects_sine_pitch():
    pc = PitchController(sample_rate=SR)
    t = np.arange(SR) / SR  # 1 second
    audio = np.sin(2 * np.pi * 440.0 * t)
    notes = pc.audio_to_midi_notes(audio, SR, note_duration=1.0)
    assert notes, "expected at least one detected note"
    assert notes[0] == pytest.approx(69, abs=1)


def test_audio_to_midi_notes_empty_audio_defaults_to_c4():
    pc = PitchController(sample_rate=SR)
    assert pc.audio_to_midi_notes(np.zeros(16), SR) == [60]


# ---------------------------------------------------------------------------
# SingingSynthesizer / SingingVoice
# ---------------------------------------------------------------------------


def test_formant_synthesis_produces_normalized_audio():
    processor = PhonemeProcessor()
    seq = processor.align_to_melody(processor.text_to_phonemes("la"), [60], tempo_bpm=240.0)
    pc = PitchController(sample_rate=SR)
    curve = pc.create_pitch_curve([60], [seq.total_duration_ms / 1000.0])
    synth = SingingSynthesizer(FormantConfig(sample_rate=SR))
    audio = synth.synthesize(seq, curve)
    assert audio.shape[0] == int(seq.total_duration_ms / 1000.0 * SR)
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 0.9 + 1e-9


def test_singing_voice_preview_and_backend_fallback():
    voice = SingingVoice(backend="auto", sample_rate=SR)
    # Without neural deps/models the auto backend must fall back to formant
    if voice.neural is None or not voice.neural.is_available():
        assert voice._choose_backend() == "formant"
    audio = voice.preview("la la", [60, 62, 64], tempo_bpm=240.0)
    assert audio.size > 0
    assert np.all(np.isfinite(audio))


def test_singing_voice_sing_matches_preview_when_no_neural():
    voice = create_singing_voice(backend="formant", sample_rate=SR)
    audio = voice.sing("la", [60], tempo_bpm=240.0)
    assert audio.size > 0


def test_singing_voice_save_writes_wav(tmp_path):
    voice = create_singing_voice(backend="formant", sample_rate=SR)
    audio = voice.preview("la", [60], tempo_bpm=240.0)
    out = tmp_path / "preview.wav"
    voice.save(out, audio)
    assert out.exists()
    assert out.read_bytes()[:4] == b"RIFF"


def test_singing_voice_dev_is_behavior_compatible():
    dev = create_singing_voice_dev(backend="formant", sample_rate=SR)
    assert isinstance(dev, SingingVoiceDev)
    assert isinstance(dev, SingingVoice)
    assert dev.prompt.strip()
    audio = dev.preview("la", [60], tempo_bpm=240.0)
    assert audio.size > 0


# ---------------------------------------------------------------------------
# InstrumentSynthesizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instrument",
    ["piano", "guitar", "strings", "flute", "trumpet", "violin", "unknown"],
)
def test_instrument_synthesis_all_presets(instrument):
    synth = InstrumentSynthesizer(instrument, sample_rate=SR)
    audio = synth.synthesize_notes([60, 64], [0.1, 0.1])
    assert audio.shape[0] == int(0.2 * SR)
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 0.9 + 1e-9


def test_instrument_adsr_survives_notes_shorter_than_release():
    # Regression: piano preset has release_time=0.3s; notes shorter than the
    # envelope segments previously overflowed the buffer (ValueError).
    synth = InstrumentSynthesizer("piano", sample_rate=SR)
    for duration in (0.05, 0.01, 0.001):
        audio = synth.synthesize_notes([60], [duration])
        assert np.all(np.isfinite(audio))


def test_formant_synthesis_high_fricatives_and_glides_at_low_sample_rate():
    """Regression: S/Z (6 kHz band) and glide formants exceed Nyquist at low
    sample rates; with scipy present the filters must degrade to passthrough,
    not raise ValueError."""
    from music_brain.voice.phoneme_processor import Phoneme, PhonemeSequence

    synth = SingingSynthesizer(FormantConfig(sample_rate=SR))
    pc = PitchController(sample_rate=SR)
    phonemes = [
        Phoneme(symbol="S", duration_ms=100.0, start_time_ms=0.0),
        Phoneme(symbol="Y", duration_ms=100.0, start_time_ms=100.0),
        Phoneme(symbol="IY", duration_ms=100.0, start_time_ms=200.0),
    ]
    seq = PhonemeSequence(phonemes=phonemes, total_duration_ms=300.0)
    curve = pc.create_pitch_curve([69], [0.3])
    audio = synth.synthesize(seq, curve)
    assert np.all(np.isfinite(audio))


def test_formant_synthesis_survives_ultra_short_phonemes():
    """Regression: buffers shorter than filtfilt's padlen must not raise when
    scipy is installed (CI has scipy; local dev may not)."""
    from music_brain.voice.phoneme_processor import Phoneme, PhonemeSequence

    synth = SingingSynthesizer(FormantConfig(sample_rate=SR))
    pc = PitchController(sample_rate=SR)
    symbols = ["AE", "S", "M", "L"]
    phonemes = [
        Phoneme(symbol=s, duration_ms=1.0, start_time_ms=i * 1.0) for i, s in enumerate(symbols)
    ]
    seq = PhonemeSequence(phonemes=phonemes, total_duration_ms=4.0)
    curve = pc.create_pitch_curve([60], [0.004])
    audio = synth.synthesize(seq, curve)
    assert np.all(np.isfinite(audio))


def test_instrument_brightness_filter_clamped_at_low_sample_rate():
    """Regression: brightness=1.0 puts the high-pass cutoff at Nyquist for
    SR=8000; the filter must be skipped, not raise."""
    from music_brain.voice import InstrumentConfig

    synth = InstrumentSynthesizer("piano", sample_rate=SR)
    synth.config = InstrumentConfig(sample_rate=SR, brightness=1.0)
    audio = synth.synthesize_notes([72], [0.2])
    assert np.all(np.isfinite(audio))


def test_get_instrument_preset_fallback():
    from music_brain.voice import get_instrument_preset

    preset = get_instrument_preset("piano")
    assert preset.harmonics == 8
    default = get_instrument_preset("does-not-exist")
    assert default.sample_rate == 44100


# ---------------------------------------------------------------------------
# Voice input / learning (including security regressions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "../../../etc/passwd",
        "..\\..\\windows",
        "C:evil",
        "a/b\\c:d",
        "nul\x00byte",
    ],
)
def test_sanitize_id_strips_traversal_tokens(raw):
    cleaned = _sanitize_id(raw)
    for token in ("/", "\\", "..", ":", "\x00"):
        assert token not in cleaned


def test_sample_store_confines_hostile_ids(tmp_path):
    mgr = VoiceLearningManager(storage_dir=tmp_path)
    mgr.add_sample(np.zeros(64), sample_id="../../escape")
    stored = list((tmp_path / "samples").iterdir())
    assert stored, "expected sanitized sample artifacts inside the store"
    assert all(p.parent == tmp_path / "samples" for p in stored)
    # Nothing may leak next to (outside) the store
    assert not [p for p in tmp_path.parent.iterdir() if "escape" in p.name]


def test_profile_save_load_roundtrip_confined(tmp_path):
    mgr = VoiceLearningManager(storage_dir=tmp_path, sample_rate=SR)
    profile = LearnedVoiceProfile(
        name="../hostile",
        characteristics={"mean_pitch": 200.0, "mfcc_mean": [0.0] * 13},
        sample_count=1,
        total_duration=1.0,
        created="c",
        updated="u",
    )
    path = mgr.save_profile(profile)
    assert path.parent == tmp_path / "profiles"
    loaded = mgr.load_profile("../hostile")
    assert loaded is not None
    assert loaded.characteristics["mean_pitch"] == pytest.approx(200.0)
    assert "__hostile" in mgr.list_profiles()


def test_voice_mimic_extracts_characteristics_without_librosa():
    mimic = VoiceMimic(sample_rate=SR)
    t = np.arange(SR // 2) / SR
    audio = np.sin(2 * np.pi * 220.0 * t)
    chars = mimic.extract_voice_characteristics(audio)
    for key in ("mean_pitch", "pitch_range", "brightness", "breathiness"):
        assert key in chars
    assert chars["mean_pitch"] > 0


def test_voice_learner_builds_profile_from_samples():
    from music_brain.voice import VoiceSample

    learner = VoiceLearner(sample_rate=SR)
    t = np.arange(SR // 2) / SR
    samples = [
        VoiceSample(audio=np.sin(2 * np.pi * 220.0 * t), sample_rate=SR),
        VoiceSample(audio=np.sin(2 * np.pi * 260.0 * t), sample_rate=SR),
    ]
    profile = learner.learn_from_samples(samples, "test-profile")
    assert profile.sample_count == 2
    assert profile.total_duration == pytest.approx(1.0, rel=1e-6)
    assert "mean_pitch" in profile.characteristics
    updated = learner.update_profile(profile, samples[:1])
    assert updated.sample_count == 3


# ---------------------------------------------------------------------------
# NeuralBackend degraded mode
# ---------------------------------------------------------------------------


def test_neural_backend_reports_unavailable_without_deps_or_models(tmp_path):
    backend = NeuralBackend(model_path=str(tmp_path / "missing.ckpt"))
    if backend.is_available():
        pytest.skip("neural runtime + models present in this environment")
    assert backend.get_backend_type() == "none"
    assert backend.synthesize(None, None) is None
    assert "NEURAL VOICE BACKEND SETUP" in backend.get_setup_instructions()


def test_check_neural_availability_keys():
    availability = check_neural_availability()
    assert set(availability) == {"torch", "onnxruntime", "diffsinger", "model_files"}
    assert all(isinstance(v, bool) for v in availability.values())


def test_neural_backend_honors_kelly_model_root(monkeypatch, tmp_path):
    monkeypatch.setenv("KELLY_MODEL_ROOT", str(tmp_path))
    try:
        importlib.reload(neural_backend_module)
        assert neural_backend_module.DEFAULT_MODEL_DIR == tmp_path / "voice"
    finally:
        monkeypatch.delenv("KELLY_MODEL_ROOT", raising=False)
        importlib.reload(neural_backend_module)
    assert "KELLY_MODEL_ROOT" not in os.environ


# ---------------------------------------------------------------------------
# music_brain.common
# ---------------------------------------------------------------------------


def test_common_ppq_constant():
    assert mb_common.PPQ == 480


def test_midi_file_context_with_existing_path(tmp_path):
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06")
    with mb_common.midi_file_context(midi_path=str(midi)) as path:
        assert path == str(midi)


def test_midi_file_context_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        with mb_common.midi_file_context(midi_path=str(tmp_path / "nope.mid")):
            pass


def test_midi_file_context_requires_an_input():
    with pytest.raises(ValueError):
        with mb_common.midi_file_context():
            pass


def test_midi_file_context_base64_creates_and_cleans_temp_file():
    payload = base64.b64encode(b"MThd\x00\x00\x00\x06").decode("ascii")
    with mb_common.midi_file_context(midi_base64=payload) as path:
        assert os.path.exists(path)
        with open(path, "rb") as handle:
            assert handle.read(4) == b"MThd"
    assert not os.path.exists(path)


def test_make_midi_payload_roundtrip(tmp_path):
    midi = tmp_path / "clip.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06")
    payload = mb_common.make_midi_payload(str(midi))
    assert payload["filename"] == "clip.mid"
    assert base64.b64decode(payload["midi_base64"]).startswith(b"MThd")


def test_parse_intent_json_builds_intent():
    intent = mb_common.parse_intent_json(json.dumps({"title": "T", "mood_primary": "grief"}))
    assert intent.title == "T"


# ---------------------------------------------------------------------------
# emotion_api facade
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def brain():
    return emotion_api.MusicBrain(use_neural=False)


def test_emotion_api_core_dependencies_resolve():
    # The port adapted these imports to main's layout; they must resolve.
    assert emotion_api.EmotionalState is not None
    assert emotion_api.get_parameters_for_state is not None
    assert emotion_api.TimingFeel is not None


def test_capabilities_reports_dict(brain):
    caps = brain.get_capabilities()
    assert isinstance(caps, dict)
    assert caps["neural_emotion"] is False
    for key in ("arrangement", "dynamics_engine", "drum_humanizer", "suggestions"):
        assert key in caps


def test_generate_from_text_keyword_path(brain):
    music = brain.generate_from_text("processing grief and loss")
    assert music.emotional_state.primary_emotion == "grief"
    assert -1.0 <= music.emotional_state.valence <= 1.0
    assert 0.0 <= music.emotional_state.arousal <= 1.0
    assert music.musical_params.tempo_suggested > 0
    payload = music.to_dict()
    assert payload["emotional_state"]["primary_emotion"] == "grief"
    assert "density" in payload["musical_params"]
    assert music.summary()


def test_generate_from_intent_applies_tempo_constraints(brain):
    intent = brain.create_intent(
        title="Test Song",
        core_event="Processing loss",
        mood_primary="grief",
        technical_key="F",
        technical_mode="major",
        tempo_range=(78, 86),
    )
    music = brain.generate_from_intent(intent)
    assert music.emotional_state.primary_emotion == "grief"
    assert music.musical_params.tempo_min == 78
    assert music.musical_params.tempo_max == 86
    assert music.musical_params.tempo_suggested == 82


def test_fluent_chain_maps_text_to_music(brain):
    result = brain.process("anxiety and tension").map_to_emotion().map_to_music().get()
    assert result["emotional_state"]["primary_emotion"] in ("anxiety", "tension")
    assert result["musical_params"]["tempo"] > 0


def test_fluent_chain_overrides(brain):
    chain = brain.process("hope").map_to_music().with_tempo(111).with_dissonance(2.0)
    assert chain.musical_params.tempo_suggested == 111
    assert chain.musical_params.dissonance == 1.0  # clamped


def test_generate_arrangement_uses_main_generator(brain):
    music = brain.generate_from_text("hopeful energy")
    arrangement = brain.generate_arrangement(music)
    assert isinstance(arrangement, dict)
    if emotion_api.HAS_ARRANGEMENT:
        assert "error" not in arrangement
    else:
        assert arrangement == {"error": "Arrangement generator not available"}


def test_get_dynamics_profile(brain):
    profile = brain.get_dynamics_profile(["intro", "verse", "chorus"])
    if brain.dynamics_engine is None:
        assert "error" in profile
    else:
        assert set(profile["sections"]) == {"intro", "verse", "chorus"}


def test_export_to_logic_degrades_without_mixer_module(brain, tmp_path):
    music = brain.generate_from_text("calm")
    if emotion_api.export_to_logic_automation is None:
        with pytest.raises(ImportError):
            brain.export_to_logic(music, str(tmp_path / "out"))
    else:
        paths = brain.export_to_logic(music, str(tmp_path / "out"))
        assert "automation" in paths


def test_suggest_rules_returns_list(brain):
    suggestions = brain.suggest_rules("grief")
    assert isinstance(suggestions, list)
