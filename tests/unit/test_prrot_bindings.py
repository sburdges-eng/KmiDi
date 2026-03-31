"""
Unit tests for penta_core_native.prrot pybind11 bindings.

Tests the C++ PRROT engine directly through the pybind11 interface,
NOT through the Python PRROTBridge wrapper. Uses synthetic sine waves
so no audio files or voice profiles are needed.

Requires: penta_core_native built (cmake --build build --target penta_core_native)
Skip gracefully if native library not available.
"""

import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Skip if native module unavailable
# ---------------------------------------------------------------------------

try:
    import penta_core_native
    prrot = penta_core_native.prrot
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE,
    reason="penta_core_native not built (run cmake --build build --target penta_core_native)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100.0
DURATION_S = 0.5
FREQ_HZ = 440.0  # A4


def make_sine(
    frequency_hz: float = FREQ_HZ,
    duration_s: float = DURATION_S,
    sample_rate: float = SAMPLE_RATE,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Return a float32 sine wave as a 1-D numpy array."""
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    wave = amplitude * np.sin(2.0 * math.pi * frequency_hz * t)
    return wave.astype(np.float32)


def make_silence(duration_s: float = DURATION_S) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Shared PRROTEngine instance."""
    e = prrot.PRROTEngine()
    e.initialize()
    return e


@pytest.fixture(scope="module")
def sine_wave():
    return make_sine()


@pytest.fixture(scope="module")
def silence():
    return make_silence()


# ---------------------------------------------------------------------------
# Test: PhonemeType enum
# ---------------------------------------------------------------------------

class TestPhonemeType:
    def test_vowels_exist(self):
        assert prrot.PhonemeType.AA is not None
        assert prrot.PhonemeType.EH is not None
        assert prrot.PhonemeType.IY is not None
        assert prrot.PhonemeType.UW is not None

    def test_consonants_exist(self):
        assert prrot.PhonemeType.B is not None
        assert prrot.PhonemeType.S is not None
        assert prrot.PhonemeType.M is not None
        assert prrot.PhonemeType.NG is not None

    def test_special_types(self):
        assert prrot.PhonemeType.SIL is not None
        assert prrot.PhonemeType.UNKNOWN is not None

    def test_phoneme_to_string(self):
        s = prrot.phoneme_to_string(prrot.PhonemeType.AA)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_string_to_phoneme(self):
        p = prrot.string_to_phoneme("AA")
        assert p == prrot.PhonemeType.AA

    def test_is_vowel(self):
        assert prrot.is_vowel(prrot.PhonemeType.AA) is True
        assert prrot.is_vowel(prrot.PhonemeType.EH) is True
        assert prrot.is_vowel(prrot.PhonemeType.B) is False

    def test_is_consonant(self):
        assert prrot.is_consonant(prrot.PhonemeType.B) is True
        assert prrot.is_consonant(prrot.PhonemeType.S) is True
        assert prrot.is_consonant(prrot.PhonemeType.AA) is False


# ---------------------------------------------------------------------------
# Test: Data types
# ---------------------------------------------------------------------------

class TestDataTypes:
    def test_phoneme_timing_construction(self):
        pt = prrot.PhonemeTiming()
        assert hasattr(pt, "phoneme")
        assert hasattr(pt, "start_time_ms")
        assert hasattr(pt, "duration_ms")
        assert hasattr(pt, "is_stressed")
        assert hasattr(pt, "stress_level")
        assert hasattr(pt, "prominence")

    def test_phoneme_timing_repr(self):
        pt = prrot.PhonemeTiming()
        r = repr(pt)
        assert "PhonemeTiming" in r

    def test_pitch_target_construction(self):
        pt = prrot.PitchTarget()
        assert hasattr(pt, "time_ms")
        assert hasattr(pt, "midi_note")
        assert hasattr(pt, "cents_offset")
        assert hasattr(pt, "confidence")

    def test_breath_marker_construction(self):
        bm = prrot.BreathMarker()
        assert hasattr(bm, "time_ms")
        assert hasattr(bm, "duration_ms")
        assert hasattr(bm, "intensity")

    def test_voice_profile_construction(self):
        vp = prrot.VoiceProfile()
        assert hasattr(vp, "profile_id")
        assert hasattr(vp, "profile_name")
        assert callable(getattr(vp, "is_valid", None))

    def test_phoneme_control_data_construction(self):
        cd = prrot.PhonemeControlData()
        assert hasattr(cd, "phoneme_sequence")
        assert hasattr(cd, "pitch_targets")

    def test_phoneme_control_data_repr(self):
        cd = prrot.PhonemeControlData()
        r = repr(cd)
        assert "PhonemeControlData" in r


# ---------------------------------------------------------------------------
# Test: PRROTEngine instantiation
# ---------------------------------------------------------------------------

class TestEngineInstantiation:
    def test_construction(self):
        e = prrot.PRROTEngine()
        assert e is not None

    def test_initialize(self):
        e = prrot.PRROTEngine()
        e.initialize()  # should not throw

    def test_multiple_instances(self):
        e1 = prrot.PRROTEngine()
        e2 = prrot.PRROTEngine()
        e1.initialize()
        e2.initialize()
        # Both should be independent


# ---------------------------------------------------------------------------
# Test: process_audio_segment
# ---------------------------------------------------------------------------

class TestProcessAudio:

    def test_sine_wave(self, engine, sine_wave):
        result = engine.process_audio_segment(sine_wave, SAMPLE_RATE, 120.0)
        assert isinstance(result, prrot.PhonemeControlData)


    def test_silence(self, engine, silence):
        result = engine.process_audio_segment(silence, SAMPLE_RATE, 120.0)
        assert isinstance(result, prrot.PhonemeControlData)


    def test_default_args(self, engine, sine_wave):
        result = engine.process_audio_segment(sine_wave)
        assert isinstance(result, prrot.PhonemeControlData)


    def test_short_audio(self, engine):
        short = np.zeros(100, dtype=np.float32)
        result = engine.process_audio_segment(short, SAMPLE_RATE)
        assert isinstance(result, prrot.PhonemeControlData)

    def test_rejects_2d_array(self, engine):
        audio_2d = np.zeros((2, 1000), dtype=np.float32)
        with pytest.raises(RuntimeError, match="1D"):
            engine.process_audio_segment(audio_2d)


# ---------------------------------------------------------------------------
# Test: analyze_phonemes
# ---------------------------------------------------------------------------

class TestAnalyzePhonemes:
    def test_sine_wave(self, engine, sine_wave):
        phonemes = engine.analyze_phonemes(sine_wave, SAMPLE_RATE)
        assert isinstance(phonemes, list)

    def test_silence(self, engine, silence):
        phonemes = engine.analyze_phonemes(silence, SAMPLE_RATE)
        assert isinstance(phonemes, list)

    def test_phoneme_timing_fields(self, engine, sine_wave):
        phonemes = engine.analyze_phonemes(sine_wave, SAMPLE_RATE)
        if len(phonemes) > 0:
            pt = phonemes[0]
            assert isinstance(pt, prrot.PhonemeTiming)
            assert pt.start_time_ms >= 0.0
            assert pt.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Test: detect_breath_markers
# ---------------------------------------------------------------------------

class TestDetectBreathMarkers:

    def test_sine_wave(self, engine, sine_wave):
        markers = engine.detect_breath_markers(sine_wave, SAMPLE_RATE)
        assert isinstance(markers, list)


    def test_silence(self, engine, silence):
        markers = engine.detect_breath_markers(silence, SAMPLE_RATE)
        assert isinstance(markers, list)


    def test_marker_fields(self, engine, sine_wave):
        markers = engine.detect_breath_markers(sine_wave, SAMPLE_RATE)
        if len(markers) > 0:
            bm = markers[0]
            assert isinstance(bm, prrot.BreathMarker)
            assert bm.time_ms >= 0.0
            assert bm.intensity >= 0.0


# ---------------------------------------------------------------------------
# Test: VoiceProfile
# ---------------------------------------------------------------------------

class TestVoiceProfile:
    def test_default_profile(self, engine):
        profile = engine.get_voice_profile()
        assert isinstance(profile, prrot.VoiceProfile)

    def test_load_default(self, engine):
        vp = prrot.VoiceProfile()
        engine.load_voice_profile(vp)  # should not throw
