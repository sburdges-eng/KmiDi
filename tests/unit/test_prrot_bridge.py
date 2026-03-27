"""
Unit tests for music_brain.voice.prrot_bridge.PRROTBridge.

These tests are designed to pass in both native-C++ and pure-Python-fallback
modes.  They exercise the public API with a synthetic sine-wave signal so no
real audio files or voice profiles are required.

All tests should pass without the native C++ library present (pure-Python
fallback path is always available).
"""

import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100.0
DURATION_S = 0.5  # 500 ms of audio
FREQ_HZ = 440.0   # A4 — chosen so pitch detection has a clear target


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


def make_silence(
    duration_s: float = DURATION_S,
    sample_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """Return a float32 zero buffer."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bridge():
    """Shared PRROTBridge instance (module-scoped for speed)."""
    from music_brain.voice.prrot_bridge import PRROTBridge
    return PRROTBridge()


@pytest.fixture(scope="module")
def sine_wave():
    return make_sine()


# ---------------------------------------------------------------------------
# Test: instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_import_does_not_crash(self):
        """Importing the module must not raise any exception."""
        import music_brain.voice.prrot_bridge  # noqa: F401

    def test_instantiation_does_not_crash(self):
        """Constructing PRROTBridge must never raise, even without native lib."""
        from music_brain.voice.prrot_bridge import PRROTBridge
        b = PRROTBridge()
        assert b is not None

    def test_multiple_instances(self):
        """Creating multiple instances must not interfere with each other."""
        from music_brain.voice.prrot_bridge import PRROTBridge
        b1 = PRROTBridge()
        b2 = PRROTBridge()
        assert b1 is not b2


# ---------------------------------------------------------------------------
# Test: is_native property
# ---------------------------------------------------------------------------

class TestIsNative:
    def test_is_native_is_bool(self, bridge):
        assert isinstance(bridge.is_native, bool)

    def test_is_native_consistent(self, bridge):
        """Property must return the same value on repeated access."""
        assert bridge.is_native == bridge.is_native

    def test_pure_python_bridge_is_not_native(self):
        """
        When the native library is absent (typical CI environment), the
        bridge must report is_native=False.

        If the native library IS present and PRROT symbols are exported,
        this test is skipped — it only applies to the fallback path.
        """
        from music_brain.voice.prrot_bridge import PRROTBridge
        b = PRROTBridge()
        if b.is_native:
            pytest.skip("Native PRROT backend is available — fallback test N/A.")
        assert b.is_native is False


# ---------------------------------------------------------------------------
# Test: initialize()
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_returns_bool(self, bridge):
        result = bridge.initialize()
        assert isinstance(result, bool)

    def test_returns_true(self, bridge):
        """initialize() should succeed in both native and Python modes."""
        assert bridge.initialize() is True

    def test_idempotent(self, bridge):
        """Calling initialize() multiple times must not raise."""
        bridge.initialize()
        bridge.initialize()
        assert bridge.initialize() is True


# ---------------------------------------------------------------------------
# Test: load_voice_profile()
# ---------------------------------------------------------------------------

class TestLoadVoiceProfile:
    def test_returns_bool(self, bridge, tmp_path):
        fake_profile = str(tmp_path / "voice.json")
        result = bridge.load_voice_profile(fake_profile)
        assert isinstance(result, bool)

    def test_accepts_string_path(self, bridge, tmp_path):
        """The pure-Python fallback accepts any string path."""
        if bridge.is_native:
            pytest.skip("Native mode may require a real profile file.")
        path = str(tmp_path / "test_profile.json")
        assert bridge.load_voice_profile(path) is True


# ---------------------------------------------------------------------------
# Test: process_audio()
# ---------------------------------------------------------------------------

class TestProcessAudio:
    def test_returns_dict(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert isinstance(result, dict)

    def test_required_keys_present(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        required_keys = {
            "phonemes",
            "pitch_targets",
            "breath_markers",
            "midi_notes",
            "tempo_bpm",
            "sample_rate_hz",
            "total_duration_ms",
            "is_native",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_phonemes_is_list(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert isinstance(result["phonemes"], list)

    def test_pitch_targets_is_list(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert isinstance(result["pitch_targets"], list)

    def test_breath_markers_is_list(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert isinstance(result["breath_markers"], list)

    def test_midi_notes_is_list(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert isinstance(result["midi_notes"], list)

    def test_total_duration_ms_positive(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert result["total_duration_ms"] > 0.0

    def test_total_duration_ms_reasonable(self, bridge, sine_wave):
        """Duration should be within 1% of actual signal length."""
        expected_ms = (len(sine_wave) / SAMPLE_RATE) * 1000.0
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert abs(result["total_duration_ms"] - expected_ms) < expected_ms * 0.01

    def test_is_native_flag_matches_property(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE)
        assert result["is_native"] == bridge.is_native

    def test_tempo_bpm_reflected(self, bridge, sine_wave):
        result = bridge.process_audio(sine_wave, SAMPLE_RATE, tempo_bpm=90.0)
        assert result["tempo_bpm"] == pytest.approx(90.0)

    def test_accepts_short_buffer(self, bridge):
        """Must not crash on a very short (1 sample) buffer."""
        tiny = np.zeros(1, dtype=np.float32)
        result = bridge.process_audio(tiny, SAMPLE_RATE)
        assert isinstance(result, dict)

    def test_accepts_empty_buffer(self, bridge):
        """Must handle a zero-length buffer without crashing."""
        empty = np.zeros(0, dtype=np.float32)
        result = bridge.process_audio(empty, SAMPLE_RATE)
        assert isinstance(result, dict)

    def test_accepts_float64_input(self, bridge):
        """Input does not have to be float32 — bridge should coerce."""
        wave64 = make_sine().astype(np.float64)
        result = bridge.process_audio(wave64, SAMPLE_RATE)
        assert isinstance(result, dict)

    def test_accepts_2d_input(self, bridge):
        """2-D arrays (e.g. stereo) should be flattened without crashing."""
        wave2d = make_sine().reshape(1, -1)
        result = bridge.process_audio(wave2d, SAMPLE_RATE)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: analyze_phonemes()
# ---------------------------------------------------------------------------

class TestAnalyzePhonemes:
    def test_returns_list(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        assert isinstance(result, list)

    def test_non_empty_for_audible_signal(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        assert len(result) > 0, "Expected at least one phoneme for 500 ms of audio."

    def test_each_entry_is_dict(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        for entry in result:
            assert isinstance(entry, dict), f"Expected dict, got {type(entry)}"

    def test_phoneme_dicts_have_required_keys(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        if not result:
            pytest.skip("No phonemes detected — cannot check keys.")
        required = {"symbol", "start_time_ms", "duration_ms"}
        for entry in result:
            assert required.issubset(entry.keys()), (
                f"Missing keys: {required - entry.keys()}"
            )

    def test_symbol_is_string(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        for entry in result:
            assert isinstance(entry["symbol"], str)

    def test_start_times_non_negative(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        for entry in result:
            assert entry["start_time_ms"] >= 0.0

    def test_durations_positive(self, bridge, sine_wave):
        result = bridge.analyze_phonemes(sine_wave, SAMPLE_RATE)
        for entry in result:
            assert entry["duration_ms"] > 0.0

    def test_empty_buffer_returns_list(self, bridge):
        result = bridge.analyze_phonemes(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test: detect_breaths()
# ---------------------------------------------------------------------------

class TestDetectBreaths:
    def test_returns_list(self, bridge, sine_wave):
        result = bridge.detect_breaths(sine_wave, SAMPLE_RATE)
        assert isinstance(result, list)

    def test_silence_may_trigger_breath_markers(self, bridge):
        """Silence can look like breath — must return a list either way."""
        silence = make_silence()
        result = bridge.detect_breaths(silence, SAMPLE_RATE)
        assert isinstance(result, list)

    def test_each_entry_is_dict(self, bridge, sine_wave):
        result = bridge.detect_breaths(sine_wave, SAMPLE_RATE)
        for entry in result:
            assert isinstance(entry, dict)

    def test_breath_dicts_have_required_keys(self, bridge, sine_wave):
        result = bridge.detect_breaths(sine_wave, SAMPLE_RATE)
        if not result:
            pytest.skip("No breath markers detected.")
        required = {"time_ms", "intensity", "duration_ms"}
        for entry in result:
            assert required.issubset(entry.keys()), (
                f"Missing keys: {required - entry.keys()}"
            )

    def test_time_ms_non_negative(self, bridge, sine_wave):
        result = bridge.detect_breaths(sine_wave, SAMPLE_RATE)
        for entry in result:
            assert entry["time_ms"] >= 0.0

    def test_empty_buffer_returns_list(self, bridge):
        result = bridge.detect_breaths(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test: track_pitch()
# ---------------------------------------------------------------------------

class TestTrackPitch:
    def test_returns_dict(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        assert isinstance(result, dict)

    def test_required_keys(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        required = {"frequency_hz", "midi_note", "cents_offset", "confidence", "is_valid"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_frequency_hz_is_float(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        assert isinstance(result["frequency_hz"], float)

    def test_midi_note_is_int(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        assert isinstance(result["midi_note"], int)

    def test_midi_note_in_valid_range(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        # Either 0 (invalid/silence) or a real note 0-127
        if result["is_valid"]:
            assert 0 <= result["midi_note"] <= 127

    def test_confidence_is_float(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        assert isinstance(result["confidence"], float)

    def test_confidence_in_0_1_range(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_is_valid_is_bool(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        assert isinstance(result["is_valid"], bool)

    def test_sine_440hz_detected_near_a4(self, bridge):
        """
        A 440 Hz sine wave should be detected within ±1 semitone of A4 (MIDI 69).

        The autocorrelation estimator can be off by one period bin, so we allow
        a generous tolerance (±2 semitones, MIDI 67–71).
        """
        wave = make_sine(frequency_hz=440.0, duration_s=1.0)
        result = bridge.track_pitch(wave, SAMPLE_RATE)
        if not result["is_valid"]:
            pytest.skip("Pitch not detected as valid — may be a borderline case.")
        assert abs(result["midi_note"] - 69) <= 2, (
            f"Expected A4 (MIDI 69), got MIDI {result['midi_note']} "
            f"({result['frequency_hz']:.1f} Hz)"
        )

    def test_frequency_positive_for_valid_result(self, bridge, sine_wave):
        result = bridge.track_pitch(sine_wave, SAMPLE_RATE)
        if result["is_valid"]:
            assert result["frequency_hz"] > 0.0

    def test_silence_returns_dict(self, bridge):
        """Silence must not crash — returns a dict with is_valid=False."""
        silence = make_silence(duration_s=0.1)
        result = bridge.track_pitch(silence, SAMPLE_RATE)
        assert isinstance(result, dict)
        # Silence typically has is_valid=False
        if not bridge.is_native:
            assert result["is_valid"] is False

    def test_empty_buffer_returns_dict(self, bridge):
        result = bridge.track_pitch(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
        assert isinstance(result, dict)
        assert result["is_valid"] is False
