"""DualClip — aligned audio + MIDI representation.

Goal: Audio/MIDI dual representation (direct),
Hybrid symbolic/audio generation (partial)."""

import numpy as np
import pytest

from music_brain.audio.dual_clip import DualClip, MidiEvent


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_clip_records_audio_and_events():
    audio = np.arange(44100, dtype=np.float32)
    events = [MidiEvent(time_sec=0.5, pitch=60, velocity=100, duration_sec=0.25)]
    clip = DualClip(audio=audio, sample_rate=44100, events=events)
    assert clip.duration_sec == pytest.approx(1.0)
    assert len(clip.events) == 1


def test_event_out_of_audio_range_raises():
    audio = np.zeros(44100, dtype=np.float32)
    bad_event = MidiEvent(time_sec=5.0, pitch=60, velocity=100, duration_sec=0.1)
    with pytest.raises(ValueError, match="time_sec"):
        DualClip(audio=audio, sample_rate=44100, events=[bad_event])


def test_negative_time_event_raises():
    audio = np.zeros(44100, dtype=np.float32)
    with pytest.raises(ValueError, match="time_sec"):
        DualClip(
            audio=audio,
            sample_rate=44100,
            events=[MidiEvent(time_sec=-0.1, pitch=60, velocity=100, duration_sec=0.1)],
        )


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


def test_slice_returns_aligned_audio_and_filtered_events():
    audio = np.arange(44100, dtype=np.float32)
    events = [
        MidiEvent(time_sec=0.1, pitch=60, velocity=100, duration_sec=0.05),
        MidiEvent(time_sec=0.5, pitch=64, velocity=90, duration_sec=0.05),
        MidiEvent(time_sec=0.9, pitch=67, velocity=80, duration_sec=0.05),
    ]
    clip = DualClip(audio=audio, sample_rate=44100, events=events)
    sliced = clip.slice(start_sec=0.4, end_sec=0.8)
    # Audio sample count: 0.4s @ 44100 = 17640 samples
    assert sliced.audio.shape == (17640,)
    # First sample of slice equals original audio[start_sample]
    assert sliced.audio[0] == pytest.approx(0.4 * 44100, abs=1.0)
    # Only the middle event survives, rebased so time_sec=0 is the slice start.
    assert len(sliced.events) == 1
    assert sliced.events[0].pitch == 64
    assert sliced.events[0].time_sec == pytest.approx(0.1, abs=1e-6)  # 0.5 - 0.4


def test_slice_with_zero_length_returns_empty_clip():
    audio = np.ones(44100, dtype=np.float32)
    clip = DualClip(audio=audio, sample_rate=44100, events=[])
    sliced = clip.slice(start_sec=0.3, end_sec=0.3)
    assert sliced.audio.shape == (0,)
    assert sliced.events == []


def test_slice_clamps_to_clip_bounds():
    audio = np.ones(44100, dtype=np.float32)
    clip = DualClip(audio=audio, sample_rate=44100, events=[])
    sliced = clip.slice(start_sec=-0.5, end_sec=2.0)
    # Clamped to [0, 1]; same length as original.
    assert sliced.audio.shape == (44100,)


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


def test_concat_joins_audio_and_offsets_event_times():
    audio_a = np.ones(22050, dtype=np.float32)  # 0.5 s
    audio_b = np.ones(22050, dtype=np.float32) * 2.0
    events_a = [MidiEvent(time_sec=0.1, pitch=60, velocity=100, duration_sec=0.05)]
    events_b = [MidiEvent(time_sec=0.2, pitch=64, velocity=90, duration_sec=0.05)]
    a = DualClip(audio=audio_a, sample_rate=44100, events=events_a)
    b = DualClip(audio=audio_b, sample_rate=44100, events=events_b)
    joined = a.concat(b)
    # Audio length: 0.5 + 0.5 = 1.0s
    assert joined.duration_sec == pytest.approx(1.0)
    # Events: first one at 0.1, second one at 0.5 + 0.2 = 0.7
    assert len(joined.events) == 2
    assert joined.events[0].time_sec == pytest.approx(0.1)
    assert joined.events[1].time_sec == pytest.approx(0.7)


def test_concat_sample_rate_mismatch_raises():
    a = DualClip(audio=np.zeros(100, dtype=np.float32), sample_rate=44100, events=[])
    b = DualClip(audio=np.zeros(100, dtype=np.float32), sample_rate=48000, events=[])
    with pytest.raises(ValueError, match="sample_rate"):
        a.concat(b)


# ---------------------------------------------------------------------------
# iter_events_in_range
# ---------------------------------------------------------------------------


def test_iter_events_in_range_filters_by_time():
    audio = np.zeros(44100, dtype=np.float32)
    events = [
        MidiEvent(time_sec=0.1, pitch=60, velocity=100, duration_sec=0.05),
        MidiEvent(time_sec=0.5, pitch=64, velocity=90, duration_sec=0.05),
        MidiEvent(time_sec=0.9, pitch=67, velocity=80, duration_sec=0.05),
    ]
    clip = DualClip(audio=audio, sample_rate=44100, events=events)
    inside = list(clip.iter_events_in_range(start_sec=0.4, end_sec=0.8))
    assert [e.pitch for e in inside] == [64]


def test_iter_events_in_range_inclusive_start_exclusive_end():
    audio = np.zeros(44100, dtype=np.float32)
    events = [
        MidiEvent(time_sec=0.0, pitch=60, velocity=100, duration_sec=0.05),
        MidiEvent(time_sec=0.5, pitch=64, velocity=90, duration_sec=0.05),
    ]
    clip = DualClip(audio=audio, sample_rate=44100, events=events)
    # Inclusive at start, exclusive at end (matches numpy / Python slice convention).
    inside = list(clip.iter_events_in_range(start_sec=0.0, end_sec=0.5))
    assert [e.pitch for e in inside] == [60]


# ---------------------------------------------------------------------------
# event_to_sample_index / sample_index_to_sec
# ---------------------------------------------------------------------------


def test_event_to_sample_index_uses_sample_rate():
    audio = np.zeros(44100, dtype=np.float32)
    clip = DualClip(audio=audio, sample_rate=44100, events=[])
    event = MidiEvent(time_sec=0.5, pitch=60, velocity=100, duration_sec=0.05)
    assert clip.event_to_sample_index(event) == 22050


def test_sample_index_to_sec_is_inverse_of_event_to_sample_index():
    audio = np.zeros(44100, dtype=np.float32)
    clip = DualClip(audio=audio, sample_rate=44100, events=[])
    assert clip.sample_index_to_sec(22050) == pytest.approx(0.5)
