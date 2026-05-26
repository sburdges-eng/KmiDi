"""DualClip — audio buffer + aligned MIDI event list as one object.

A small container for code that needs to keep audio and MIDI in sync —
e.g. inference outputs (audio) joined with the symbolic events that
generated them, or training pairs where each audio chunk has its
corresponding MIDI annotation.

Operations preserve alignment: :meth:`slice` extracts both a sample
range and the events that fall inside it (rebased to start at 0);
:meth:`concat` joins two clips and offsets the second clip's event
times by the first clip's duration.

Goal-list ties: Audio/MIDI dual representation (direct),
Hybrid symbolic/audio generation (the host-side container for hybrid
outputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass(frozen=True)
class MidiEvent:
    """Minimal MIDI note event keyed to wall time in seconds."""

    time_sec: float
    pitch: int
    velocity: int
    duration_sec: float

    def __post_init__(self) -> None:
        if self.time_sec < 0:
            raise ValueError(f"time_sec must be >= 0; got {self.time_sec}")
        if self.duration_sec < 0:
            raise ValueError(f"duration_sec must be >= 0; got {self.duration_sec}")
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch must be in [0, 127]; got {self.pitch}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity must be in [0, 127]; got {self.velocity}")


class DualClip:
    """Aligned audio buffer + MIDI event list.

    Events whose ``time_sec`` lies outside ``[0, duration_sec)`` are
    rejected at construction.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        events: List[MidiEvent],
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self.audio: np.ndarray = np.asarray(audio, dtype=np.float32)
        self.sample_rate: int = int(sample_rate)
        duration = self.duration_sec
        for ev in events:
            if not 0 <= ev.time_sec < max(duration, 1e-9):
                raise ValueError(
                    f"event time_sec={ev.time_sec} outside clip duration {duration}"
                )
        self.events: List[MidiEvent] = list(events)

    # ----------------------------------------------------------------- meta
    @property
    def duration_sec(self) -> float:
        return self.audio.shape[0] / self.sample_rate

    def event_to_sample_index(self, event: MidiEvent) -> int:
        return int(round(event.time_sec * self.sample_rate))

    def sample_index_to_sec(self, sample_index: int) -> float:
        return sample_index / self.sample_rate

    # ----------------------------------------------------------------- ops
    def slice(self, start_sec: float, end_sec: float) -> "DualClip":
        """Return a new DualClip covering ``[start_sec, end_sec)``.

        Bounds are clamped to the clip's own range. Events are filtered
        and rebased so the returned clip's event times start at 0.
        """
        duration = self.duration_sec
        start_sec = max(0.0, min(duration, start_sec))
        end_sec = max(start_sec, min(duration, end_sec))
        start_idx = int(round(start_sec * self.sample_rate))
        end_idx = int(round(end_sec * self.sample_rate))
        audio = self.audio[start_idx:end_idx].copy()
        rebased: List[MidiEvent] = []
        for ev in self.events:
            if start_sec <= ev.time_sec < end_sec:
                rebased.append(
                    MidiEvent(
                        time_sec=ev.time_sec - start_sec,
                        pitch=ev.pitch,
                        velocity=ev.velocity,
                        duration_sec=ev.duration_sec,
                    )
                )
        return DualClip(audio=audio, sample_rate=self.sample_rate, events=rebased)

    def concat(self, other: "DualClip") -> "DualClip":
        """Concatenate two clips. ``other``'s events are offset by self's duration."""
        if self.sample_rate != other.sample_rate:
            raise ValueError(
                f"sample_rate mismatch: {self.sample_rate} vs {other.sample_rate}"
            )
        offset = self.duration_sec
        offset_events = [
            MidiEvent(
                time_sec=ev.time_sec + offset,
                pitch=ev.pitch,
                velocity=ev.velocity,
                duration_sec=ev.duration_sec,
            )
            for ev in other.events
        ]
        return DualClip(
            audio=np.concatenate([self.audio, other.audio]),
            sample_rate=self.sample_rate,
            events=self.events + offset_events,
        )

    def iter_events_in_range(
        self, start_sec: float, end_sec: float
    ) -> Iterator[MidiEvent]:
        """Yield events whose ``time_sec`` lies in ``[start_sec, end_sec)``."""
        for ev in self.events:
            if start_sec <= ev.time_sec < end_sec:
                yield ev
