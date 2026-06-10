"""
ComprehensiveIntegrationManager — Phase 3 real-time bridge.

High-frequency bridge between the Python "Mind" (music_brain) and the
C++ "Body" (KellyFFI / engine). Polls RTState via ctypes at configurable
frequency and emits events through the existing EventBus.

Usage:
    from music_brain.integrations.comprehensive_bridge import (
        ComprehensiveIntegrationManager,
    )

    manager = ComprehensiveIntegrationManager(
        lib_path="build/libkelly_ffi.dylib",
        event_bus=bus,
        poll_hz=30,
    )
    await manager.start()
    # ... engine runs ...
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ctypes mirror of KellyRTState (must match kelly_ffi.h exactly)
# ---------------------------------------------------------------------------


class CKellyRTState(ctypes.Structure):
    _fields_ = [
        # Timing
        ("bpm", ctypes.c_double),
        ("sample_position", ctypes.c_uint64),
        ("bar_start", ctypes.c_uint64),
        ("bar", ctypes.c_uint32),
        ("beat", ctypes.c_uint32),
        ("numerator", ctypes.c_uint32),
        ("denominator", ctypes.c_uint32),
        ("playing", ctypes.c_int),
        # Emotion
        ("valence", ctypes.c_float),
        ("arousal", ctypes.c_float),
        ("dominance", ctypes.c_float),
        ("discrete_emotion_id", ctypes.c_int16),
        ("emotion_intensity", ctypes.c_float),
        ("emotion_confidence", ctypes.c_float),
        # Musical intent
        ("tempo_bias", ctypes.c_float),
        ("rhythmic_density", ctypes.c_float),
        ("groove_strength", ctypes.c_float),
        ("harmonic_tension", ctypes.c_float),
        ("harmonic_motion", ctypes.c_float),
        ("melodic_activity", ctypes.c_float),
        ("texture_density", ctypes.c_float),
        ("dynamic_range", ctypes.c_float),
        # Track params
        ("track_params", ctypes.c_float * 16),
        # Sequence
        ("sequence", ctypes.c_uint64),
    ]


# RT parameter targets (must match KellyRTTarget enum)
RT_TARGET_BPM = 0
RT_TARGET_EMOTION = 1
RT_TARGET_INTENT = 2
RT_TARGET_TRACK_PARAM = 3
RT_TARGET_TRANSPORT = 4


@dataclass
class RTSnapshot:
    """Python-friendly snapshot of engine state."""

    # Timing
    bpm: float = 120.0
    sample_position: int = 0
    bar: int = 0
    beat: int = 0
    numerator: int = 4
    denominator: int = 4
    playing: bool = False

    # Emotion
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    discrete_emotion_id: int = -1
    emotion_intensity: float = 0.0
    emotion_confidence: float = 0.0

    # Musical intent
    tempo_bias: float = 0.0
    rhythmic_density: float = 0.5
    groove_strength: float = 0.5
    harmonic_tension: float = 0.5
    harmonic_motion: float = 0.5
    melodic_activity: float = 0.5
    texture_density: float = 0.5
    dynamic_range: float = 0.5

    # Track params
    track_params: list[float] = field(default_factory=lambda: [0.0] * 16)

    # Monotonic counter
    sequence: int = 0

    @classmethod
    def from_c(cls, c_state: CKellyRTState) -> RTSnapshot:
        return cls(
            bpm=c_state.bpm,
            sample_position=c_state.sample_position,
            bar=c_state.bar,
            beat=c_state.beat,
            numerator=c_state.numerator,
            denominator=c_state.denominator,
            playing=bool(c_state.playing),
            valence=c_state.valence,
            arousal=c_state.arousal,
            dominance=c_state.dominance,
            discrete_emotion_id=c_state.discrete_emotion_id,
            emotion_intensity=c_state.emotion_intensity,
            emotion_confidence=c_state.emotion_confidence,
            tempo_bias=c_state.tempo_bias,
            rhythmic_density=c_state.rhythmic_density,
            groove_strength=c_state.groove_strength,
            harmonic_tension=c_state.harmonic_tension,
            harmonic_motion=c_state.harmonic_motion,
            melodic_activity=c_state.melodic_activity,
            texture_density=c_state.texture_density,
            dynamic_range=c_state.dynamic_range,
            track_params=[c_state.track_params[i] for i in range(16)],
            sequence=c_state.sequence,
        )


class ComprehensiveIntegrationManager:
    """
    Real-time bridge between Python Mind and C++ Body.

    Polls the engine's RTState at a configurable frequency and emits
    events through the project's EventBus when state changes are detected.

    Events emitted:
        engine.state_update  — every poll tick (carries full RTSnapshot)
        engine.new_bar       — when bar counter changes
        engine.emotion_change — when emotion VAD changes beyond threshold
        engine.transport      — when playing state changes
    """

    def __init__(
        self,
        lib_path: str | Path,
        event_bus: Any,  # music_brain.agents.events.EventBus
        brain_handle: Optional[ctypes.c_void_p] = None,
        poll_hz: float = 30.0,
        emotion_threshold: float = 0.05,
    ):
        self._lib_path = Path(lib_path)
        self._bus = event_bus
        self._brain = brain_handle
        self._poll_interval = 1.0 / poll_hz
        self._emotion_threshold = emotion_threshold

        self._lib: Optional[ctypes.CDLL] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._prev: Optional[RTSnapshot] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_lib(self) -> None:
        """Load the shared library and bind function signatures."""
        self._lib = ctypes.CDLL(str(self._lib_path))

        # kelly_brain_get_rt_state(const KellyBrain*, KellyRTState*) -> int
        self._lib.kelly_brain_get_rt_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CKellyRTState),
        ]
        self._lib.kelly_brain_get_rt_state.restype = ctypes.c_int

        # kelly_brain_push_rt_param(KellyBrain*, int target, uint8 idx, float val) -> int
        self._lib.kelly_brain_push_rt_param.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint8,
            ctypes.c_float,
        ]
        self._lib.kelly_brain_push_rt_param.restype = ctypes.c_int

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return
        self._load_lib()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("ComprehensiveIntegrationManager started at %.0f Hz", 1.0 / self._poll_interval)

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ComprehensiveIntegrationManager stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        c_state = CKellyRTState()
        while self._running:
            try:
                self._read_state(c_state)
                snap = RTSnapshot.from_c(c_state)
                await self._detect_and_emit(snap)
                self._prev = snap
            except Exception:
                logger.exception("Error in RT poll loop")
            await asyncio.sleep(self._poll_interval)

    def _read_state(self, c_state: CKellyRTState) -> None:
        if self._lib is None:
            return
        rc = self._lib.kelly_brain_get_rt_state(self._brain, ctypes.byref(c_state))
        if rc != 0:
            logger.warning("kelly_brain_get_rt_state returned %d", rc)

    # ------------------------------------------------------------------
    # Change detection & event emission
    # ------------------------------------------------------------------

    async def _detect_and_emit(self, snap: RTSnapshot) -> None:
        # Always emit the full state update
        await self._bus.emit("engine.state_update", snap.__dict__)

        if self._prev is None:
            return

        # New bar
        if snap.bar != self._prev.bar:
            await self._bus.emit(
                "engine.new_bar",
                {
                    "bar": snap.bar,
                    "beat": snap.beat,
                    "bpm": snap.bpm,
                },
            )

        # Emotion change (VAD delta exceeds threshold)
        dv = abs(snap.valence - self._prev.valence)
        da = abs(snap.arousal - self._prev.arousal)
        dd = abs(snap.dominance - self._prev.dominance)
        if max(dv, da, dd) > self._emotion_threshold:
            await self._bus.emit(
                "engine.emotion_change",
                {
                    "valence": snap.valence,
                    "arousal": snap.arousal,
                    "dominance": snap.dominance,
                    "intensity": snap.emotion_intensity,
                    "delta": {"valence": dv, "arousal": da, "dominance": dd},
                },
            )

        # Transport change
        if snap.playing != self._prev.playing:
            await self._bus.emit(
                "engine.transport",
                {
                    "playing": snap.playing,
                    "bpm": snap.bpm,
                    "bar": snap.bar,
                },
            )

    # ------------------------------------------------------------------
    # Parameter push (Python → C++)
    # ------------------------------------------------------------------

    def push_bpm(self, bpm: float) -> bool:
        return self._push_param(RT_TARGET_BPM, 0, bpm)

    def push_emotion(
        self, valence: float, arousal: float, dominance: float, intensity: float = 1.0
    ) -> bool:
        ok = True
        ok &= self._push_param(RT_TARGET_EMOTION, 0, valence)
        ok &= self._push_param(RT_TARGET_EMOTION, 1, arousal)
        ok &= self._push_param(RT_TARGET_EMOTION, 2, dominance)
        ok &= self._push_param(RT_TARGET_EMOTION, 3, intensity)
        return ok

    def push_intent(self, param_name: str, value: float) -> bool:
        intent_map = {
            "tempo_bias": 0,
            "rhythmic_density": 1,
            "groove_strength": 2,
            "harmonic_tension": 3,
            "harmonic_motion": 4,
            "melodic_activity": 5,
            "texture_density": 6,
            "dynamic_range": 7,
        }
        idx = intent_map.get(param_name)
        if idx is None:
            logger.error("Unknown intent parameter: %s", param_name)
            return False
        return self._push_param(RT_TARGET_INTENT, idx, value)

    def push_track_param(self, index: int, value: float) -> bool:
        if not 0 <= index < 16:
            return False
        return self._push_param(RT_TARGET_TRACK_PARAM, index, value)

    def push_transport(self, playing: bool) -> bool:
        return self._push_param(RT_TARGET_TRANSPORT, 0, 1.0 if playing else 0.0)

    def _push_param(self, target: int, index: int, value: float) -> bool:
        if self._lib is None:
            logger.warning("Library not loaded, cannot push param")
            return False
        rc = self._lib.kelly_brain_push_rt_param(self._brain, target, index, ctypes.c_float(value))
        if rc != 0:
            logger.warning("push_rt_param failed: target=%d idx=%d rc=%d", target, index, rc)
            return False
        return True

    # ------------------------------------------------------------------
    # DAW bridge (Phase 4b)
    # ------------------------------------------------------------------

    def send_to_daw(
        self,
        daw_name: str,
        midi_data: list[dict],
        channel: int = 0,
    ) -> bool:
        """
        Send MIDI data to a DAW via the registered bridge.

        Args:
            daw_name: DAW identifier ("ableton", "logic_pro", "reaper", "bitwig")
            midi_data: List of MIDI event dicts, each with:
                - "type": "note_on" | "note_off" | "cc" | "pitch_bend"
                - "note" / "cc" / "value": int
                - "velocity": int (for note_on)
                - "channel": int (optional, overrides channel arg)
            channel: Default MIDI channel for all events.

        Returns:
            True if all events were sent successfully.
        """
        from music_brain.agents.daw_protocol import DAWType, get_daw_bridge

        # Resolve DAW type from name
        try:
            daw_type = DAWType(daw_name)
        except ValueError:
            logger.error("Unknown DAW: %s", daw_name)
            return False

        bridge = get_daw_bridge(daw_type)
        if bridge is None:
            logger.error("No bridge available for %s", daw_name)
            return False

        if not bridge.is_connected:
            if not bridge.connect():
                logger.error("Failed to connect to %s", daw_name)
                return False

        ok = True
        for event in midi_data:
            ch = event.get("channel", channel)
            evt_type = event.get("type", "")
            try:
                if evt_type == "note_on":
                    bridge.send_note_on(event["note"], event.get("velocity", 100), ch)
                elif evt_type == "note_off":
                    bridge.send_note_off(event["note"], ch)
                elif evt_type == "cc":
                    bridge.send_cc(event["cc"], event["value"], ch)
                elif evt_type == "pitch_bend":
                    bridge.send_pitch_bend(event["value"], ch)
                else:
                    logger.warning("Unknown MIDI event type: %s", evt_type)
                    ok = False
            except Exception:
                logger.exception("Failed to send %s event to %s", evt_type, daw_name)
                ok = False

        return ok

    # ------------------------------------------------------------------
    # Direct state access (non-polling)
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Optional[RTSnapshot]:
        """One-shot state read outside the poll loop."""
        if self._lib is None:
            self._load_lib()
        c_state = CKellyRTState()
        self._read_state(c_state)
        return RTSnapshot.from_c(c_state)
