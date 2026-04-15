"""
PRROT Voice Bridge — Phase 4b integration.

Python wrapper around the PRROT pybind11 bindings that integrates voice
analysis into the ComprehensiveIntegrationManager and EventBus.

Usage:
    from music_brain.integrations.prrot_bridge import PRROTBridge

    bridge = PRROTBridge(event_bus=bus)
    bridge.initialize()

    # Analyze audio (numpy float32 array)
    control_data = bridge.process_audio(audio_array, sample_rate=48000)

    # Or use within the integration manager
    manager.voice_bridge = bridge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the compiled pybind11 module
_prrot_available = False
_prrot = None
try:
    # The pybind11 module is built as part of the penta_core_native package
    from penta_core_native import prrot as _prrot
    _prrot_available = True
except ImportError:
    try:
        # Fallback: try direct import if built as standalone module
        import prrot as _prrot  # type: ignore
        _prrot_available = True
    except ImportError:
        logger.info("PRROT bindings not available (not built yet)")


@dataclass
class VoiceAnalysisResult:
    """Result of voice audio analysis."""

    phonemes: list[dict]
    pitch_targets: list[dict]
    breath_markers: list[dict]
    control_data: Any = None  # Raw PhonemeControlData if available

    @property
    def has_data(self) -> bool:
        return len(self.phonemes) > 0 or len(self.pitch_targets) > 0


class PRROTBridge:
    """
    Wraps PRROT engine for Python-side voice analysis.

    Emits events via EventBus when voice analysis produces results:
        voice.phonemes_detected — phoneme timing data
        voice.breath_detected  — breath marker locations
        voice.control_data     — full control data output
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
    ):
        self._bus = event_bus
        self._engine = None
        self._initialized = False

    @property
    def available(self) -> bool:
        return _prrot_available

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> bool:
        """Initialize the PRROT engine."""
        if not _prrot_available:
            logger.warning("PRROT bindings not available — voice features disabled")
            return False

        try:
            self._engine = _prrot.PRROTEngine()
            if self._engine.initialize():
                self._initialized = True
                logger.info("PRROT engine initialized")
                return True
            else:
                logger.error("PRROT engine failed to initialize")
                return False
        except Exception:
            logger.exception("Failed to create PRROT engine")
            return False

    def load_voice_profile(self, profile_id: str, profile_name: str = "") -> bool:
        """Load a voice profile into the engine."""
        if not self._initialized:
            return False
        try:
            profile = _prrot.VoiceProfile()
            profile.profile_id = profile_id
            profile.profile_name = profile_name or profile_id
            return self._engine.load_voice_profile(profile)
        except Exception:
            logger.exception("Failed to load voice profile %s", profile_id)
            return False

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: float = 48000.0,
        tempo_bpm: float = 120.0,
    ) -> Optional[VoiceAnalysisResult]:
        """
        Process audio through PRROT and emit events.

        Args:
            audio: Mono float32 numpy array.
            sample_rate: Audio sample rate in Hz.
            tempo_bpm: Current tempo for timing calculations.

        Returns:
            VoiceAnalysisResult or None if engine not available.
        """
        if not self._initialized:
            return None

        audio = np.ascontiguousarray(audio, dtype=np.float32)

        try:
            # This releases the GIL internally
            control_data = self._engine.process_audio_segment(
                audio, sample_rate, tempo_bpm
            )

            # Convert to Python-friendly format
            phonemes = [
                {
                    "phoneme": _prrot.phoneme_to_string(pt.phoneme),
                    "start_ms": pt.start_time_ms,
                    "duration_ms": pt.duration_ms,
                    "stressed": pt.is_stressed,
                    "prominence": pt.prominence,
                }
                for pt in control_data.phoneme_sequence
            ]

            pitch_targets = [
                {
                    "time_ms": pt.time_ms,
                    "midi_note": pt.midi_note,
                    "cents_offset": pt.cents_offset,
                    "confidence": pt.confidence,
                }
                for pt in control_data.pitch_targets
            ]

            # Detect breath markers separately
            breath_markers_raw = self._engine.detect_breath_markers(audio, sample_rate)
            breath_markers = [
                {
                    "time_ms": bm.time_ms,
                    "duration_ms": bm.duration_ms,
                    "confidence": bm.confidence,
                }
                for bm in breath_markers_raw
            ]

            result = VoiceAnalysisResult(
                phonemes=phonemes,
                pitch_targets=pitch_targets,
                breath_markers=breath_markers,
                control_data=control_data,
            )

            # Emit events
            if self._bus is not None:
                if phonemes:
                    await self._bus.emit("voice.phonemes_detected", {
                        "phonemes": phonemes,
                        "count": len(phonemes),
                    })
                if breath_markers:
                    await self._bus.emit("voice.breath_detected", {
                        "markers": breath_markers,
                        "count": len(breath_markers),
                    })
                await self._bus.emit("voice.control_data", {
                    "phoneme_count": len(phonemes),
                    "pitch_target_count": len(pitch_targets),
                    "breath_count": len(breath_markers),
                })

            return result

        except Exception:
            logger.exception("PRROT processing failed")
            return None

    def analyze_phonemes_sync(
        self,
        audio: np.ndarray,
        sample_rate: float = 48000.0,
    ) -> list[dict]:
        """Synchronous phoneme analysis (no event emission)."""
        if not self._initialized:
            return []
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        try:
            timings = self._engine.analyze_phonemes(audio, sample_rate)
            return [
                {
                    "phoneme": _prrot.phoneme_to_string(pt.phoneme),
                    "start_ms": pt.start_time_ms,
                    "duration_ms": pt.duration_ms,
                    "stressed": pt.is_stressed,
                }
                for pt in timings
            ]
        except Exception:
            logger.exception("Phoneme analysis failed")
            return []
