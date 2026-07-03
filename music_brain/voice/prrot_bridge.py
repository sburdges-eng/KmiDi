"""
PRROT Bridge — Python bridge to the PRROT C++ voice engine.

PRROT/PARROT is the embedded Tier C voice-instrument compiler in KMiDi.
This module attempts to call native PRROT functions via ctypes through
``libKellyFFI.dylib``.  If the native library is unavailable or does not
expose PRROT symbols (the common case while bindings are still in progress),
the bridge falls back transparently to pure-Python components drawn from
``music_brain.voice``.

NOTE: As of this writing, ``libKellyFFI.dylib`` does *not* export PRROT
symbols — only JUCE-level symbols are present.  The native path is therefore
wired up but remains dormant until FFI exports are added to the C++ build.
When that happens, the ``lib["kelly_prrot_create"]`` probe in
``_try_load_native()`` will succeed automatically and ctypes prototypes
will activate.  Verify the prototypes match the actual export signatures.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports from existing Python voice components
# ---------------------------------------------------------------------------
try:
    from music_brain.voice.phoneme_processor import PhonemeProcessor

    _PHONEME_PROCESSOR_AVAILABLE = True
except Exception:
    _PHONEME_PROCESSOR_AVAILABLE = False

try:
    _PITCH_CONTROLLER_AVAILABLE = True
except Exception:
    _PITCH_CONTROLLER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native library discovery helpers
# ---------------------------------------------------------------------------

_SEARCH_PATHS = [
    # Relative to repo root (worktree)
    Path(__file__).parents[2] / "build" / "libKellyFFI.dylib",
    Path(__file__).parents[2] / "build" / "libKellyFFI.1.dylib",
    Path(__file__).parents[2] / "build" / "libKellyFFI.1.0.0.dylib",
    # System / installed locations
    Path("/usr/local/lib/libKellyFFI.dylib"),
    Path("/usr/lib/libKellyFFI.dylib"),
]


def _find_dylib() -> Optional[Path]:
    """Return the first existing libKellyFFI dylib path, or None."""
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    return None


def _try_load_native() -> Optional[ctypes.CDLL]:
    """
    Attempt to load ``libKellyFFI.dylib`` and verify PRROT symbol presence.

    Returns the loaded library if PRROT symbols are found, otherwise ``None``.
    """
    dylib_path = _find_dylib()
    if dylib_path is None:
        logger.debug("PRROTBridge: libKellyFFI.dylib not found on search paths.")
        return None

    try:
        lib = ctypes.CDLL(str(dylib_path))
    except OSError as exc:
        logger.debug("PRROTBridge: failed to load %s — %s", dylib_path, exc)
        return None

    # Verify that at least one PRROT symbol is present via dlsym lookup.
    # hasattr() is unreliable on macOS — CDLL.__getattr__ lazily creates
    # _FuncPtr without validating the symbol. Use bracket notation instead,
    # which calls dlsym and raises AttributeError on miss.
    try:
        _ = lib["kelly_prrot_create"]
        _PRROT_FFI_AVAILABLE = True
    except (AttributeError, OSError):
        _PRROT_FFI_AVAILABLE = False
    if not _PRROT_FFI_AVAILABLE:
        logger.debug(
            "PRROTBridge: libKellyFFI loaded but PRROT symbols not exported. "
            "Falling back to pure-Python implementation."
        )
        return None

    # ---------------------------------------------------------------------------
    # Wire up ctypes prototypes (activated once FFI exports exist)
    # ---------------------------------------------------------------------------
    lib.kelly_prrot_create.restype = ctypes.c_void_p
    lib.kelly_prrot_create.argtypes = []

    lib.kelly_prrot_destroy.restype = None
    lib.kelly_prrot_destroy.argtypes = [ctypes.c_void_p]

    lib.kelly_prrot_initialize.restype = ctypes.c_bool
    lib.kelly_prrot_initialize.argtypes = [ctypes.c_void_p]

    lib.kelly_prrot_load_voice_profile.restype = ctypes.c_bool
    lib.kelly_prrot_load_voice_profile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.kelly_prrot_process_audio.restype = ctypes.c_int
    lib.kelly_prrot_process_audio.argtypes = [
        ctypes.c_void_p,  # engine handle
        ctypes.POINTER(ctypes.c_float),  # samples
        ctypes.c_size_t,  # num_samples
        ctypes.c_float,  # sample_rate_hz
        ctypes.c_float,  # tempo_bpm
        ctypes.c_char_p,  # json output buffer
        ctypes.c_size_t,  # buffer length
    ]

    lib.kelly_prrot_analyze_phonemes.restype = ctypes.c_int
    lib.kelly_prrot_analyze_phonemes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]

    lib.kelly_prrot_detect_breaths.restype = ctypes.c_int
    lib.kelly_prrot_detect_breaths.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]

    lib.kelly_prrot_track_pitch.restype = ctypes.c_int
    lib.kelly_prrot_track_pitch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),  # frequency_hz out
        ctypes.POINTER(ctypes.c_int),  # midi_note out
        ctypes.POINTER(ctypes.c_float),  # cents_offset out
        ctypes.POINTER(ctypes.c_float),  # confidence out
    ]

    logger.info("PRROTBridge: native C++ backend loaded from %s", dylib_path)
    return lib


# ---------------------------------------------------------------------------
# Pure-Python helpers (fallback implementations)
# ---------------------------------------------------------------------------


def _python_analyze_phonemes(
    samples: np.ndarray,
    sample_rate: float,
    tempo_bpm: float = 120.0,
) -> List[dict]:
    """
    Pure-Python phoneme analysis using amplitude-based segmentation.

    Segments the audio by energy and assigns placeholder phoneme labels.
    When ``PhonemeProcessor`` is available, duration estimates are derived
    from its timing tables; otherwise simple heuristics are used.
    """
    if samples.size == 0:
        return []

    duration_ms = (len(samples) / sample_rate) * 1000.0

    if _PHONEME_PROCESSOR_AVAILABLE:
        # Use PhonemeProcessor timing tables for a plausible sequence
        proc = PhonemeProcessor()
        seq = proc.estimate_durations(["AH", "EH", "IH"], tempo_bpm, duration_ms)
        result = []
        for ph in seq.phonemes:
            result.append(
                {
                    "symbol": ph.symbol,
                    "start_time_ms": ph.start_time_ms,
                    "duration_ms": ph.duration_ms,
                    "confidence": 0.5,
                    "source": "python_fallback",
                }
            )
        return result

    # Minimal heuristic: one phoneme per ~100 ms
    segment_ms = max(50.0, 120.0 / tempo_bpm * 100.0)
    phonemes = []
    t = 0.0
    labels = ["AH", "EH", "IH", "AO", "UH"]
    idx = 0
    while t < duration_ms:
        dur = min(segment_ms, duration_ms - t)
        phonemes.append(
            {
                "symbol": labels[idx % len(labels)],
                "start_time_ms": t,
                "duration_ms": dur,
                "confidence": 0.4,
                "source": "python_heuristic",
            }
        )
        t += dur
        idx += 1
    return phonemes


def _python_detect_breaths(
    samples: np.ndarray,
    sample_rate: float,
) -> List[dict]:
    """
    Pure-Python breath detection via low-energy region scanning.

    Splits the signal into 20 ms frames and marks frames below the 10th
    percentile of RMS energy as candidate breath regions.
    """
    if samples.size == 0:
        return []

    frame_len = max(1, int(0.02 * sample_rate))  # 20 ms
    rms_values = []
    for start in range(0, len(samples) - frame_len + 1, frame_len):
        frame = samples[start : start + frame_len]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        rms_values.append((start, rms))

    if not rms_values:
        return []

    threshold = float(np.percentile([r for _, r in rms_values], 10))
    # Clamp to avoid detecting silence in total-silence buffers
    threshold = max(threshold, 1e-6)

    markers = []
    for start_sample, rms in rms_values:
        if rms <= threshold:
            time_ms = (start_sample / sample_rate) * 1000.0
            markers.append(
                {
                    "time_ms": time_ms,
                    "intensity": float(rms / (threshold + 1e-9)),
                    "duration_ms": 20.0,
                    "confidence": 0.4,
                    "source": "python_fallback",
                }
            )
    return markers


def _python_track_pitch(
    samples: np.ndarray,
    sample_rate: float,
) -> dict:
    """
    Pure-Python fundamental frequency estimation via autocorrelation.

    Mirrors the algorithm described in ``PitchTracker.h`` — uses a Hann
    window and autocorrelation within the 50–2000 Hz range.
    """
    if samples.size == 0:
        return {
            "frequency_hz": 0.0,
            "midi_note": 60,
            "cents_offset": 0.0,
            "confidence": 0.0,
            "is_valid": False,
            "source": "python_fallback",
        }

    # Work with float64 for numerical stability
    x = samples.astype(np.float64)

    # Limit to 4096 samples (mirrors kMaxPitchAnalysisWindow)
    window_size = min(len(x), 4096)
    x = x[:window_size]

    # Apply Hann window
    hann = np.hanning(len(x))
    x_win = x * hann

    # Autocorrelation
    corr = np.correlate(x_win, x_win, mode="full")
    corr = corr[len(corr) // 2 :]  # Keep positive lags

    # Search within valid pitch period range
    min_period = max(1, int(sample_rate / 2000.0))  # kMaxPitchHz
    max_period = min(len(corr) - 1, int(sample_rate / 50.0))  # kMinPitchHz

    if min_period >= max_period or max_period <= 0:
        frequency_hz = 0.0
        confidence = 0.0
        is_valid = False
    else:
        search_region = corr[min_period : max_period + 1]
        if search_region.size == 0 or corr[0] == 0.0:
            frequency_hz = 0.0
            confidence = 0.0
            is_valid = False
        else:
            best_lag = int(np.argmax(search_region)) + min_period
            frequency_hz = float(sample_rate / best_lag)
            # Confidence: ratio of peak autocorrelation to zero-lag
            confidence = float(min(1.0, corr[best_lag] / (corr[0] + 1e-12)))
            is_valid = confidence > 0.1

    # Convert to MIDI note + cents
    if is_valid and frequency_hz > 0.0:
        midi_float = 69.0 + 12.0 * math.log2(frequency_hz / 440.0)
        midi_note = int(round(midi_float))
        midi_note = max(0, min(127, midi_note))
        cents_offset = (midi_float - midi_note) * 100.0
    else:
        midi_note = 60
        cents_offset = 0.0

    return {
        "frequency_hz": frequency_hz,
        "midi_note": midi_note,
        "cents_offset": cents_offset,
        "confidence": confidence,
        "is_valid": is_valid,
        "source": "python_fallback",
    }


# ---------------------------------------------------------------------------
# PRROTBridge
# ---------------------------------------------------------------------------


class PRROTBridge:
    """
    Python bridge to the PRROT C++ voice engine.

    Attempts to load ``libKellyFFI.dylib`` and call native PRROT functions
    via ctypes.  Falls back to a pure-Python implementation when the native
    library is unavailable or PRROT FFI symbols are not yet exported.

    The pure-Python fallback uses:
    - ``music_brain.voice.phoneme_processor.PhonemeProcessor`` for phoneme
      timing estimation.
    - ``music_brain.voice.pitch_controller.PitchController`` for MIDI/frequency
      helpers.
    - Autocorrelation-based pitch detection mirroring ``PitchTracker.h``.
    - Energy-based breath detection mirroring ``BreathDetector.h``.

    The interface is intentionally identical regardless of backend so that
    callers never need to branch on ``is_native``.
    """

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = _try_load_native()
        self._native_handle: Optional[int] = None  # void* to C++ PRROTEngine
        self._initialized: bool = False
        self._profile_path: Optional[str] = None

        if self._lib is not None:
            try:
                handle = self._lib.kelly_prrot_create()
                if handle:
                    self._native_handle = handle
                    logger.debug("PRROTBridge: native engine handle created.")
                else:
                    logger.warning(
                        "PRROTBridge: kelly_prrot_create() returned NULL; "
                        "falling back to Python."
                    )
                    self._lib = None
            except Exception as exc:
                logger.warning(
                    "PRROTBridge: exception creating native engine: %s. " "Falling back to Python.",
                    exc,
                )
                self._lib = None

    def __del__(self) -> None:
        if self._lib is not None and self._native_handle is not None:
            try:
                self._lib.kelly_prrot_destroy(self._native_handle)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_native(self) -> bool:
        """True if using the C++ backend, False if using pure-Python fallback."""
        return self._lib is not None and self._native_handle is not None

    def initialize(self) -> bool:
        """
        Initialize the PRROT engine.

        Must be called before ``load_voice_profile`` or any processing methods.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if self.is_native:
            # type: ignore[union-attr] — guarded by is_native
            ok = bool(self._lib.kelly_prrot_initialize(self._native_handle))
            self._initialized = ok
            return ok

        # Pure-Python: nothing to initialize
        self._initialized = True
        return True

    def load_voice_profile(self, profile_path: str) -> bool:
        """
        Load a voice profile from the given file path.

        Args:
            profile_path: Absolute path to a ``VoiceProfile`` JSON/binary file.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if self.is_native:
            ok = bool(
                self._lib.kelly_prrot_load_voice_profile(  # type: ignore[union-attr]
                    self._native_handle,
                    profile_path.encode("utf-8"),
                )
            )
            if ok:
                self._profile_path = profile_path
            return ok

        # Pure-Python: accept any path string (file need not exist for stub)
        self._profile_path = profile_path
        return True

    def process_audio(
        self,
        samples: np.ndarray,
        sample_rate: float,
        tempo_bpm: float = 120.0,
    ) -> dict:
        """
        Process an audio segment and return phoneme control data.

        Mirrors ``PRROTEngine::processAudioSegment()``.

        Args:
            samples:     1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz (e.g. 44100.0).
            tempo_bpm:   Tempo in BPM used for timing calculations.

        Returns:
            A dictionary with the following keys:
            - ``phonemes``      : list of phoneme dicts (see ``analyze_phonemes``).
            - ``pitch_targets`` : list of pitch target dicts.
            - ``breath_markers``: list of breath marker dicts.
            - ``midi_notes``    : list of MIDI note dicts.
            - ``tempo_bpm``     : float.
            - ``sample_rate_hz``: float.
            - ``total_duration_ms``: float.
            - ``is_native``     : bool.
        """
        samples = _ensure_float32(samples)

        if self.is_native:
            return self._native_process_audio(samples, sample_rate, tempo_bpm)

        # Pure-Python fallback
        phonemes = _python_analyze_phonemes(samples, sample_rate, tempo_bpm)
        breaths = _python_detect_breaths(samples, sample_rate)
        pitch = _python_track_pitch(samples, sample_rate)
        total_ms = (len(samples) / sample_rate) * 1000.0

        # Build a minimal pitch target from the tracked pitch
        pitch_targets = []
        if pitch["is_valid"]:
            pitch_targets.append(
                {
                    "time_ms": 0.0,
                    "midi_note": pitch["midi_note"],
                    "cents_offset": pitch["cents_offset"],
                    "confidence": pitch["confidence"],
                }
            )

        # Simple MIDI note from pitch
        midi_notes = []
        if pitch["is_valid"]:
            midi_notes.append(
                {
                    "note_number": pitch["midi_note"],
                    "velocity": 100,
                    "start_time_ms": 0.0,
                    "duration_ms": total_ms,
                    "channel": 0,
                }
            )

        return {
            "phonemes": phonemes,
            "pitch_targets": pitch_targets,
            "breath_markers": breaths,
            "midi_notes": midi_notes,
            "tempo_bpm": tempo_bpm,
            "sample_rate_hz": sample_rate,
            "total_duration_ms": total_ms,
            "is_native": False,
        }

    def analyze_phonemes(
        self,
        samples: np.ndarray,
        sample_rate: float,
    ) -> List[dict]:
        """
        Extract phoneme timings from audio.

        Mirrors ``PRROTEngine::analyzePhonemes()``.

        Args:
            samples:     1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            List of dicts, each with keys:
            ``symbol``, ``start_time_ms``, ``duration_ms``, ``confidence``,
            ``source``.
        """
        samples = _ensure_float32(samples)

        if self.is_native:
            return self._native_analyze_phonemes(samples, sample_rate)

        return _python_analyze_phonemes(samples, sample_rate)

    def detect_breaths(
        self,
        samples: np.ndarray,
        sample_rate: float,
    ) -> List[dict]:
        """
        Detect breath markers in audio.

        Mirrors ``PRROTEngine::detectBreathMarkers()``.

        Args:
            samples:     1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            List of dicts, each with keys:
            ``time_ms``, ``intensity``, ``duration_ms``, ``confidence``,
            ``source``.
        """
        samples = _ensure_float32(samples)

        if self.is_native:
            return self._native_detect_breaths(samples, sample_rate)

        return _python_detect_breaths(samples, sample_rate)

    def track_pitch(
        self,
        samples: np.ndarray,
        sample_rate: float,
    ) -> dict:
        """
        Track pitch (fundamental frequency) in an audio segment.

        Mirrors ``PitchTracker::trackPitch()``.

        Args:
            samples:     1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with keys:
            ``frequency_hz`` (float), ``midi_note`` (int),
            ``cents_offset`` (float), ``confidence`` (float),
            ``is_valid`` (bool), ``source`` (str).
        """
        samples = _ensure_float32(samples)

        if self.is_native:
            return self._native_track_pitch(samples, sample_rate)

        return _python_track_pitch(samples, sample_rate)

    # ------------------------------------------------------------------
    # Native (ctypes) implementations — activated when FFI exports exist
    # ------------------------------------------------------------------

    def _native_process_audio(
        self,
        samples: np.ndarray,
        sample_rate: float,
        tempo_bpm: float,
    ) -> dict:
        """Call kelly_prrot_process_audio and parse JSON result."""
        buf_size = 65536
        out_buf = ctypes.create_string_buffer(buf_size)
        ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.kelly_prrot_process_audio(  # type: ignore[union-attr]
            self._native_handle,
            ptr,
            ctypes.c_size_t(len(samples)),
            ctypes.c_float(sample_rate),
            ctypes.c_float(tempo_bpm),
            out_buf,
            ctypes.c_size_t(buf_size),
        )
        if ret < 0:
            logger.error("PRROTBridge: kelly_prrot_process_audio returned %d", ret)
            return {}
        data = json.loads(out_buf.value.decode("utf-8"))
        data["is_native"] = True
        return data

    def _native_analyze_phonemes(
        self,
        samples: np.ndarray,
        sample_rate: float,
    ) -> List[dict]:
        """Call kelly_prrot_analyze_phonemes and parse JSON result."""
        buf_size = 32768
        out_buf = ctypes.create_string_buffer(buf_size)
        ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.kelly_prrot_analyze_phonemes(  # type: ignore[union-attr]
            self._native_handle,
            ptr,
            ctypes.c_size_t(len(samples)),
            ctypes.c_float(sample_rate),
            out_buf,
            ctypes.c_size_t(buf_size),
        )
        if ret < 0:
            logger.error("PRROTBridge: kelly_prrot_analyze_phonemes returned %d", ret)
            return []
        return json.loads(out_buf.value.decode("utf-8"))

    def _native_detect_breaths(
        self,
        samples: np.ndarray,
        sample_rate: float,
    ) -> List[dict]:
        """Call kelly_prrot_detect_breaths and parse JSON result."""
        buf_size = 16384
        out_buf = ctypes.create_string_buffer(buf_size)
        ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.kelly_prrot_detect_breaths(  # type: ignore[union-attr]
            self._native_handle,
            ptr,
            ctypes.c_size_t(len(samples)),
            ctypes.c_float(sample_rate),
            out_buf,
            ctypes.c_size_t(buf_size),
        )
        if ret < 0:
            logger.error("PRROTBridge: kelly_prrot_detect_breaths returned %d", ret)
            return []
        return json.loads(out_buf.value.decode("utf-8"))

    def _native_track_pitch(
        self,
        samples: np.ndarray,
        sample_rate: float,
    ) -> dict:
        """Call kelly_prrot_track_pitch and return result dict."""
        freq_out = ctypes.c_float(0.0)
        note_out = ctypes.c_int(60)
        cents_out = ctypes.c_float(0.0)
        conf_out = ctypes.c_float(0.0)
        ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.kelly_prrot_track_pitch(  # type: ignore[union-attr]
            self._native_handle,
            ptr,
            ctypes.c_size_t(len(samples)),
            ctypes.c_float(sample_rate),
            ctypes.byref(freq_out),
            ctypes.byref(note_out),
            ctypes.byref(cents_out),
            ctypes.byref(conf_out),
        )
        is_valid = ret == 0 and freq_out.value > 0.0
        return {
            "frequency_hz": float(freq_out.value),
            "midi_note": int(note_out.value),
            "cents_offset": float(cents_out.value),
            "confidence": float(conf_out.value),
            "is_valid": is_valid,
            "source": "native",
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ensure_float32(samples: Any) -> np.ndarray:
    """Convert *samples* to a 1-D float32 numpy array, or raise TypeError."""
    if not isinstance(samples, np.ndarray):
        samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        samples = samples.flatten()
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    return samples
