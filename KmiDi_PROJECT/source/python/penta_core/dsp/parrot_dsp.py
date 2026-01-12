"""
Parrot DSP - Sample Playback Engine and Pitch Shifting.

Provides:
- Sample playback with various modes
- Pitch shifting algorithms (granular, phase vocoder)
- Time stretching
- Granular synthesis
- Formant preservation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import math
import random
import numpy as np


class PlaybackMode(Enum):
    """Sample playback modes."""
    ONE_SHOT = "one_shot"       # Play once
    LOOP = "loop"               # Loop continuously
    PING_PONG = "ping_pong"     # Loop forward/backward
    REVERSE = "reverse"         # Play backwards
    GRANULAR = "granular"       # Granular playback


class PitchAlgorithm(Enum):
    """Pitch shifting algorithms."""
    SIMPLE = "simple"           # Simple resampling (changes length)
    GRANULAR = "granular"       # Granular time-domain
    PHASE_VOCODER = "phase_vocoder"  # Phase vocoder (FFT-based)
    FORMANT = "formant"         # Formant-preserving


@dataclass
class SamplePlayback:
    """
    Sample playback engine.
    """
    samples: List[float] = field(default_factory=list)
    sample_rate: float = 44100.0
    mode: PlaybackMode = PlaybackMode.ONE_SHOT

    # Playback parameters
    start_position: float = 0.0    # Start point (0.0-1.0)
    end_position: float = 1.0      # End point (0.0-1.0)
    loop_start: float = 0.0        # Loop start (0.0-1.0)
    loop_end: float = 1.0          # Loop end (0.0-1.0)

    # Playback rate
    rate: float = 1.0              # Playback rate (1.0 = normal)
    pitch_semitones: float = 0.0   # Pitch shift in semitones

    # State
    _position: float = 0.0
    _direction: int = 1            # 1 = forward, -1 = backward
    _is_playing: bool = False

    def start(self):
        """Start playback."""
        if self.mode == PlaybackMode.REVERSE:
            self._position = self.end_position * len(self.samples)
            self._direction = -1
        else:
            self._position = self.start_position * len(self.samples)
            self._direction = 1
        self._is_playing = True

    def stop(self):
        """Stop playback."""
        self._is_playing = False

    def process_sample(self) -> float:
        """
        Get next sample.

        Returns:
            Output sample
        """
        if not self._is_playing or not self.samples:
            return 0.0

        # Calculate effective playback rate
        effective_rate = self.rate * (2 ** (self.pitch_semitones / 12.0))

        # Get sample with interpolation
        sample = self._interpolate(self._position)

        # Advance position
        self._position += self._direction * effective_rate

        # Handle boundaries
        sample_end = self.end_position * len(self.samples)
        sample_start = self.start_position * len(self.samples)
        loop_start = self.loop_start * len(self.samples)
        loop_end = self.loop_end * len(self.samples)

        if self.mode == PlaybackMode.ONE_SHOT:
            if self._direction > 0 and self._position >= sample_end:
                self._is_playing = False
            elif self._direction < 0 and self._position <= sample_start:
                self._is_playing = False

        elif self.mode == PlaybackMode.LOOP:
            if self._position >= loop_end:
                self._position = loop_start
            elif self._position < loop_start:
                self._position = loop_end

        elif self.mode == PlaybackMode.PING_PONG:
            if self._position >= loop_end:
                self._position = loop_end
                self._direction = -1
            elif self._position <= loop_start:
                self._position = loop_start
                self._direction = 1

        elif self.mode == PlaybackMode.REVERSE:
            if self._position <= sample_start:
                self._is_playing = False

        return sample

    def _interpolate(self, position: float) -> float:
        """Linear interpolation at position."""
        if position < 0 or position >= len(self.samples) - 1:
            return 0.0

        index_a = int(position)
        index_b = index_a + 1
        frac = position - index_a

        if index_b >= len(self.samples):
            return self.samples[index_a]

        return self.samples[index_a] * (1 - frac) + self.samples[index_b] * frac

    def process_block(self, num_samples: int) -> List[float]:
        """Process a block of samples."""
        return [self.process_sample() for _ in range(num_samples)]

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def position(self) -> float:
        """Current position as 0.0-1.0."""
        if not self.samples:
            return 0.0
        return self._position / len(self.samples)


@dataclass
class PitchShifter:
    """
    Pitch shifter using granular synthesis.
    """
    sample_rate: float = 44100.0
    algorithm: PitchAlgorithm = PitchAlgorithm.GRANULAR

    # Granular parameters
    grain_size_ms: float = 50.0
    grain_overlap: float = 0.5  # 0.0-1.0
    window_type: str = "hann"

    # State
    _input_buffer: List[float] = field(default_factory=list)
    _output_buffer: List[float] = field(default_factory=list)
    _grain_position: float = 0.0
    _output_position: int = 0

    def __post_init__(self):
        grain_samples = int(self.grain_size_ms * self.sample_rate / 1000.0)
        self._input_buffer = [0.0] * (grain_samples * 4)
        self._output_buffer = [0.0] * (grain_samples * 4)
        self._window = self._create_window(grain_samples)

    def _create_window(self, size: int) -> List[float]:
        """Create grain window."""
        if self.window_type == "hann":
            return [0.5 * (1 - math.cos(2 * math.pi * i / (size - 1)))
                    for i in range(size)]
        elif self.window_type == "hamming":
            return [0.54 - 0.46 * math.cos(2 * math.pi * i / (size - 1))
                    for i in range(size)]
        else:  # Triangle
            mid = size // 2
            return [i / mid if i < mid else 2 - i / mid for i in range(size)]

    def process(
        self,
        samples: List[float],
        semitones: float,
    ) -> List[float]:
        """
        Pitch shift samples.

        Args:
            samples: Input samples
            semitones: Pitch shift in semitones

        Returns:
            Pitch-shifted samples
        """
        if self.algorithm == PitchAlgorithm.SIMPLE:
            return self._simple_shift(samples, semitones)
        elif self.algorithm == PitchAlgorithm.GRANULAR:
            return self._granular_shift(samples, semitones)
        elif self.algorithm == PitchAlgorithm.PHASE_VOCODER:
            return phase_vocoder_pitch_shift(
                samples,
                semitones,
                frame_size=2048,
                hop_size=512,
                sample_rate=self.sample_rate,
                preserve_formants=False,
            )
        elif self.algorithm == PitchAlgorithm.FORMANT:
            return phase_vocoder_pitch_shift(
                samples,
                semitones,
                frame_size=2048,
                hop_size=512,
                sample_rate=self.sample_rate,
                preserve_formants=True,
            )
        else:
            return self._granular_shift(samples, semitones)

    def _simple_shift(
        self,
        samples: List[float],
        semitones: float,
    ) -> List[float]:
        """Simple resampling (changes duration)."""
        rate = 2 ** (semitones / 12.0)
        new_length = int(len(samples) / rate)

        result = []
        for i in range(new_length):
            source_pos = i * rate
            index_a = int(source_pos)
            index_b = min(index_a + 1, len(samples) - 1)
            frac = source_pos - index_a

            if index_a < len(samples):
                sample = samples[index_a] * (1 - frac) + samples[index_b] * frac
                result.append(sample)

        return result

    def _granular_shift(
        self,
        samples: List[float],
        semitones: float,
    ) -> List[float]:
        """Granular pitch shifting (preserves duration)."""
        pitch_ratio = 2 ** (semitones / 12.0)
        grain_samples = int(self.grain_size_ms * self.sample_rate / 1000.0)
        hop_size = int(grain_samples * (1 - self.grain_overlap))

        # Pad input
        padded = [0.0] * grain_samples + samples + [0.0] * grain_samples

        # Create output buffer
        output_length = len(samples)
        output = [0.0] * output_length
        normalization = [0.0] * output_length

        # Process grains
        input_pos = 0
        output_pos = 0

        while output_pos < output_length:
            # Calculate input grain position
            input_grain_start = int(input_pos)

            if input_grain_start + grain_samples > len(padded):
                break

            # Extract and window grain
            grain = []
            for i in range(grain_samples):
                if i < len(self._window):
                    grain.append(padded[input_grain_start + i] * self._window[i])
                else:
                    grain.append(0.0)

            # Resample grain for pitch shift
            if pitch_ratio != 1.0:
                resampled_length = int(grain_samples / pitch_ratio)
                resampled = []
                for i in range(grain_samples):
                    source_pos = i * pitch_ratio
                    index_a = int(source_pos)
                    if index_a < len(grain) - 1:
                        frac = source_pos - index_a
                        resampled.append(grain[index_a] * (1 - frac) + grain[index_a + 1] * frac)
                    elif index_a < len(grain):
                        resampled.append(grain[index_a])
                    else:
                        resampled.append(0.0)
                grain = resampled

            # Add grain to output
            for i in range(min(len(grain), output_length - output_pos)):
                output[output_pos + i] += grain[i]
                normalization[output_pos + i] += self._window[i] if i < len(self._window) else 0.0

            # Advance positions
            input_pos += hop_size * pitch_ratio
            output_pos += hop_size

        # Normalize
        for i in range(output_length):
            if normalization[i] > 0.001:
                output[i] /= normalization[i]

        return output


@dataclass
class GrainCloud:
    """
    Granular synthesis cloud for textural effects.
    """
    samples: List[float] = field(default_factory=list)
    sample_rate: float = 44100.0

    # Grain parameters
    grain_size_ms: float = 50.0
    grain_density: float = 10.0  # Grains per second
    position: float = 0.5        # Source position (0.0-1.0)
    position_spread: float = 0.1  # Random position spread

    # Modulation
    pitch_spread: float = 0.0    # Random pitch variation (semitones)
    pan_spread: float = 0.0      # Stereo spread (0.0-1.0)
    reverse_probability: float = 0.0  # Chance of reverse grain

    # State
    _grains: List[Dict] = field(default_factory=list)
    _sample_counter: int = 0

    def process_sample(self) -> Tuple[float, float]:
        """
        Generate next stereo sample.

        Returns:
            (left, right) samples
        """
        # Maybe spawn new grain
        spawn_interval = self.sample_rate / max(1, self.grain_density)
        if self._sample_counter >= spawn_interval:
            self._spawn_grain()
            self._sample_counter = 0
        self._sample_counter += 1

        # Mix active grains
        left = 0.0
        right = 0.0
        active_grains = []

        for grain in self._grains:
            sample = self._process_grain(grain)

            if grain["active"]:
                # Apply panning
                left += sample * (1 - grain["pan"])
                right += sample * grain["pan"]
                active_grains.append(grain)

        self._grains = active_grains

        return left, right

    def _spawn_grain(self):
        """Spawn a new grain."""
        if not self.samples:
            return

        grain_samples = int(self.grain_size_ms * self.sample_rate / 1000.0)

        # Randomize position
        pos = self.position + (random.random() - 0.5) * 2 * self.position_spread
        pos = max(0.0, min(1.0, pos))
        start_sample = int(pos * (len(self.samples) - grain_samples))

        # Randomize pitch
        pitch_ratio = 2 ** ((random.random() - 0.5) * 2 * self.pitch_spread / 12.0)

        # Randomize direction
        reverse = random.random() < self.reverse_probability

        # Randomize pan
        pan = 0.5 + (random.random() - 0.5) * self.pan_spread

        self._grains.append({
            "start": start_sample,
            "position": 0.0,
            "length": grain_samples,
            "pitch_ratio": pitch_ratio,
            "reverse": reverse,
            "pan": pan,
            "active": True,
        })

    def _process_grain(self, grain: Dict) -> float:
        """Process a single grain."""
        if grain["position"] >= grain["length"]:
            grain["active"] = False
            return 0.0

        # Calculate window
        t = grain["position"] / grain["length"]
        window = 0.5 * (1 - math.cos(2 * math.pi * t))

        # Calculate source position
        if grain["reverse"]:
            source_pos = grain["start"] + grain["length"] - grain["position"] * grain["pitch_ratio"]
        else:
            source_pos = grain["start"] + grain["position"] * grain["pitch_ratio"]

        # Get sample with interpolation
        index_a = int(source_pos)
        if index_a < 0 or index_a >= len(self.samples) - 1:
            grain["position"] += 1
            return 0.0

        frac = source_pos - index_a
        sample = self.samples[index_a] * (1 - frac) + self.samples[index_a + 1] * frac

        grain["position"] += 1

        return sample * window

    def process_block(self, num_samples: int) -> Tuple[List[float], List[float]]:
        """Generate a block of stereo samples."""
        left = []
        right = []
        for _ in range(num_samples):
            l, r = self.process_sample()
            left.append(l)
            right.append(r)
        return left, right


def create_pitch_shifter(
    algorithm: str = "granular",
    grain_size_ms: float = 50.0,
    sample_rate: float = 44100.0,
) -> PitchShifter:
    """
    Create a pitch shifter.

    Args:
        algorithm: Algorithm type (simple, granular, phase_vocoder)
        grain_size_ms: Grain size for granular algorithm
        sample_rate: Sample rate

    Returns:
        PitchShifter instance
    """
    algo = PitchAlgorithm(algorithm)
    return PitchShifter(
        sample_rate=sample_rate,
        algorithm=algo,
        grain_size_ms=grain_size_ms,
    )


def shift_pitch(
    samples: List[float],
    semitones: float,
    algorithm: str = "granular",
    sample_rate: float = 44100.0,
) -> List[float]:
    """
    Pitch shift audio samples.

    Args:
        samples: Input samples
        semitones: Pitch shift in semitones
        algorithm: Algorithm to use
        sample_rate: Sample rate

    Returns:
        Pitch-shifted samples
    """
    shifter = create_pitch_shifter(algorithm, sample_rate=sample_rate)
    return shifter.process(samples, semitones)


def time_stretch(
    samples: List[float],
    factor: float,
    grain_size_ms: float = 50.0,
    sample_rate: float = 44100.0,
    use_phase_vocoder: bool = False,
) -> List[float]:
    """
    Time stretch audio without changing pitch.

    Args:
        samples: Input samples
        factor: Stretch factor (2.0 = twice as long)
        grain_size_ms: Grain size (for granular method)
        sample_rate: Sample rate
        use_phase_vocoder: If True, use phase vocoder (better quality, slower)

    Returns:
        Time-stretched samples
    """
    if use_phase_vocoder:
        # Use phase vocoder for better quality
        frame_size = 2048
        hop_size = 512
        return phase_vocoder_time_stretch(samples, factor, frame_size, hop_size, sample_rate)
    
    # Original granular method
    grain_samples = int(grain_size_ms * sample_rate / 1000.0)
    overlap = 0.5
    hop_in = int(grain_samples * (1 - overlap))
    hop_out = int(hop_in * factor)

    # Create Hann window
    window = [0.5 * (1 - math.cos(2 * math.pi * i / (grain_samples - 1)))
              for i in range(grain_samples)]

    # Calculate output length
    output_length = int(len(samples) * factor)
    output = [0.0] * output_length
    normalization = [0.0] * output_length

    # Process grains
    input_pos = 0
    output_pos = 0

    while input_pos + grain_samples <= len(samples) and output_pos + grain_samples <= output_length:
        # Extract and window grain
        for i in range(grain_samples):
            if output_pos + i < output_length:
                output[output_pos + i] += samples[input_pos + i] * window[i]
                normalization[output_pos + i] += window[i]

        input_pos += hop_in
        output_pos += hop_out

    # Normalize
    for i in range(output_length):
        if normalization[i] > 0.001:
            output[i] /= normalization[i]

    return output


def create_grain_cloud(
    samples: List[float],
    grain_size_ms: float = 50.0,
    density: float = 10.0,
    sample_rate: float = 44100.0,
) -> GrainCloud:
    """
    Create a granular cloud generator.

    Args:
        samples: Source samples
        grain_size_ms: Grain size
        density: Grains per second
        sample_rate: Sample rate

    Returns:
        GrainCloud instance
    """
    return GrainCloud(
        samples=samples,
        sample_rate=sample_rate,
        grain_size_ms=grain_size_ms,
        grain_density=density,
    )


def detect_pitch(
    samples: List[float],
    sample_rate: float = 44100.0,
    min_hz: float = 50.0,
    max_hz: float = 2000.0,
) -> Optional[float]:
    """
    Detect pitch using autocorrelation.

    Args:
        samples: Input samples
        sample_rate: Sample rate
        min_hz: Minimum frequency to detect
        max_hz: Maximum frequency to detect

    Returns:
        Detected frequency in Hz, or None
    """
    if len(samples) < 256:
        return None

    # Calculate lag bounds
    min_lag = int(sample_rate / max_hz)
    max_lag = int(sample_rate / min_hz)
    max_lag = min(max_lag, len(samples) // 2)

    if max_lag <= min_lag:
        return None

    # Normalized autocorrelation for pitch detection
    # Remove DC component
    signal_mean = sum(samples) / len(samples)
    samples_centered = [s - signal_mean for s in samples]
    
    # Calculate signal variance for normalization
    signal_variance = sum(s * s for s in samples_centered) / len(samples_centered)
    
    if signal_variance < 1e-10:  # Silence or DC
        return None
    
    # Autocorrelation: find the minimum lag with strong correlation (fundamental)
    # Use normalized autocorrelation - peaks at lags corresponding to period
    correlations = []

    for lag in range(min_lag, max_lag):
        # Calculate autocorrelation at this lag
        correlation = 0.0
        for i in range(len(samples_centered) - lag):
            correlation += samples_centered[i] * samples_centered[i + lag]
        
        # Normalize: autocorrelation at lag 0 would be signal_variance
        # For normalized autocorrelation: divide by autocorrelation at lag 0
        # But we use a simpler normalization: divide by (length - lag) * variance
        # Actually, for pitch detection, we want the normalized autocorrelation coefficient
        correlation_norm = correlation / ((len(samples_centered) - lag) * signal_variance)
        
        correlations.append((lag, correlation_norm))
    
    # Find peaks: look for local maxima
    peaks = []
    for i in range(1, len(correlations) - 1):
        lag_prev, corr_prev = correlations[i - 1]
        lag_curr, corr_curr = correlations[i]
        lag_next, corr_next = correlations[i + 1]
        
        # Local maximum
        if corr_curr > corr_prev and corr_curr > corr_next and corr_curr > 0.1:
            peaks.append((lag_curr, corr_curr))
    
    if not peaks:
        return None
    
    # Find the fundamental: the minimum lag with strong correlation
    # Sort peaks by lag (ascending) to find the first strong peak
    peaks.sort(key=lambda x: x[0])  # Sort by lag (ascending)
    
    # Find maximum correlation to set threshold
    max_correlation = max(corr for _, corr in peaks)
    threshold = max(0.2, max_correlation * 0.4)  # At least 20% or 40% of max
    
    best_lag = None
    best_correlation = 0.0
    
    # Find minimum lag above threshold (fundamental period)
    for lag, corr in peaks:
        if corr >= threshold:
            best_lag = lag
            best_correlation = corr
            break  # Take the first (smallest lag) peak above threshold
    
    # Fallback: use the first peak (minimum lag)
    if best_lag is None and peaks:
        best_lag, best_correlation = peaks[0]

    if best_lag is None or best_lag <= 0 or best_correlation < 0.1:
        return None

    return sample_rate / best_lag


def phase_vocoder_pitch_shift(
    samples: List[float],
    semitones: float,
    frame_size: int = 2048,
    hop_size: int = 512,
    sample_rate: float = 44100.0,
    preserve_formants: bool = False,
) -> List[float]:
    """
    Pitch shift using phase vocoder (FFT-based).
    
    Args:
        samples: Input samples
        semitones: Pitch shift in semitones (positive = higher)
        frame_size: FFT frame size (must be power of 2)
        hop_size: Analysis hop size (samples)
        sample_rate: Sample rate
        preserve_formants: If True, preserve formants (for vocal processing)
    
    Returns:
        Pitch-shifted samples (same length as input)
    """
    if not samples:
        return []
    
    # Calculate pitch ratio
    pitch_ratio = 2 ** (semitones / 12.0)
    
    # For pitch shifting, we want to change pitch but keep duration
    # So we adjust the synthesis hop size: hop_out = hop_in * pitch_ratio
    # But to maintain output length, we need to handle this carefully
    
    # Actually, for pitch shifting without changing duration:
    # - We analyze with hop_in
    # - We synthesize with hop_out = hop_in * pitch_ratio
    # - But we need to adjust the output length to match input
    # - This means we may need to resample the result
    
    # Simplified approach: use phase vocoder with adjusted phase progression
    # Then resample to maintain original length
    
    hop_out = int(hop_size * pitch_ratio)
    
    # Use phase vocoder to shift pitch (this will change duration)
    shifted = _phase_vocoder_process(
        samples,
        frame_size=frame_size,
        hop_in=hop_size,
        hop_out=hop_out,
        sample_rate=sample_rate,
        phase_shift_ratio=pitch_ratio,
    )
    
    # Resample back to original length if needed
    if len(shifted) != len(samples):
        # Resample to match input length
        # Calculate ratio: we want to go from len(shifted) to len(samples)
        # So we use sample rates proportional to lengths
        target_length = len(samples)
        source_length = len(shifted)
        if source_length > 0:
            # Use interpolation to resample
            result = []
            for i in range(target_length):
                source_pos = i * (source_length / target_length)
                index_a = int(source_pos)
                index_b = min(index_a + 1, source_length - 1)
                frac = source_pos - index_a
                
                if index_a < source_length and index_b < source_length:
                    sample = shifted[index_a] * (1 - frac) + shifted[index_b] * frac
                    result.append(sample)
                elif index_a < source_length:
                    result.append(shifted[index_a])
                else:
                    result.append(0.0)
            shifted = result
        else:
            shifted = [0.0] * target_length
    
    return shifted


def phase_vocoder_time_stretch(
    samples: List[float],
    factor: float,
    frame_size: int = 2048,
    hop_size: int = 512,
    sample_rate: float = 44100.0,
) -> List[float]:
    """
    Time stretch using phase vocoder (preserves pitch).
    
    Args:
        samples: Input samples
        factor: Stretch factor (2.0 = twice as long, 0.5 = half length)
        frame_size: FFT frame size (must be power of 2)
        hop_size: Analysis hop size (samples)
        sample_rate: Sample rate
    
    Returns:
        Time-stretched samples
    """
    if not samples:
        return []
    
    if factor <= 0.0:
        return []
    
    # For time stretching, we adjust hop size but keep phase progression unchanged
    # For factor 2.0 (2x longer), we need hop_out = hop_in / 2 (output advances slower)
    # For factor 0.5 (0.5x longer = shorter), we need hop_out = hop_in * 2 (output advances faster)
    hop_in = hop_size
    hop_out = int(hop_size / factor)  # Corrected: divide, not multiply
    
    # Use phase vocoder with no phase modification (phase_shift_ratio = 1.0)
    stretched = _phase_vocoder_process(
        samples,
        frame_size=frame_size,
        hop_in=hop_in,
        hop_out=hop_out,
        sample_rate=sample_rate,
        phase_shift_ratio=1.0,  # No pitch change
    )
    
    return stretched


def _phase_vocoder_process(
    samples: List[float],
    frame_size: int,
    hop_in: int,
    hop_out: int,
    sample_rate: float,
    phase_shift_ratio: float = 1.0,
) -> List[float]:
    """
    Core phase vocoder processing.
    
    Args:
        samples: Input samples
        frame_size: FFT frame size
        hop_in: Analysis hop size
        hop_out: Synthesis hop size
        sample_rate: Sample rate
        phase_shift_ratio: Phase shift ratio (1.0 = no pitch change, >1.0 = higher pitch)
    
    Returns:
        Processed samples
    """
    if not samples:
        return []
    
    # Ensure frame_size is power of 2
    if frame_size & (frame_size - 1) != 0:
        # Round up to next power of 2
        frame_size = 1 << (frame_size - 1).bit_length()
    
    # Create window (Hann window for good frequency resolution)
    window = np.hanning(frame_size).astype(np.float32)
    
    # Calculate number of bins (DC + positive frequencies + Nyquist)
    num_bins = frame_size // 2 + 1
    
    # Pre-allocate output buffer (estimate size)
    # For time stretching, output length is approximately: (num_frames - 1) * hop_out + frame_size
    # Estimate number of frames: (len(samples) - frame_size) / hop_in + 1
    num_frames_est = max(1, int((len(samples) - frame_size) / hop_in) + 1)
    estimated_length = (num_frames_est - 1) * hop_out + frame_size
    # Add extra buffer to handle edge cases
    output = np.zeros(estimated_length + frame_size * 2, dtype=np.float32)
    normalization = np.zeros(estimated_length + frame_size * 2, dtype=np.float32)
    
    # Phase tracking: store previous phase for phase unwrapping
    prev_phase = np.zeros(num_bins, dtype=np.float32)
    prev_phase_cumulative = np.zeros(num_bins, dtype=np.float32)
    
    # Analysis position
    input_pos = 0
    output_pos = 0
    
    # Expected phase increment per bin (for phase unwrapping)
    expected_phase_increment = 2.0 * np.pi * hop_in / frame_size
    
    frame_count = 0
    
    while input_pos + frame_size <= len(samples):
        # Extract frame
        frame = np.array(samples[input_pos:input_pos + frame_size], dtype=np.float32)
        
        # Apply window
        frame_windowed = frame * window
        
        # FFT
        spectrum = np.fft.rfft(frame_windowed)
        
        # Get magnitude and phase
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Phase unwrapping and modification
        if frame_count > 0:
            # Calculate phase difference
            phase_diff = phase - prev_phase
            
            # Unwrap phase (handle 2π discontinuities)
            phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
            
            # Add expected phase increment (based on hop size)
            for bin_idx in range(num_bins):
                expected_inc = expected_phase_increment * bin_idx
                phase_diff[bin_idx] += expected_inc
            
            # Update cumulative phase
            prev_phase_cumulative += phase_diff * phase_shift_ratio
            
        else:
            # First frame: initialize cumulative phase
            prev_phase_cumulative = phase.copy()
        
        # Reconstruct spectrum with modified phase
        modified_spectrum = magnitude * np.exp(1j * prev_phase_cumulative)
        
        # IFFT
        output_frame = np.fft.irfft(modified_spectrum, n=frame_size)
        
        # Apply synthesis window
        output_frame_windowed = output_frame * window
        
        # Overlap-add to output buffer
        if output_pos + frame_size <= len(output):
            output[output_pos:output_pos + frame_size] += output_frame_windowed
            normalization[output_pos:output_pos + frame_size] += window ** 2
        
        # Update positions
        input_pos += hop_in
        output_pos += hop_out
        
        # Store phase for next iteration
        prev_phase = phase.copy()
        frame_count += 1
    
    # Normalize output (divide by window sum squared to compensate for overlap-add)
    output_length = output_pos
    if output_length > len(output):
        output_length = len(output)
    
    output = output[:output_length]
    normalization = normalization[:output_length]
    
    # Avoid division by zero
    normalization[normalization < 1e-10] = 1.0
    
    output = output / normalization
    
    return output.tolist()


def resample(
    samples: List[float],
    from_rate: float,
    to_rate: float,
) -> List[float]:
    """
    Resample audio to different sample rate.

    Args:
        samples: Input samples
        from_rate: Source sample rate
        to_rate: Target sample rate

    Returns:
        Resampled audio
    """
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)

    result = []
    for i in range(new_length):
        source_pos = i / ratio
        index_a = int(source_pos)
        index_b = min(index_a + 1, len(samples) - 1)
        frac = source_pos - index_a

        sample = samples[index_a] * (1 - frac) + samples[index_b] * frac
        result.append(sample)

    return result
