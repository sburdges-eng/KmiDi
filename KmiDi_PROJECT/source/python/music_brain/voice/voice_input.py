"""
Voice Input - Recording and Voice Mimicking

Records user voice and enables mimicking/cloning capabilities.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
import soundfile as sf

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Voice recording disabled.")


class VoiceRecorder:
    """
    Records audio from microphone.
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        """
        Initialize voice recorder.

        Args:
            sample_rate: Sample rate for recording
            channels: Number of channels (1=mono, 2=stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.available = SOUNDDEVICE_AVAILABLE

    def record(self, duration_seconds: float) -> Optional[np.ndarray]:
        """
        Record audio from microphone.

        Args:
            duration_seconds: Duration to record

        Returns:
            Audio array or None if recording failed
        """
        if not self.available:
            print("Recording not available (sounddevice not installed)")
            return None

        try:
            print(f"Recording for {duration_seconds} seconds...")
            audio = sd.rec(
                int(duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            print("Recording complete")

            # Convert to mono if stereo
            if self.channels == 2 and len(audio.shape) == 2:
                audio = np.mean(audio, axis=1)

            return audio
        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def record_until_silence(
        self,
        silence_threshold: float = 0.01,
        max_duration: float = 10.0
    ) -> Optional[np.ndarray]:
        """
        Record until silence is detected.

        Args:
            silence_threshold: RMS threshold for silence
            max_duration: Maximum recording duration

        Returns:
            Audio array or None
        """
        if not self.available:
            return None

        chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
        audio_chunks = []
        silence_count = 0
        max_silence_chunks = 10  # 1 second of silence

        print("Recording (speak now, will stop after silence)...")

        try:
            for _ in range(int(max_duration * 10)):  # Check every 100ms
                chunk = sd.rec(chunk_size, samplerate=self.sample_rate, channels=self.channels, dtype='float32')
                sd.wait()

                if self.channels == 2 and len(chunk.shape) == 2:
                    chunk = np.mean(chunk, axis=1)

                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > silence_threshold:
                    audio_chunks.append(chunk)
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= max_silence_chunks:
                        break

            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                print(f"Recording complete: {len(audio) / self.sample_rate:.2f} seconds")
                return audio
            else:
                print("No audio recorded")
                return None
        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def save_recording(self, audio: np.ndarray, output_path: str) -> bool:
        """
        Save recording to file.

        Args:
            audio: Audio array
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            sf.write(output_path, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False


class VoiceMimic:
    """
    Mimics user voice by extracting voice characteristics.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize voice mimic.

        Args:
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate

    def extract_voice_characteristics(self, audio: np.ndarray) -> Dict:
        """
        Extract voice characteristics from audio.

        Args:
            audio: Audio signal

        Returns:
            Dictionary of voice characteristics
        """
        characteristics = {}

        # Extract pitch statistics
        from music_brain.voice.pitch_controller import PitchController
        pitch_controller = PitchController(self.sample_rate)

        frequencies, _ = pitch_controller.extract_pitch_from_audio(audio, self.sample_rate)
        frequencies = frequencies[frequencies > 0]

        if len(frequencies) > 0:
            characteristics["mean_pitch"] = float(np.mean(frequencies))
            characteristics["pitch_range"] = float(np.max(frequencies) - np.min(frequencies))
            characteristics["pitch_variance"] = float(np.var(frequencies))
        else:
            characteristics["mean_pitch"] = 220.0
            characteristics["pitch_range"] = 200.0
            characteristics["pitch_variance"] = 0.0

        # Extract spectral characteristics
        try:
            import librosa
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            characteristics["brightness"] = float(np.mean(spectral_centroids))

            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            characteristics["rolloff"] = float(np.mean(spectral_rolloff))

            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            characteristics["zcr"] = float(np.mean(zcr))
        except ImportError:
            # Fallback: simple spectral analysis
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)

            # Spectral centroid
            if np.sum(magnitude) > 0:
                characteristics["brightness"] = float(np.sum(freqs * magnitude) / np.sum(magnitude))
            else:
                characteristics["brightness"] = 2000.0

            characteristics["rolloff"] = 5000.0
            characteristics["zcr"] = 0.1

        # Extract formant-like characteristics (simplified)
        characteristics["formant_emphasis"] = 0.5  # Default
        characteristics["breathiness"] = min(0.5, characteristics.get("zcr", 0.1) * 5)

        return characteristics

    def apply_voice_characteristics(
        self,
        base_audio: np.ndarray,
        characteristics: Dict
    ) -> np.ndarray:
        """
        Apply voice characteristics to audio (voice conversion).

        Args:
            base_audio: Base audio to modify
            characteristics: Voice characteristics to apply

        Returns:
            Modified audio
        """
        # Adjust pitch (simple pitch shifting)
        target_pitch = characteristics.get("mean_pitch", 220.0)

        # Estimate current pitch
        pitch_controller = PitchController(self.sample_rate)
        frequencies, _ = pitch_controller.extract_pitch_from_audio(base_audio, self.sample_rate)
        current_pitch = np.mean(frequencies[frequencies > 0]) if len(frequencies[frequencies > 0]) > 0 else 220.0

        # Pitch shift
        if current_pitch > 0:
            pitch_ratio = target_pitch / current_pitch
            if abs(pitch_ratio - 1.0) > 0.01:  # Only shift if significant difference
                try:
                    import librosa
                    base_audio = librosa.effects.pitch_shift(
                        base_audio,
                        sr=self.sample_rate,
                        n_steps=12 * np.log2(pitch_ratio)
                    )
                except ImportError:
                    # Simple resampling-based pitch shift (crude)
                    if pitch_ratio != 1.0:
                        from scipy import signal
                        new_length = int(len(base_audio) / pitch_ratio)
                        base_audio = signal.resample(base_audio, new_length)

        # Adjust brightness (spectral tilt)
        brightness = characteristics.get("brightness", 2000.0)
        try:
            import librosa
            # Apply high-frequency emphasis/de-emphasis
            fft = np.fft.rfft(base_audio)
            freqs = np.fft.rfftfreq(len(base_audio), 1 / self.sample_rate)

            # Create brightness filter
            target_brightness = 2000.0  # Reference
            if brightness != target_brightness:
                tilt = (brightness - target_brightness) / target_brightness
                filter_gain = 1.0 + tilt * (freqs / target_brightness)
                filter_gain = np.clip(filter_gain, 0.1, 2.0)
                fft = fft * filter_gain

            base_audio = np.fft.irfft(fft, n=len(base_audio))
        except Exception:
            pass  # Skip if FFT fails

        # Add breathiness
        breathiness = characteristics.get("breathiness", 0.2)
        if breathiness > 0:
            noise = np.random.randn(len(base_audio)) * breathiness * 0.1
            base_audio = base_audio + noise

        return base_audio
