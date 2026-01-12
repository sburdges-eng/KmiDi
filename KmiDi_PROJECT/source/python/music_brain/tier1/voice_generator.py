"""
Tier 1 Voice Generator: Lightweight speech synthesis without fine-tuning.

Uses pretrained text-to-speech (TTS) models:
  1. pyttsx3: Lightweight, offline, no downloads needed
  2. gTTS: Free, requires internet
  3. TTS 2.0: Modern, higher quality (glow-tts, fastspeech2)

For iDAW: Generate therapeutic guidance text with emotion-driven prosody.

Emotion mapping:
  - GRIEF: Slower, lower pitch, more somber
  - JOY: Faster, higher pitch, brighter
  - CALM: Natural pace, even tone
  - ANGER: Faster, sharp articulation
"""

import numpy as np
from typing import Optional, Dict
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class Tier1VoiceGenerator:
    """
    Tier 1 voice synthesis: Pretrained TTS, no fine-tuning.

    Fallback chain:
      1. Try TTS 2.0 (best quality)
      2. Fall back to pyttsx3 (always available)
    """

    def __init__(
        self,
        device: str = "mps",
        tts_backend: str = "auto",
        verbose: bool = True
    ):
        """
        Initialize Tier 1 voice generator.

        Args:
            device: "mps", "cuda", "cpu"
            tts_backend: "auto", "tts2", "pyttsx3", "gtts"
            verbose: Enable logging
        """
        self.device = device
        self.verbose = verbose
        self.tts_backend = tts_backend
        self.tts = None
        self.backend_used = None

        self._initialize_tts()

    def _initialize_tts(self):
        """Try to load TTS in order of preference"""
        # Try TTS 2.0 first
        if self.tts_backend in ["auto", "tts2"]:
            try:
                from TTS.api import TTS as TTS2
                self.tts = TTS2(
                    model_name="tts_models/en/ljspeech/tacotron2-DDC",
                    gpu=(self.device == "cuda")
                )
                self.backend_used = "TTS2"
                self._log("✓ Using TTS 2.0 (Tacotron2 + vocoder)")
                return
            except ImportError:
                if self.tts_backend == "tts2":
                    self._log("⚠ TTS 2.0 not installed; install with: pip install TTS")

        # Fall back to pyttsx3
        if self.tts_backend in ["auto", "pyttsx3"]:
            try:
                import pyttsx3
                self.tts = pyttsx3.init()
                self._configure_pyttsx3()
                self.backend_used = "pyttsx3"
                self._log("✓ Using pyttsx3 (local, offline)")
                return
            except ImportError:
                self._log("⚠ pyttsx3 not installed; install with: pip install pyttsx3")

        # Fall back to gTTS
        if self.tts_backend in ["auto", "gtts"]:
            try:
                from gtts import gTTS
                self.tts = gTTS
                self.backend_used = "gTTS"
                self._log("✓ Using gTTS (internet-based)")
                return
            except ImportError:
                self._log("⚠ gTTS not installed; install with: pip install gtts")

        self._log("✗ No TTS backend available!")

    def _configure_pyttsx3(self):
        """Configure pyttsx3 for better quality"""
        # Get available voices
        voices = self.tts.getProperty('voices')

        # Use female voice if available
        if len(voices) > 1:
            self.tts.setProperty('voice', voices[1].id)
        else:
            self.tts.setProperty('voice', voices[0].id)

        # Set default rate and volume
        self.tts.setProperty('rate', 150)  # Words per minute
        self.tts.setProperty('volume', 1.0)  # 0.0 to 1.0

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def speak_emotion(
        self,
        text: str,
        emotion: str = "neutral",
        gender: str = "female",
        sample_rate: int = 22050
    ) -> np.ndarray:
        """
        Generate speech with emotion control.

        Args:
            text: Text to speak
            emotion: "grief", "joy", "calm", "anger", "neutral"
            gender: "male", "female"
            sample_rate: Output sample rate

        Returns:
            audio: (num_samples,) float32 waveform
        """
        # Map emotion to prosody
        prosody = self._emotion_to_prosody(emotion)

        if self.backend_used == "TTS2":
            return self._speak_tts2(text, prosody, sample_rate)
        elif self.backend_used == "pyttsx3":
            return self._speak_pyttsx3(text, prosody, sample_rate)
        elif self.backend_used == "gTTS":
            return self._speak_gtts(text, prosody, sample_rate)
        else:
            # Fallback: silence
            self._log("⚠ No TTS available; returning silence")
            return np.zeros(int(sample_rate * 2), dtype=np.float32)

    def _emotion_to_prosody(self, emotion: str) -> Dict[str, float]:
        """
        Map emotion to prosody parameters.

        Returns:
            prosody: Dict with rate, pitch, volume, etc.
        """
        prosody_map = {
            "grief": {
                "rate": 0.7,       # Slower
                "pitch": -50,      # Lower
                "volume": 0.6,     # Quieter
                "pauses": True,    # More pauses
            },
            "joy": {
                "rate": 1.3,       # Faster
                "pitch": 50,       # Higher
                "volume": 1.0,     # Full volume
                "pauses": False,
            },
            "calm": {
                "rate": 1.0,       # Normal
                "pitch": 0,        # Normal pitch
                "volume": 0.8,
                "pauses": True,
            },
            "anger": {
                "rate": 1.5,       # Much faster
                "pitch": 50,       # Raised
                "volume": 1.0,
                "pauses": False,
            },
            "neutral": {
                "rate": 1.0,
                "pitch": 0,
                "volume": 0.8,
                "pauses": False,
            }
        }

        return prosody_map.get(emotion, prosody_map["neutral"])

    def _speak_tts2(
        self,
        text: str,
        prosody: Dict,
        sample_rate: int
    ) -> np.ndarray:
        """TTS 2.0 synthesis"""
        try:
            # TTS 2.0 returns numpy array or audio bytes
            wav = self.tts.tts(text=text, speaker_idx=0)

            # Ensure float32
            if isinstance(wav, list):
                audio = np.array(wav, dtype=np.float32)
            else:
                audio = wav

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio

        except Exception as e:
            self._log(f"⚠ TTS 2.0 error: {e}")
            return np.zeros(int(sample_rate * 2), dtype=np.float32)

    def _speak_pyttsx3(
        self,
        text: str,
        prosody: Dict,
        sample_rate: int
    ) -> np.ndarray:
        """pyttsx3 synthesis"""
        try:
            import scipy.io.wavfile as wavfile

            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            # Configure prosody
            rate = int(prosody.get("rate", 1.0) * 150)  # 150 WPM base
            volume = prosody.get("volume", 0.8)

            self.tts.setProperty('rate', max(50, min(300, rate)))
            self.tts.setProperty('volume', volume)

            # Generate
            self.tts.save_to_file(text, temp_path)
            self.tts.runAndWait()

            # Load audio
            try:
                sr, audio_data = wavfile.read(temp_path)
                audio = audio_data.astype(np.float32) / 32768.0

                # Resample if needed
                if sr != sample_rate:
                    from scipy import signal
                    ratio = sample_rate / sr
                    new_length = int(len(audio) * ratio)
                    audio = signal.resample(audio, new_length)

                # Clean up
                Path(temp_path).unlink(missing_ok=True)

                return audio

            except Exception as e:
                self._log(f"⚠ Failed to load generated WAV: {e}")
                return np.zeros(int(sample_rate * 2), dtype=np.float32)

        except Exception as e:
            self._log(f"⚠ pyttsx3 error: {e}")
            return np.zeros(int(sample_rate * 2), dtype=np.float32)

    def _speak_gtts(
        self,
        text: str,
        prosody: Dict,
        sample_rate: int
    ) -> np.ndarray:
        """gTTS (Google Text-to-Speech) synthesis"""
        try:
            from gtts import gTTS
            import scipy.io.wavfile as wavfile

            # gTTS doesn't support prosody directly; generate and return
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name

            tts = gTTS(text=text, lang='en', slow=prosody.get("rate", 1.0) < 1.0)
            tts.save(temp_path)

            # Convert MP3 to WAV (requires ffmpeg)
            try:
                import subprocess
                wav_path = temp_path.replace(".mp3", ".wav")
                subprocess.run([
                    "ffmpeg", "-i", temp_path, "-acodec", "pcm_s16le",
                    "-ar", str(sample_rate), wav_path
                ], capture_output=True)

                sr, audio_data = wavfile.read(wav_path)
                audio = audio_data.astype(np.float32) / 32768.0

                # Clean up
                Path(temp_path).unlink(missing_ok=True)
                Path(wav_path).unlink(missing_ok=True)

                return audio

            except Exception:
                self._log("⚠ gTTS requires ffmpeg for WAV conversion")
                return np.zeros(int(sample_rate * 2), dtype=np.float32)

        except Exception as e:
            self._log(f"⚠ gTTS error: {e}")
            return np.zeros(int(sample_rate * 2), dtype=np.float32)

    def generate_therapeutic_response(
        self,
        emotion_intent: str,
        wound: str
    ) -> str:
        """
        Generate therapeutic guidance text based on emotion + wound.

        Args:
            emotion_intent: "GRIEF", "JOY", "CALM", etc.
            wound: Description of core wound (e.g., "I feel lost")

        Returns:
            response_text: Therapeutic guidance
        """
        responses = {
            "GRIEF": (
                "Your grief is valid. The pain you feel is a testament to what you've loved. "
                "This music honors what you've lost, and the depth of your feeling."
            ),
            "JOY": (
                "Your joy deserves to be celebrated and amplified. "
                "Let this music be a voice for the happiness and hope you feel."
            ),
            "CALM": (
                "You are safe in this moment. Let this music wrap around you like a blanket. "
                "Your breath can follow its pace. You are exactly where you need to be."
            ),
            "ANGER": (
                "Your anger has power. It can be a force for change and truth. "
                "Channel it, don't suppress it. This music gives your anger a voice."
            ),
            "ANXIETY": (
                "What you're feeling is real, and it will pass. "
                "Let this music anchor you to the present moment, to your breath, to now."
            ),
            "HOPE": (
                "Your hope is a strength. Even in darkness, you're reaching toward light. "
                "This music believes in your tomorrow."
            )
        }

        return responses.get(emotion_intent, responses["CALM"])


# Convenience function
def generate_tier1_voice(
    text: str,
    emotion: str = "neutral",
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Quick wrapper: Generate voice in one line.

    Example:
        text = "This is how you're feeling right now."
        audio = generate_tier1_voice(text, emotion="calm")
    """
    gen = Tier1VoiceGenerator(verbose=False)
    return gen.speak_emotion(text, emotion=emotion, sample_rate=sample_rate)
