"""Core music generation engine.

Pure business logic - no GUI dependencies.
Can run headless (CLI, tests).
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .models import EmotionIntent, GenerationResult

logger = logging.getLogger(__name__)


class MusicEngine:
    """Core music generation engine.
    
    This class contains all business logic for music generation.
    It has no knowledge of GUI components and can run headless.
    """
    
    def __init__(
        self, music_brain_api_url: str = "http://127.0.0.1:8000"
    ):
        """Initialize engine.

        Args:
            music_brain_api_url: URL of Music Brain API server
        """
        self.api_url = music_brain_api_url
        self.logger = logger
    
    def generate_music(self, intent: EmotionIntent) -> GenerationResult:
        """Generate music from emotional intent.
        
        Args:
            intent: Emotional intent specification
            
        Returns:
            GenerationResult with success status and MIDI path
        """
        try:
            self.logger.info(f"Generating music for intent: {intent.mood_primary}")
            
            # Call Music Brain API
            result = self._call_music_brain_api(intent)
            
            # Parse API response
            if result.get("status") == "success":
                result_data = result.get("result", {})
                
                # Extract chord progression if available
                chords = result_data.get("chords", [])
                if isinstance(chords, str):
                    chords = chords.split(", ")
                
                # Extract metadata
                metadata = result_data.get("metadata", {})
                
                return GenerationResult(
                    success=True,
                    midi_path=result_data.get("midi_path"),
                    chords=chords or ["F", "C", "Am", "Dm"],
                    key=metadata.get("key", "F major"),
                    tempo=metadata.get("tempo", 82),
                    metadata={"intent": intent.to_dict(), **metadata}
                )
            else:
                error_msg = result.get("detail", "Unknown error")
                return GenerationResult(
                    success=False,
                    error=error_msg
                )
        except Exception as e:
            self.logger.error(f"Music generation failed: {e}")
            return GenerationResult(
                success=False,
                error=str(e)
            )
    
    def analyze_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze audio file (onsets, LUFS, similarity).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info(f"Analyzing audio: {audio_path}")
        
        try:
            # Use penta_core for audio analysis if available
            try:
                import numpy as np
                import soundfile as sf
                from penta_core.dsp.parrot_dsp import detect_pitch
                
                # Load audio
                audio, sample_rate = sf.read(str(audio_path))
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Detect pitch
                pitch = detect_pitch(audio.tolist(), sample_rate=sample_rate)
                
                # Basic analysis
                rms = np.sqrt(np.mean(audio ** 2))
                
                return {
                    "onsets": [],  # Would use OnsetDetector for full analysis
                    "lufs": None,  # Would use loudness meter
                    "similarity": None,
                    "pitch": pitch,
                    "rms": float(rms),
                    "sample_rate": int(sample_rate),
                    "duration": float(len(audio) / sample_rate),
                }
            except ImportError:
                self.logger.warning("Audio analysis libraries not available")
                return {
                    "onsets": [],
                    "lufs": None,
                    "similarity": None,
                    "error": "Audio analysis libraries not installed",
                }
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {
                "onsets": [],
                "lufs": None,
                "similarity": None,
                "error": str(e),
            }
    
    def _call_music_brain_api(
        self, intent: EmotionIntent
    ) -> Dict[str, Any]:
        """Call Music Brain API.

        Args:
            intent: Emotional intent

        Returns:
            API response dictionary
        """
        try:
            import requests
            
            # Map intent to API request format
            api_request = {
                "intent": {
                    "emotional_intent": intent.mood_primary or intent.core_event or "unknown",
                    "core_wound": intent.core_event,
                    "core_desire": intent.core_longing,
                    "technical": {
                        "key": intent.technical_key,
                        "bpm": intent.technical_bpm,
                        "genre": intent.technical_genre,
                    } if any([intent.technical_key, intent.technical_bpm, intent.technical_genre]) else None,
                },
                "output_format": "midi",
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=api_request,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except ImportError:
            self.logger.warning("requests library not available, using local API")
            # Fallback to local music_brain API
            try:
                from music_brain.api import api as music_api
                
                # Use therapy_session directly
                result = music_api.therapy_session(
                    text=intent.mood_primary or intent.core_event or "grief",
                    motivation=7,
                    chaos_tolerance=0.5,
                    output_midi=None,
                )
                return {"status": "success", "result": result}
            except ImportError:
                self.logger.error("Music brain API not available")
                raise
        except requests.RequestException as e:
            self.logger.error(f"API call failed: {e}")
            raise


# Global engine instance (can be replaced for testing)
_engine: Optional[MusicEngine] = None


def get_engine() -> MusicEngine:
    """Get global engine instance."""
    global _engine
    if _engine is None:
        _engine = MusicEngine()
    return _engine


def set_engine(engine: MusicEngine) -> None:
    """Set global engine instance (for testing)."""
    global _engine
    _engine = engine

