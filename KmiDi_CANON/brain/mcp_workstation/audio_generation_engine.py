"""
Audio generation engine for orchestrator.

Provides AudioGenerationEngine: optional Audiocraft MusicGen for texture generation.
When not installed or model fails to load, returns status "stubbed" so orchestrator continues.
Contract satisfied: generate_audio_texture() returns dict with status in {stubbed, completed, failed, timeout},
details, and when completed: output_path, sample_rate, optionally audio_data_base64.

Deps (optional): audiocraft, torch. Install when audio gen needed.
Env: KMI_DI_AUDIO_MODEL_ID or AUDIOCRAFT_MODEL_ID — hub id (e.g. musicgen-small). Default: musicgen-small.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import threading
import time

try:
    from audiocraft.models import MusicGen  # type: ignore[import-not-found]
    from audiocraft.data.audio import (  # type: ignore[import-not-found]
        audio_write,
    )

    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    AUDIOCRAFT_AVAILABLE = False
    MusicGen = None
    audio_write = None
    logging.getLogger(__name__).warning(
        "audiocraft not found. Audio generation will be stubbed."
    )


def _log():
    return logging.getLogger(__name__)


class AudioGenerationEngine:
    def __init__(
        self,
        model_id: Optional[str] = None,
        output_dir: str = "./audio_output",
    ):
        import os
        self.model_id = model_id or os.environ.get(
            "KMI_DI_AUDIO_MODEL_ID",
            os.environ.get("AUDIOCRAFT_MODEL_ID", "musicgen-small"),
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.lock = threading.Lock()  # For mutual exclusion with LLM

    def _load_model(self) -> None:
        """Load audio model (MusicGen). No-op if audiocraft unavailable or load fails."""
        if not AUDIOCRAFT_AVAILABLE:
            _log().info("Audiocraft not available. Cannot load audio model.")
            return
        if self.model is not None:
            return
        _log().info("Loading audio model: %s", self.model_id)
        try:
            self.model = MusicGen.get_pretrained(self.model_id)
            _log().info("Audio model loaded.")
        except Exception as e:
            _log().warning("Audio model load failed: %s", e)
            self.model = None
            raise

    def generate_audio_texture(
        self,
        prompt: str,
        duration: int = 10,
        temperature: float = 1.0,
        assume_locked: bool = False,
        lock_timeout: Optional[float] = 300.0,
    ) -> Dict[str, Any]:
        """Generates an audio texture based on the prompt.

        Args:
            prompt: Text description for audio generation.
            duration: Target duration in seconds.
            temperature: Generation temperature.
            assume_locked: If True, skip lock acquisition (caller holds lock).
            lock_timeout: Timeout in seconds for lock acquisition.
                None = block forever, 0 = non-blocking.
        """
        if AUDIOCRAFT_AVAILABLE and self.model is None:
            try:
                self._load_model()
            except Exception as e:
                _log().warning("Audio model load failed; stub: %s", e)
                self.model = None

        if not AUDIOCRAFT_AVAILABLE or self.model is None:
            _log().info(
                "Audio generation not available or model not loaded. Returning stubbed result."
            )
            return {
                "status": "stubbed",
                "prompt": prompt,
                "audio_data_base64": "<placeholder_audio_data>",
                "details": (
                    "Audiocraft library not installed or model "
                    "failed to load. Returning placeholder audio."
                ),
            }

        def _generate() -> Dict[str, Any]:
            _log().info("Generating audio texture: %s", prompt[:80])
            try:
                import base64
                import io

                # Generate audio using MusicGen model
                self.model.set_generation_params(duration=duration)
                wav = self.model.generate([prompt])  # Returns tensor [batch, channels, samples]

                # Convert to audio file and encode as base64
                prompt_safe = prompt.replace(" ", "_")[:50]
                output_path = self.output_dir / f"audio_{prompt_safe}.wav"

                # Use audiocraft's audio_write to save the file
                audio_write(
                    str(output_path.with_suffix('')),  # audio_write adds extension
                    wav[0].cpu(),  # First sample in batch
                    self.model.sample_rate,
                    strategy="loudness",
                )

                # Read the saved file and encode to base64
                with open(str(output_path), "rb") as f:
                    audio_bytes = f.read()
                audio_data_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                return {
                    "status": "completed",
                    "prompt": prompt,
                    "audio_data_base64": audio_data_base64,
                    "output_path": str(output_path),
                    "sample_rate": self.model.sample_rate,
                    "details": "Audio texture generated successfully.",
                }
            except Exception as e:
                _log().exception("Audio generation failed: %s", e)
                return {
                    "status": "failed",
                    "prompt": prompt,
                    "audio_data_base64": "<error_audio_data>",
                    "details": f"Audio generation failed: {e}",
                }

        if assume_locked:
            return _generate()

        # Use timeout-based lock acquisition to avoid indefinite blocking
        # Note: lock_timeout=0 means non-blocking (immediate return)
        #       lock_timeout=None means block forever
        #       lock_timeout>0 means wait up to that many seconds
        # CRITICAL: Must use "is not None" not "if lock_timeout" to correctly
        # distinguish between None (block forever) and 0 (non-blocking attempt)
        if lock_timeout is not None:
            # Pass timeout=0 for non-blocking, or positive value for timed wait
            acquired = self.lock.acquire(timeout=lock_timeout)
        else:
            # No timeout specified - block until lock is acquired
            acquired = self.lock.acquire()
        if not acquired:
            timeout_msg = (
                f"Could not acquire audio lock within {lock_timeout}s timeout."
                if lock_timeout is not None
                else "Could not acquire audio lock."
            )
            timeout_display = lock_timeout if lock_timeout is not None else "N/A"
            _log().warning("Audio generation lock timeout after %s", timeout_display)
            return {
                "status": "timeout",
                "prompt": prompt,
                "audio_data_base64": "<timeout_audio_data>",
                "details": timeout_msg,
            }
        try:
            return _generate()
        finally:
            self.lock.release()

    def acquire_lock(self, timeout: Optional[float] = None) -> bool:
        """Acquires the lock for audio generation.

        Returns True if successful, False otherwise.
        """
        if timeout is None:
            return self.lock.acquire()
        return self.lock.acquire(timeout=timeout)

    def release_lock(self):
        """Releases the lock for audio generation."""
        try:
            self.lock.release()
        except RuntimeError:
            _log().warning("Attempted to release an unheld audio generation lock.")


# Example usage (for testing)
if __name__ == "__main__":
    log = _log()
    log.info("Initializing Audio Generation Engine...")
    engine = AudioGenerationEngine()
    # For real use, you'd call engine._load_model() if needed.

    # Test generation (if audiocraft is available)
    if AUDIOCRAFT_AVAILABLE:
        try:
            engine._load_model()
            log.info("Attempting to generate audio...")
            result = engine.generate_audio_texture(
                prompt="a subtle, evolving drone with metallic overtones"
            )
            log.info("Audio Generation Result: %s %s", result["status"], result["details"])
            if result["status"] == "completed":
                log.info("Simulated audio data generated.")
        except Exception as e:
            log.exception("Caught error during example usage: %s", e)
    else:
        log.info("Skipping audio generation example: audiocraft not available.")

    # Test lock mechanism
    log.info("Testing lock mechanism...")
    if engine.acquire_lock(timeout=1):
        log.info("Lock acquired.")
        time.sleep(2)  # Simulate work
        engine.release_lock()
        log.info("Lock released.")
    else:
        log.warning("Could not acquire lock.")

    # Another attempt (should acquire quickly now)
    if engine.acquire_lock(timeout=1):
        log.info("Lock acquired again.")
        engine.release_lock()
        log.info("Lock released again.")
