"""
Phoneme Aligner Integration Example

This file contains example integration code for loading a Q4 quantized model.
Copy the relevant section into phoneme_aligner.py to integrate.

See docs/PHONEME_ALIGNER_INTEGRATION.md for detailed instructions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .utils.memory_monitor import MemoryMonitor
from .utils.external_ssd import ExternalSSDManager
from .voice_profile import VoiceProfile, PhonemeType

logger = logging.getLogger(__name__)


# =============================================================================
# OPTION 1: llama-cpp-python Integration
# =============================================================================


class PhonemeAlignerLlamaCpp:
    """Phoneme aligner using llama-cpp-python"""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize phoneme aligner"""
        self.ssd_manager = ExternalSSDManager()
        self.memory_monitor = MemoryMonitor()
        self.model = None
        self.model_path = model_path or self.ssd_manager.get_model_path("phoneme_aligner_q4.bin")

    def load_model(self) -> bool:
        """Load ML model (3B parameter, Q4 quantization)"""
        try:
            logger.info(f"Loading phoneme alignment model from {self.model_path}")

            # Check memory before loading
            within_limit, warning = self.memory_monitor.check_memory_limit()
            if not within_limit:
                raise RuntimeError(f"Cannot load model: {warning}")

            # Check if model file exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    "See docs/PHONEME_ALIGNER_INTEGRATION.md for setup instructions"
                )

            # Load using llama-cpp-python
            try:
                from llama_cpp import Llama

                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=4096,  # Context window
                    n_threads=4,  # CPU threads
                    verbose=False,
                    n_gpu_layers=0,  # Set to >0 for GPU support
                )

                logger.info("Model loaded successfully")
                return True

            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed. Install with: " "pip install llama-cpp-python"
                )

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def align_phonemes(
        self, audio_samples: np.ndarray, sample_rate: int, transcript: Optional[str] = None
    ) -> List[Tuple[PhonemeType, float, float]]:
        """
        Align phonemes to audio using ML model

        Note: This is a template. Actual implementation depends on:
        - Model input format (audio features, spectrogram, etc.)
        - Model output format (phoneme boundaries, probabilities, etc.)
        - Phoneme conversion logic
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Aligning phonemes: {len(audio_samples)} samples at {sample_rate}Hz")

        # TODO: Implement actual alignment
        # 1. Extract audio features (MFCC, spectrogram, etc.)
        # 2. Convert transcript to phonemes (if provided)
        # 3. Prepare model input
        # 4. Run model inference
        # 5. Parse model output to get phoneme boundaries
        # 6. Return list of (phoneme_type, start_time_ms, end_time_ms)

        # Placeholder implementation
        aligned_phonemes = []

        return aligned_phonemes


# =============================================================================
# OPTION 2: Alternative - Montreal Forced Aligner (MFA)
# =============================================================================


class PhonemeAlignerMFA:
    """Phoneme aligner using Montreal Forced Aligner"""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize with MFA"""
        self.model_path = model_path
        # MFA doesn't require a model file in the same way
        # It uses pre-trained acoustic models

    def load_model(self) -> bool:
        """Load MFA (if needed)"""
        try:
            import montreal_forced_alignment as mfa

            logger.info("MFA available")
            return True
        except ImportError:
            logger.error(
                "Montreal Forced Aligner not installed. Install with: "
                "conda install -c conda-forge montreal-forced-alignment"
            )
            return False

    def align_phonemes(
        self, audio_samples: np.ndarray, sample_rate: int, transcript: Optional[str] = None
    ) -> List[Tuple[PhonemeType, float, float]]:
        """Align using MFA"""
        # MFA requires audio file and text file
        # Implementation would involve:
        # 1. Save audio to temporary file
        # 2. Create text file with transcript
        # 3. Run MFA alignment
        # 4. Parse output TextGrid file
        # 5. Convert to phoneme boundaries

        aligned_phonemes = []
        return aligned_phonemes


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
To integrate into phoneme_aligner.py:

1. Choose integration method:
   - llama-cpp-python (for 3B Q4 quantized model)
   - MFA (alternative, doesn't require 3B model)

2. For llama-cpp-python:
   - Install: pip install llama-cpp-python
   - Replace load_model() method with PhonemeAlignerLlamaCpp.load_model()
   - Implement align_phonemes() based on your model's input/output format
   - Place model file at: python/prrot/models/phoneme_aligner_q4.bin

3. For MFA:
   - Install: conda install -c conda-forge montreal-forced-alignment
   - Replace with PhonemeAlignerMFA implementation
   - Note: MFA works differently and may not require the 3B model

4. Update align_phonemes() implementation:
   - Extract audio features appropriate for your model
   - Convert transcript to phonemes
   - Run model inference
   - Parse output to get boundaries
   - Return list of (phoneme_type, start_ms, end_ms)

5. Test integration:
   - Run: python scripts/test_models.py
   - Or manually test with sample audio

See docs/PHONEME_ALIGNER_INTEGRATION.md for complete instructions.
"""
