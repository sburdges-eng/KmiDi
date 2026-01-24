"""
Timbre Embeddings Integration Example

This file contains example integration code for Wav2Vec2 and Whisper.
Copy the relevant section into timbre_embeddings.py to integrate.

See docs/TIMBRE_EXTRACTOR_INTEGRATION.md for detailed instructions.
"""

import torch
import torchaudio
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# OPTION 1: Wav2Vec2 Integration
# =============================================================================


class TimbreEmbeddingExtractorWav2Vec2:
    """Wav2Vec2-based timbre embedding extractor"""

    def __init__(self, embedding_dim: int = 256, model_name: str = "facebook/wav2vec2-base-960h"):
        """Initialize with Wav2Vec2 model"""
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """Lazy load model"""
        if self.model is None:
            try:
                from transformers import Wav2Vec2Model, Wav2Vec2Processor

                logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)
                self.model.eval()
                self.model.to(self.device)
                logger.info("Wav2Vec2 model loaded successfully")
            except ImportError:
                raise ImportError(
                    "transformers and torchaudio required. Install with: "
                    "pip install transformers torchaudio"
                )

    def extract_embedding(self, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract timbre embedding"""
        self._load_model()

        # Convert to tensor
        if isinstance(audio_samples, np.ndarray):
            audio_tensor = torch.from_numpy(audio_samples).float()
        else:
            audio_tensor = audio_samples.float()

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)

        # Process audio
        inputs = self.processor(
            audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Pool to fixed size (mean pooling)
        embedding = features.mean(dim=1).squeeze().cpu().numpy()

        # Project to desired dimension if needed
        if embedding.shape[0] != self.embedding_dim:
            if embedding.shape[0] > self.embedding_dim:
                embedding = embedding[: self.embedding_dim]
            else:
                padding = np.zeros(self.embedding_dim - embedding.shape[0])
                embedding = np.concatenate([embedding, padding])

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)


# =============================================================================
# OPTION 2: Whisper Integration
# =============================================================================


class TimbreEmbeddingExtractorWhisper:
    """Whisper-based timbre embedding extractor"""

    def __init__(self, embedding_dim: int = 256, model_size: str = "base"):
        """Initialize with Whisper model"""
        self.embedding_dim = embedding_dim
        self.model_size = model_size
        self.model = None

    def _load_model(self):
        """Lazy load model"""
        if self.model is None:
            try:
                import whisper

                logger.info(f"Loading Whisper model: {self.model_size}")
                self.model = whisper.load_model(self.model_size)
                logger.info("Whisper model loaded successfully")
            except ImportError:
                raise ImportError(
                    "openai-whisper required. Install with: pip install openai-whisper"
                )

    def extract_embedding(self, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract timbre embedding"""
        self._load_model()

        # Whisper expects 16kHz
        if sample_rate != 16000:
            import librosa

            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)

        audio = audio_samples.astype(np.float32)

        # Extract mel spectrogram
        mel = whisper.audio.log_mel_spectrogram(audio).to(self.model.device)

        # Extract encoder features
        with torch.no_grad():
            features = self.model.encoder(mel)  # [1, n_frames, hidden_dim]

        # Pool to fixed size
        embedding = features.mean(dim=1).squeeze().cpu().numpy()

        # Project to desired dimension if needed
        if embedding.shape[0] != self.embedding_dim:
            if embedding.shape[0] > self.embedding_dim:
                embedding = embedding[: self.embedding_dim]
            else:
                padding = np.zeros(self.embedding_dim - embedding.shape[0])
                embedding = np.concatenate([embedding, padding])

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
To integrate into timbre_embeddings.py:

1. Choose Wav2Vec2 or Whisper (Wav2Vec2 recommended)

2. For Wav2Vec2:
   - Install: pip install transformers torchaudio
   - Replace TimbreEmbeddingExtractor class with TimbreEmbeddingExtractorWav2Vec2
   - Or modify existing class to use Wav2Vec2 internally

3. For Whisper:
   - Install: pip install openai-whisper
   - Replace TimbreEmbeddingExtractor class with TimbreEmbeddingExtractorWhisper
   - Or modify existing class to use Whisper internally

4. Update __init__ to accept model selection:
   def __init__(self, embedding_dim: int = 256, model_type: str = "wav2vec2"):
       if model_type == "wav2vec2":
           self._extractor = TimbreEmbeddingExtractorWav2Vec2(embedding_dim)
       elif model_type == "whisper":
           self._extractor = TimbreEmbeddingExtractorWhisper(embedding_dim)
       else:
           raise ValueError(f"Unknown model_type: {model_type}")

5. Delegate methods to extractor:
   def extract_embedding(self, audio_samples, sample_rate):
       return self._extractor.extract_embedding(audio_samples, sample_rate)

See docs/TIMBRE_EXTRACTOR_INTEGRATION.md for complete instructions.
"""
