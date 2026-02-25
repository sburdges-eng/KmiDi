# Timbre Embedding Extractor Integration Guide

## Overview

The Timbre Embedding Extractor requires a pre-trained encoder model (Wav2Vec2 or Whisper) to extract non-reconstructive timbre embeddings from audio. This guide explains how to integrate it.

## Requirements

- **Model Type:** Pre-trained audio encoder
- **Options:** Wav2Vec2, Whisper, or similar
- **Purpose:** Extract timbre embeddings for voice cloning/synthesis
- **Output:** Fixed-size embedding vector (non-reconstructive)
- **Location:** `python/prrot/timbre_embeddings.py`

## Current Status

**File:** `python/prrot/timbre_embeddings.py`
**Status:** Returns random embeddings (placeholder)
**Extraction:** Lines 23-51 (placeholder)
**Phoneme Embeddings:** Lines 53-81 (placeholder)

## Integration Options

### Option 1: Wav2Vec2 (Recommended)

Wav2Vec2 is a self-supervised learning model that learns powerful audio representations.

#### Step 1: Install Dependencies

```bash
pip install transformers torchaudio
```

#### Step 2: Update Implementation

Update `python/prrot/timbre_embeddings.py`:

```python
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TimbreEmbeddingExtractor:
    """Extract non-reconstructive timbre embeddings using Wav2Vec2"""

    def __init__(self, embedding_dim: int = 256, model_name: str = "facebook/wav2vec2-base-960h"):
        """Initialize timbre embedding extractor"""
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """Lazy load model"""
        if self.model is None:
            logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            logger.info("Model loaded successfully")

    def extract_embedding(self, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract timbre embedding from audio using Wav2Vec2

        Args:
            audio_samples: Audio samples (mono)
            sample_rate: Sample rate in Hz

        Returns:
            Timbre embedding vector (non-reconstructive, normalized)
        """
        self._load_model()

        logger.info(f"Extracting timbre embedding: {len(audio_samples)} samples at {sample_rate}Hz")

        # Convert to tensor
        if isinstance(audio_samples, np.ndarray):
            audio_tensor = torch.from_numpy(audio_samples).float()
        else:
            audio_tensor = audio_samples.float()

        # Resample to 16kHz if needed (Wav2Vec2 requirement)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)

        # Process audio
        inputs = self.processor(
            audio_tensor,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state
            features = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]

        # Pool to fixed size (mean pooling)
        embedding = features.mean(dim=1).squeeze().cpu().numpy()  # Shape: [hidden_dim]

        # Project to desired embedding dimension if needed
        if embedding.shape[0] != self.embedding_dim:
            # Use PCA or linear projection
            # For simplicity, just take first N dimensions or use interpolation
            if embedding.shape[0] > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                # Pad with zeros (or use learned projection)
                padding = np.zeros(self.embedding_dim - embedding.shape[0])
                embedding = np.concatenate([embedding, padding])

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)

    def extract_phoneme_embeddings(
        self,
        audio_samples: np.ndarray,
        sample_rate: int,
        phoneme_boundaries: List[tuple],  # (start_ms, end_ms)
    ) -> List[np.ndarray]:
        """
        Extract timbre embeddings for each phoneme segment

        Args:
            audio_samples: Audio samples
            sample_rate: Sample rate
            phoneme_boundaries: Phoneme boundaries in milliseconds

        Returns:
            List of timbre embeddings (one per phoneme)
        """
        embeddings = []

        for start_ms, end_ms in phoneme_boundaries:
            start_sample = int((start_ms / 1000.0) * sample_rate)
            end_sample = int((end_ms / 1000.0) * sample_rate)

            if start_sample < len(audio_samples) and end_sample <= len(audio_samples):
                segment = audio_samples[start_sample:end_sample]
                embedding = self.extract_embedding(segment, sample_rate)
                embeddings.append(embedding)

        return embeddings

    def aggregate_embeddings(
        self, embeddings: List[np.ndarray], method: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate multiple embeddings into single voice profile embedding

        Args:
            embeddings: List of phoneme embeddings
            method: Aggregation method ("mean", "max", "weighted")

        Returns:
            Aggregated embedding
        """
        if not embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings_array = np.array(embeddings)

        if method == "mean":
            aggregated = embeddings_array.mean(axis=0)
        elif method == "max":
            aggregated = embeddings_array.max(axis=0)
        elif method == "weighted":
            # Weight by phoneme duration or importance
            weights = np.ones(len(embeddings)) / len(embeddings)
            aggregated = np.average(embeddings_array, axis=0, weights=weights)
        else:
            aggregated = embeddings_array.mean(axis=0)

        # Normalize
        aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)

        return aggregated.astype(np.float32)
```

#### Step 3: Test Integration

```python
from python.prrot.timbre_embeddings import TimbreEmbeddingExtractor
import numpy as np

# Initialize
extractor = TimbreEmbeddingExtractor(embedding_dim=256)

# Test extraction
audio = np.random.randn(16000)  # 1 second at 16kHz
embedding = extractor.extract_embedding(audio, 16000)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding norm: {np.linalg.norm(embedding)}")
```

### Option 2: Whisper Encoder

Whisper's encoder can also be used for timbre extraction.

#### Step 1: Install Dependencies

```bash
pip install openai-whisper
```

#### Step 2: Update Implementation

```python
import whisper
import torch
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TimbreEmbeddingExtractor:
    """Extract timbre embeddings using Whisper encoder"""

    def __init__(self, embedding_dim: int = 256, model_size: str = "base"):
        """Initialize timbre embedding extractor"""
        self.embedding_dim = embedding_dim
        self.model_size = model_size
        self.model = None

    def _load_model(self):
        """Lazy load model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Model loaded successfully")

    def extract_embedding(self, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract timbre embedding from audio using Whisper encoder
        """
        self._load_model()

        logger.info(f"Extracting timbre embedding: {len(audio_samples)} samples at {sample_rate}Hz")

        # Whisper expects 16kHz audio
        if sample_rate != 16000:
            import librosa
            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)

        # Load audio (Whisper handles loading)
        # For numpy array, convert to expected format
        audio = audio_samples.astype(np.float32)

        # Extract mel spectrogram
        mel = whisper.audio.log_mel_spectrogram(audio).to(self.model.device)

        # Extract encoder features
        with torch.no_grad():
            features = self.model.encoder(mel)  # Shape: [1, n_frames, hidden_dim]

        # Pool to fixed size (mean pooling)
        embedding = features.mean(dim=1).squeeze().cpu().numpy()  # Shape: [hidden_dim]

        # Project to desired dimension if needed
        if embedding.shape[0] != self.embedding_dim:
            if embedding.shape[0] > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                padding = np.zeros(self.embedding_dim - embedding.shape[0])
                embedding = np.concatenate([embedding, padding])

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)

    # ... rest of methods same as Wav2Vec2 version
```

### Option 3: Custom Encoder

You can also use other encoders like:
- **HuBERT:** Facebook's self-supervised model
- **ContentVec:** Content-based audio encoder
- **Custom trained encoder:** Train your own on timbre data

## Model Selection Guide

| Model | Pros | Cons | Memory | Speed |
|-------|------|------|--------|-------|
| **Wav2Vec2-base** | Good quality, fast | Smaller embedding | ~150MB | Fast |
| **Wav2Vec2-large** | Better quality | Larger, slower | ~300MB | Medium |
| **Whisper-base** | Good for speech | Larger model | ~150MB | Medium |
| **Whisper-small** | Better quality | Slower | ~500MB | Slow |

**Recommendation:** Start with `facebook/wav2Vec2-base-960h` for good balance.

## Testing

After integration, test thoroughly:

```python
from python.prrot.timbre_embeddings import TimbreEmbeddingExtractor
import numpy as np

# Initialize
extractor = TimbreEmbeddingExtractor(embedding_dim=256)

# Test with real audio file
import librosa
audio, sr = librosa.load("test_audio.wav", sr=None)
embedding = extractor.extract_embedding(audio, sr)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding stats: min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f}")

# Test phoneme embeddings
phoneme_boundaries = [(0, 100), (100, 200), (200, 300)]  # ms
phoneme_embeddings = extractor.extract_phoneme_embeddings(audio, sr, phoneme_boundaries)
print(f"Extracted {len(phoneme_embeddings)} phoneme embeddings")

# Test aggregation
aggregated = extractor.aggregate_embeddings(phoneme_embeddings, method="mean")
print(f"Aggregated embedding shape: {aggregated.shape}")
```

## Performance Optimization

### Caching

Cache model loading:

```python
# Use singleton pattern or module-level cache
_model_cache = {}

def get_extractor(model_name="wav2vec2"):
    if model_name not in _model_cache:
        _model_cache[model_name] = TimbreEmbeddingExtractor()
    return _model_cache[model_name]
```

### Batch Processing

Process multiple audio segments in batch:

```python
def extract_batch_embeddings(self, audio_segments: List[np.ndarray], sample_rate: int) -> List[np.ndarray]:
    """Extract embeddings for multiple segments in batch"""
    # Stack segments
    # Process in batch
    # Return list of embeddings
```

## Troubleshooting

### CUDA Out of Memory

```python
# Use CPU instead
extractor = TimbreEmbeddingExtractor()
extractor.device = torch.device("cpu")
```

### Sample Rate Mismatch

```python
# Always resample to model's expected rate (16kHz for Wav2Vec2/Whisper)
if sample_rate != 16000:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
```

### Embedding Dimension Mismatch

The model's hidden dimension may not match your desired embedding_dim. Options:
1. Use PCA to reduce dimensions
2. Use learned linear projection
3. Truncate/pad (simpler but less optimal)

## References

- **Current Implementation:** `python/prrot/timbre_embeddings.py`
- **Wav2Vec2 Docs:** https://huggingface.co/docs/transformers/model_doc/wav2vec2
- **Whisper Docs:** https://github.com/openai/whisper
- **Model Inventory:** `docs/MODEL_INVENTORY.md`
