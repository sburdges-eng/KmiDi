# Phoneme Aligner Model Integration Guide

## Overview

The Phoneme Aligner requires a 3B parameter Q4 quantized model for memory-constrained systems (16GB RAM). This guide explains how to obtain and integrate the model.

## Requirements

- **Model Size:** 3B parameters
- **Quantization:** Q4 (required for 16GB systems)
- **Format:** Q4 quantized binary (compatible with llama.cpp)
- **Memory:** ~1.5-2GB in memory
- **License:** Must be Apache 2.0, MIT, or BSD
- **Location:** `python/prrot/models/phoneme_aligner_q4.bin`

## Current Status

**File:** `python/prrot/phoneme_aligner.py`
**Status:** Placeholder implementation
**Model Loading:** Lines 29-60 (placeholder)
**Alignment:** Lines 62-92 (placeholder)

## Integration Options

### Option 1: Use Pre-trained Model (Recommended)

#### Step 1: Find Compatible Model

Look for 3B parameter models that:
- Support phoneme alignment or forced alignment
- Are compatible with llama.cpp
- Have permissive licenses (Apache 2.0, MIT, or BSD)

**Potential Sources:**
- Hugging Face: Search for "phoneme alignment" or "forced alignment" models
- Academic repositories: Look for published alignment models
- llama.cpp compatible models: Any 3B model that can be quantized

#### Step 2: Quantize to Q4

If you have a model in a different format:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert and quantize
./llama-quantize model.bin phoneme_aligner_q4.bin Q4
```

#### Step 3: Place Model

```bash
# Create models directory if it doesn't exist
mkdir -p python/prrot/models

# Copy model
cp phoneme_aligner_q4.bin python/prrot/models/
```

#### Step 4: Update Code

Update `python/prrot/phoneme_aligner.py`:

```python
def load_model(self) -> bool:
    """Load ML model (3B parameter, Q4 quantization)"""
    try:
        logger.info(f"Loading phoneme alignment model from {self.model_path}")

        # Check memory before loading
        within_limit, warning = self.memory_monitor.check_memory_limit()
        if not within_limit:
            raise RuntimeError(f"Cannot load model: {warning}")

        # Load using llama-cpp-python
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=4096,
                n_threads=4,
                verbose=False,
                n_gpu_layers=0  # CPU only, or set to >0 for GPU
            )
            logger.info("Model loaded successfully")
            return True
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            return False

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
```

Install dependency:
```bash
pip install llama-cpp-python
```

### Option 2: Train Custom Model

#### Step 1: Prepare Dataset

You'll need:
- Audio-text pairs with phoneme alignments
- Tools: Montreal Forced Aligner (MFA) or similar

```bash
# Install MFA
conda create -n aligner python=3.8
conda activate aligner
conda install -c conda-forge montreal-forced-alignment

# Create alignment dataset
mfa align dataset/audio dataset/dict.txt dataset/output/
```

#### Step 2: Train Model

Train a 3B parameter transformer model for phoneme alignment:

```python
# Example training script structure
import torch
from transformers import AutoModel, AutoTokenizer

# Architecture: Transformer-based (3B parameters)
# Task: Phoneme alignment from audio features
# Input: Audio features + transcript
# Output: Phoneme boundaries (start, end times)

# Training code here...
```

#### Step 3: Quantize

After training, quantize to Q4:

```bash
./llama-quantize trained_model.bin phoneme_aligner_q4.bin Q4
```

### Option 3: Use Alternative Library

If you can't find a 3B model, consider using:

1. **Montreal Forced Aligner (MFA)**
   - Direct integration without ML model
   - Requires different integration approach

2. **Gentle**
   - Forced alignment tool
   - Can be integrated as fallback

3. **Wav2Vec2 + Alignment Head**
   - Use pre-trained Wav2Vec2
   - Add alignment head
   - Smaller than 3B but may work

## Implementation Details

### Alignment Function

Update `align_phonemes` method in `phoneme_aligner.py`:

```python
def align_phonemes(
    self, audio_samples: np.ndarray, sample_rate: int, transcript: Optional[str] = None
) -> List[Tuple[PhonemeType, float, float]]:
    """
    Align phonemes to audio using ML model
    """
    if self.model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    logger.info(f"Aligning phonemes: {len(audio_samples)} samples at {sample_rate}Hz")

    # 1. Extract audio features
    # Convert audio_samples to features (MFCC, spectrogram, etc.)
    features = extract_audio_features(audio_samples, sample_rate)

    # 2. Convert transcript to phonemes (if provided)
    if transcript:
        phoneme_sequence = text_to_phonemes(transcript)
    else:
        # Use model to predict phonemes from audio
        phoneme_sequence = self.model.predict_phonemes(features)

    # 3. Run alignment model
    # This would use the loaded model to align phonemes to audio
    aligned_phonemes = self.model.align(features, phoneme_sequence)

    # 4. Return phoneme boundaries
    return [
        (phoneme, start_ms, end_ms)
        for phoneme, start_ms, end_ms in aligned_phonemes
    ]
```

## Testing

After integration, test the model:

```python
from python.prrot.phoneme_aligner import PhonemeAligner
import numpy as np

# Initialize
aligner = PhonemeAligner()

# Load model
if aligner.load_model():
    # Test alignment
    audio = np.random.randn(44100)  # 1 second at 44.1kHz
    transcript = "Hello world"

    alignments = aligner.align_phonemes(audio, 44100, transcript)
    print(f"Aligned {len(alignments)} phonemes")
else:
    print("Failed to load model")
```

## Memory Considerations

The Q4 quantization is critical for 16GB systems:

- **Full precision (FP32):** ~12GB memory
- **Q4 quantized:** ~1.5-2GB memory
- **Savings:** ~75% memory reduction

If you have more memory available, you could use:
- Q8 quantization (~3-4GB)
- FP16 (~6GB)
- Full FP32 (~12GB)

Update the quantization in `model_manager.py` if needed.

## Troubleshooting

### Model Not Found

```python
# Check model path
from pathlib import Path
model_path = Path("python/prrot/models/phoneme_aligner_q4.bin")
print(f"Model exists: {model_path.exists()}")
print(f"Model path: {model_path.absolute()}")
```

### Memory Issues

```python
# Check available memory
from python.prrot.utils.memory_monitor import MemoryMonitor
monitor = MemoryMonitor()
within_limit, warning = monitor.check_memory_limit()
print(f"Memory OK: {within_limit}")
if not within_limit:
    print(f"Warning: {warning}")
```

### Import Errors

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Or build from source if needed
pip install llama-cpp-python --no-binary :all:
```

## References

- **Current Implementation:** `python/prrot/phoneme_aligner.py`
- **Model Manager:** `python/prrot/model_manager.py`
- **PRROT Requirements:** `PRROT_BUILD_READY.md`
- **Model Inventory:** `docs/MODEL_INVENTORY.md`
