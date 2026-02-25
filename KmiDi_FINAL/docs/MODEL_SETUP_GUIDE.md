# Model Setup and Training Guide

This guide provides step-by-step instructions for setting up and training the missing models in the KmiDi system.

## Overview

The system currently has:
- ✅ 5 trained models (ready to use)
- ⚠️ 2 stub models (need training)
- ❌ 2 missing models (need to obtain/integrate)

---

## Part 1: Training Stub Models

### 1.1 Instrument Recognizer

**Status:** Stub exists, needs training
**Config:** `configs/instrument_recognizer_training.yaml`

#### Prerequisites

1. **Dataset:** Create or obtain `instrument_dataset_v1` with dual annotations:
   - Technical annotations: instrument_family, instrument_specific, technique, articulation, register
   - Emotional annotations: expression_style, energy_level, musical_role, sentiment, dynamics

2. **Data Sources:**
   - Use `penta_core/ml/datasets/instrument_synthetic.py` for synthetic data generation
   - Or collect real audio with annotations

#### Training Steps

```bash
# Navigate to training directory
cd KmiDi_TRAINING/training

# Train instrument recognizer
python train_integrated.py \
  --model instrument_recognizer \
  --config ../../KmiDi_FINAL/configs/instrument_recognizer_training.yaml \
  --epochs 100 \
  --device mps  # or cuda, cpu

# Or use training orchestrator
python -m penta_core.ml.training_orchestrator \
  --config ../../KmiDi_FINAL/configs/instrument_recognizer_training.yaml
```

#### Expected Output

- Checkpoint: `checkpoints/instrument_recognizer/best_model.pt`
- Exports: `ml/models/instrumentrecognizer.{json,onnx,mlpackage}`
- Training logs: `logs/instrument_recognizer/`

#### Validation

After training, verify the model:
```python
from penta_core.ml.model_registry import get_model
model = get_model("instrumentrecognizer")
print(f"Model status: {model.status}")
```

---

### 1.2 Emotion Node Classifier

**Status:** Stub exists, needs training
**Config:** `configs/emotion_node_classifier_training.yaml`

#### Prerequisites

1. **Dataset:** Create or obtain `emotion_thesaurus_dataset_v1` with hierarchical labels:
   - Base emotion (6 classes: HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST)
   - Sub emotion (36 classes, 6 per base)
   - Sub-sub emotion (216 classes, 6 per sub)
   - Intensity tier (6 levels: subtle, mild, moderate, strong, intense, overwhelming)
   - Key (24 keys: 12 major + 12 minor)

2. **Thesaurus Structure:**
   - Source: `data/emotion_thesaurus/`
   - Structure: 6×6×6 = 216 total emotion nodes
   - Node ID formula: `base_idx * 36 + sub_idx * 6 + subsub_idx`

#### Training Steps

```bash
# Navigate to training directory
cd KmiDi_TRAINING/training

# Train emotion node classifier
python train_integrated.py \
  --model emotion_node_classifier \
  --config ../../KmiDi_FINAL/configs/emotion_node_classifier_training.yaml \
  --epochs 150 \
  --device mps  # or cuda, cpu

# Or use training orchestrator
python -m penta_core.ml.training_orchestrator \
  --config ../../KmiDi_FINAL/configs/emotion_node_classifier_training.yaml
```

#### Expected Output

- Checkpoint: `checkpoints/emotion_node_classifier/best_model.pt`
- Exports: `ml/models/emotionnodeclassifier.{json,onnx,mlpackage}`
- Training logs: `logs/emotion_node_classifier/`

#### Validation

After training, verify the model:
```python
from penta_core.ml.model_registry import get_model
model = get_model("emotionnodeclassifier")
print(f"Model status: {model.status}")
```

---

## Part 2: Obtaining Missing Models

### 2.1 Phoneme Aligner (PRROT System)

**Status:** Missing, placeholder implementation
**Requirements:** 3B parameter Q4 quantized model
**Location:** `python/prrot/models/phoneme_aligner_q4.bin`

#### Option A: Use Pre-trained Model

1. **Find compatible model:**
   - Look for 3B parameter models compatible with llama.cpp
   - Must support Q4 quantization
   - License: Apache 2.0, MIT, or BSD

2. **Quantize to Q4:**
   ```bash
   # Using llama.cpp
   ./llama-quantize model.bin phoneme_aligner_q4.bin Q4
   ```

3. **Place model:**
   ```bash
   mkdir -p python/prrot/models
   cp phoneme_aligner_q4.bin python/prrot/models/
   ```

#### Option B: Train Custom Model

1. **Prepare dataset:**
   - Audio-text pairs with phoneme alignments
   - Use tools like Montreal Forced Aligner (MFA)

2. **Train model:**
   - Architecture: Transformer-based (3B parameters)
   - Task: Phoneme alignment from audio

3. **Quantize:**
   ```bash
   ./llama-quantize trained_model.bin phoneme_aligner_q4.bin Q4
   ```

#### Integration

Update `python/prrot/phoneme_aligner.py`:
```python
def align_phonemes(audio_path: str, text: str) -> List[PhonemeAlignment]:
    # Load Q4 quantized model
    model_path = Path(__file__).parent / "models" / "phoneme_aligner_q4.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load and run inference
    # ... implementation
```

---

### 2.2 Timbre Embedding Extractor

**Status:** Missing, returns random embeddings
**Requirements:** Pre-trained encoder (Wav2Vec2 or Whisper)
**Location:** `python/prrot/timbre_embeddings.py`

#### Option A: Use Wav2Vec2

1. **Install dependencies:**
   ```bash
   pip install transformers torchaudio
   ```

2. **Download pre-trained model:**
   ```python
   from transformers import Wav2Vec2Model
   model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
   ```

3. **Integrate into `timbre_embeddings.py`:**
   ```python
   from transformers import Wav2Vec2Model
   import torchaudio

   class TimbreEmbeddingExtractor:
       def __init__(self):
           self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
           self.model.eval()

       def extract(self, audio_path: str) -> np.ndarray:
           # Load audio
           waveform, sample_rate = torchaudio.load(audio_path)

           # Resample if needed
           if sample_rate != 16000:
               resampler = torchaudio.transforms.Resample(sample_rate, 16000)
               waveform = resampler(waveform)

           # Extract features
           with torch.no_grad():
               features = self.model(waveform).last_hidden_state

           # Pool to fixed size (e.g., mean pooling)
           embedding = features.mean(dim=1).squeeze().numpy()

           return embedding
   ```

#### Option B: Use Whisper

1. **Install dependencies:**
   ```bash
   pip install openai-whisper
   ```

2. **Use Whisper encoder:**
   ```python
   import whisper

   class TimbreEmbeddingExtractor:
       def __init__(self):
           self.model = whisper.load_model("base")

       def extract(self, audio_path: str) -> np.ndarray:
           # Load audio
           audio = whisper.load_audio(audio_path)

           # Extract encoder features
           with torch.no_grad():
               mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
               features = self.model.encoder(mel)

           # Pool to fixed size
           embedding = features.mean(dim=1).squeeze().cpu().numpy()

           return embedding
   ```

#### Integration

Update `python/prrot/timbre_embeddings.py` to use the chosen encoder instead of returning random embeddings.

---

## Part 3: Optional Models

### 3.1 LLaMA ONNX (Text-to-Music)

**Status:** Config exists, model file needed
**Location:** `ml/models/llama_onnx.json`

#### Setup

1. **Obtain LLaMA model:**
   - Download from Hugging Face or other source
   - Convert to ONNX format

2. **Update config:**
   ```json
   {
     "model_path": "path/to/llama_model.onnx",
     "provider": "cpu",  // or "coreml", "cuda"
     "enabled": true
   }
   ```

3. **Enable in code:**
   ```python
   current_state["use_llama"] = True
   # or
   current_state["llama_prompt"] = "Create a happy melody"
   ```

---

## Verification Checklist

After completing setup, verify all models:

```python
from penta_core.ml.model_registry import list_models, get_model

# List all models
models = list_models()
print(f"Total models: {len(models)}")

# Check each model
for model in models:
    print(f"{model.name}: {model.task.value} - Status: {model.status}")
    if model.path and Path(model.path).exists():
        print(f"  ✓ Model file exists")
    else:
        print(f"  ✗ Model file missing: {model.path}")
```

---

## Troubleshooting

### Training Issues

1. **Out of Memory:**
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision (AMP)

2. **Slow Training:**
   - Use GPU (CUDA/MPS)
   - Increase num_workers for data loading
   - Use mixed precision

3. **Poor Performance:**
   - Check dataset quality
   - Adjust learning rate
   - Increase training epochs
   - Try different architectures

### Model Loading Issues

1. **Model Not Found:**
   - Check model path in registry
   - Verify model files exist
   - Run model discovery: `get_registry().discover()`

2. **Format Errors:**
   - Verify model format matches backend
   - Check model version compatibility
   - Re-export model if needed

---

## Resources

- **Model Registry:** `python/penta_core/ml/model_registry.py`
- **Training Orchestrator:** `python/penta_core/ml/training_orchestrator.py`
- **Model Inventory:** `docs/MODEL_INVENTORY.md`
- **PRROT Requirements:** `PRROT_BUILD_READY.md`

---

## Next Steps

1. ✅ Create training configs (done)
2. ⏳ Prepare datasets for stub models
3. ⏳ Train instrument recognizer
4. ⏳ Train emotion node classifier
5. ⏳ Obtain/integrate phoneme aligner
6. ⏳ Integrate timbre extractor
7. ⏳ (Optional) Set up LLaMA ONNX
