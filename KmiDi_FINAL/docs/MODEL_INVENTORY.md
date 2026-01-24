# Model Inventory and Requirements

**Last Updated:** 2026-01-22
**Registry Location:** `ml/models/registry.json`

## Overview

This document provides a comprehensive inventory of all ML models in the KmiDi system, including trained models, stubs, and missing models that need to be obtained or trained.

## Model Status Legend

- ✅ **Trained** - Model is trained and available for use
- ⚠️ **Stub** - Placeholder exists, needs training
- ❌ **Missing** - Required but not available
- 🔵 **Optional** - Nice to have, not required

---

## Trained Models (Available)

### 1. emotionrecognizer ✅

**Location:** `ml/models/emotionrecognizer.*`

- **Status:** Trained (2026-01-04)
- **Task:** Emotion embedding
- **Architecture:** RTNeural (128→512→256→128→7)
- **Input:** 128 floats (audio features: MFCC, spectral, temporal)
- **Output:** 7 floats (emotion embedding)
- **Parameters:** ~500K
- **Formats:** RTNeural JSON, ONNX, CoreML
- **Inference Target:** 5ms
- **Training Info:**
  - Dataset: emotion_embedding_dataset_v1
  - Epochs: 30
  - Batch Size: 8
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss: Cross Entropy
  - Train Loss: 0.060
  - Val Loss: 1.208
  - Test Loss: 0.678
- **Integration:** Python API ✅, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available (RMS → valence, spectral centroid → arousal)

**Files:**
- `emotionrecognizer.json` (RTNeural)
- `emotionrecognizer.mlpackage/` (CoreML)
- `emotionrecognizer.onnx` (ONNX)

---

### 2. melodytransformer ✅

**Location:** `ml/models/melodytransformer.*`

- **Status:** Trained (2026-01-04)
- **Task:** Melody generation
- **Architecture:** RTNeural (1→256→256→256→88)
- **Input:** 1 float (emotion embedding scalar)
- **Output:** 88 floats (note probabilities, MIDI 0-127)
- **Parameters:** ~400K
- **Formats:** RTNeural JSON, ONNX, CoreML
- **Inference Target:** 5ms
- **Training Info:**
  - Dataset: melody_generation_dataset_v1
  - Epochs: 30
  - Batch Size: 8
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss: Cross Entropy
  - Train Loss: 3.788
  - Val Loss: 3.866
  - Test Loss: 3.863
- **Integration:** Python API ✅, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available (scale-based note probabilities)

**Files:**
- `melodytransformer.json` (RTNeural)
- `melodytransformer.mlpackage/` (CoreML)
- `melodytransformer.onnx` (ONNX)

---

### 3. harmonypredictor ✅

**Location:** `ml/models/harmonypredictor.*`

- **Status:** Trained (2026-01-04)
- **Task:** Harmony prediction
- **Architecture:** RTNeural (11264→256→128→12)
- **Input:** 11264 floats (64 emotion + 64 audio context + large context window)
- **Output:** 12 floats (chord/harmony predictions)
- **Parameters:** ~100K
- **Formats:** RTNeural JSON, ONNX, CoreML
- **Inference Target:** 3ms
- **Training Info:**
  - Dataset: harmony_prediction_dataset_v1
  - Epochs: 30
  - Batch Size: 8
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss: Cross Entropy
  - Train Loss: 0.001
  - Val Loss: 1.267
  - Test Loss: 0.898
- **Integration:** Python API ✅, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available (circle of fifths relationships)

**Files:**
- `harmonypredictor.json` (RTNeural)
- `harmonypredictor.mlpackage/` (CoreML)
- `harmonypredictor.onnx` (ONNX)

---

### 4. dynamicsengine ✅

**Location:** `ml/models/dynamicsengine.*`

- **Status:** Trained (2026-01-04)
- **Task:** Dynamics mapping
- **Architecture:** RTNeural (32→128→64→16)
- **Input:** 32 floats (compressed emotion)
- **Output:** 16 floats (velocity, timing, expression parameters)
- **Parameters:** ~20K
- **Formats:** RTNeural JSON, ONNX, CoreML
- **Inference Target:** 1ms
- **Training Info:**
  - Dataset: dynamics_mapping_dataset_v1
  - Epochs: 30
  - Batch Size: 8
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss: MSE
  - Train Loss: 0.818
  - Val Loss: 0.842
  - Test Loss: 0.883
- **Integration:** Python API ✅, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available (envelope following)

**Files:**
- `dynamicsengine.json` (RTNeural)
- `dynamicsengine.mlpackage/` (CoreML)
- `dynamicsengine.onnx` (ONNX)

---

### 5. groovepredictor ✅

**Location:** `ml/models/groovepredictor.*`

- **Status:** Trained (2026-01-04)
- **Task:** Groove prediction
- **Architecture:** RTNeural (2→128→64→32)
- **Input:** 2 floats (emotion embedding compressed)
- **Output:** 32 floats (swing, humanization, accents)
- **Parameters:** ~25K
- **Formats:** RTNeural JSON, ONNX, CoreML
- **Inference Target:** 2ms
- **Training Info:**
  - Dataset: groove_prediction_dataset_v1
  - Epochs: 30
  - Batch Size: 8
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss: MSE
  - Train Loss: 0.457
  - Val Loss: 0.027
  - Test Loss: 0.035
- **Integration:** Python API ✅, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available (tempo-based swing estimation)

**Files:**
- `groovepredictor.json` (RTNeural)
- `groovepredictor.mlpackage/` (CoreML)
- `groovepredictor.onnx` (ONNX)

---

## Stub/Placeholder Models (Need Training)

### 6. instrumentrecognizer ⚠️

**Location:** `ml/models/instrumentrecognizer.json`

- **Status:** Stub (not trained)
- **Task:** Dual instrument recognition
- **Architecture:** CNN (64→128→256→512) → Dual Heads
  - Technical Head: 256→128→80 (instrument_family, instrument_specific, technique, articulation, register)
  - Emotional Head: 256→128→80 (expression_style, energy_level, musical_role, sentiment_valence, sentiment_arousal)
- **Input:** 128 floats (audio features)
- **Output:** 160 floats (80 technical + 80 emotional)
- **Parameters:** ~2M (estimated)
- **Formats:** RTNeural JSON (placeholder only)
- **Inference Target:** 10ms
- **Note:** "Dual-head model: Technical (instrument+technique) + Emotional (expression+sentiment). Replace with trained model."
- **Integration:** Python API ❌, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available

**Training Requirements:**
- Dataset: instrument_dataset_v1 (needs to be created/collected)
- Multi-task learning with dual heads
- Loss: Multi-task loss (technical + emotional)

**Files:**
- `instrumentrecognizer.json` (stub only)

---

### 7. emotionnodeclassifier ⚠️

**Location:** `ml/models/emotionnodeclassifier.json`

- **Status:** Stub (not trained)
- **Task:** Emotion node classification (6×6×6 thesaurus validation)
- **Architecture:** CNN (64→128→256→512) → Multi-head
  - Emotion Node Head: 216 outputs (full 6×6×6 node classification)
  - Base Emotion Head: 6 outputs (HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST)
  - Sub Emotion Head: 36 outputs (6 per base emotion)
  - Intensity Tier Head: 6 outputs (subtle, mild, moderate, strong, intense, overwhelming)
  - Key Detection Head: 24 outputs (12 major + 12 minor keys)
- **Input:** 128 floats (audio features)
- **Output:** 258 floats (216 + 6 + 36 + 6 + 24)
- **Parameters:** ~3M (estimated)
- **Formats:** RTNeural JSON (placeholder only)
- **Inference Target:** 15ms
- **Note:** "6×6×6 thesaurus validation model. Classifies all 216 emotion nodes across 24 keys."
- **Integration:** Python API ❌, C++ ML Interface ❌, Tauri UI ❌
- **Fallback:** Available

**Thesaurus Structure:**
- 6×6×6 = 216 total emotion nodes
- 6 intensity tiers
- 24 keys (12 major + 12 minor)
- Source: `data/emotion_thesaurus/`

**Training Requirements:**
- Dataset: emotion_thesaurus_dataset_v1 (needs to be created)
- Hierarchical cross-entropy loss
- Multi-head architecture

**Files:**
- `emotionnodeclassifier.json` (stub only)

---

## Optional Models

### 8. llama_onnx 🔵

**Location:** `ml/models/llama_onnx.json`

- **Status:** Config exists, model file needs to be provided
- **Task:** Text → music control ideas
- **Format:** ONNX
- **Usage:** Used by `music_brain/intelligence/suggestion_engine.py` when:
  - `current_state["use_llama"]` is true, OR
  - `current_state["llama_prompt"]` is provided
- **Provider Options:** cpu, coreml, cuda
- **Note:** Optional feature for text-to-music control

**Files:**
- `llama_onnx.json` (config only, needs actual ONNX model file)

---

## Missing Models (Required for PRROT System)

### 9. Phoneme Aligner ❌

**Location:** `python/prrot/models/phoneme_aligner_q4.bin` (expected)

- **Status:** Placeholder implementation
- **Task:** Phoneme alignment for speech-to-MIDI
- **Requirements:**
  - 3B parameter model
  - Q4 quantization (required for 16GB systems)
  - Format: Q4 quantized binary (compatible with llama.cpp)
  - Size: ~1.5-2GB in memory
  - License: Must be Apache 2.0, MIT, or BSD
- **Current Implementation:** Returns placeholder
- **Reference:** `PRROT_BUILD_READY.md` line 116

**Integration Steps:**
1. Obtain or train 3B parameter model
2. Quantize to Q4 format
3. Place in `models/phoneme_aligner_q4.bin`
4. Update `python/prrot/phoneme_aligner.py` to load actual model

---

### 10. Timbre Embedding Extractor ❌

**Location:** `python/prrot/timbre_embeddings.py` (implementation)

- **Status:** Returns random embeddings (placeholder)
- **Task:** Extract timbre embeddings from audio
- **Requirements:**
  - Pre-trained encoder model
  - Options: Wav2Vec2, Whisper, or similar
  - Purpose: Extract timbre embeddings for voice cloning/synthesis
- **Current Implementation:** Returns random embeddings
- **Reference:** `PRROT_BUILD_READY.md` line 121

**Integration Steps:**
1. Choose encoder model (Wav2Vec2 recommended)
2. Download pre-trained weights
3. Integrate into `timbre_embeddings.py`
4. Update model loading logic

---

## Model Registry System

The system uses a centralized model registry for discovery and management:

**Location:** `python/penta_core/ml/model_registry.py`

**Registry File:** `ml/models/registry.json`
**Schema:** `ml/models/registry.schema.json`

**Features:**
- Automatic model discovery from directories
- Multiple backend support: ONNX, TensorFlow Lite, CoreML, PyTorch, TorchScript, RTNeural JSON
- Version management
- Task-based filtering
- Model metadata tracking

**Default Search Directories:**
- `Data_Files/models/` (project directory)
- `~/.idaw/models/` (user directory)

---

## Supported Model Tasks

From `model_registry.py`, the system supports these tasks:

- ✅ `emotion_embedding` - emotionrecognizer
- ✅ `melody_generation` - melodytransformer
- ✅ `harmony_prediction` - harmonypredictor
- ✅ `dynamics_mapping` - dynamicsengine
- ✅ `groove_prediction` - groovepredictor
- ⚠️ `intent_mapping` - (no model yet)
- ⚠️ `audio_classification` - (no model yet)
- ⚠️ `chord_prediction` - (no model yet)
- ⚠️ `chord_detection` - (no model yet)
- ⚠️ `key_detection` - (no model yet)
- ⚠️ `tempo_estimation` - (no model yet)
- ⚠️ `style_transfer` - (no model yet)
- ⚠️ `emotion_classification` - (no model yet)
- ⚠️ `audio_generation` - (no model yet)
- ⚠️ `onset_detection` - (no model yet)
- ⚠️ `beat_tracking` - (no model yet)

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Trained Models** | 5 |
| **Stub Models** | 2 |
| **Optional Models** | 1 |
| **Missing Models** | 2 |
| **Total Tracked** | 10 |

**Total Parameters (Trained):** ~1.045M
**Total Model Size (Trained):** ~4MB (RTNeural JSON format)

---

## Next Steps

### High Priority
1. **Train instrumentrecognizer** - Dual-head CNN for instrument recognition
2. **Train emotionnodeclassifier** - 6×6×6 thesaurus validation model
3. **Obtain Phoneme Aligner** - 3B parameter Q4 quantized model for PRROT

### Medium Priority
4. **Integrate Timbre Extractor** - Wav2Vec2 or Whisper for timbre embeddings
5. **Provide LLaMA ONNX** - If text-to-music features are needed

### Low Priority
6. **C++ ML Interface Integration** - Enable C++ inference for all models
7. **Tauri UI Integration** - Enable UI access to models

---

## Training Resources

**Training Script:** `KmiDi_TRAINING/training/training/train_integrated.py`

**Usage:**
```bash
# Train all models
python train_integrated.py --all --device mps

# Train specific model
python train_integrated.py --model instrument_recognizer --epochs 100

# With config file
python train_integrated.py --config integrated_training_config.yaml
```

**Output Location:** `checkpoints/`

---

## Model Cards

Model cards with detailed information are located in:
- `docs/model_cards/emotionrecognizer.md`
- `docs/model_cards/melodytransformer.md`
- `docs/model_cards/harmonypredictor.md`
- `docs/model_cards/dynamicsengine.md`
- `docs/model_cards/groovepredictor.md`
- `docs/model_cards/instrumentrecognizer.md`
- `docs/model_cards/emotionnodeclassifier.md`

---

## References

- Model Registry: `python/penta_core/ml/model_registry.py`
- PRROT Requirements: `PRROT_BUILD_READY.md`
- Training Guidelines: `docs/MK_TRAINING_GUIDELINES.md`
- Model README: `ml/models/README.md`
- Multi-Language Architecture: `docs/MULTI_LANGUAGE_ARCHITECTURE.md` - How models integrate with C++/Rust/JUCE
