# Model Management Scripts

Helper scripts for managing, verifying, and setting up ML models in the KmiDi system.

## Available Scripts

### 1. `verify_models.py`

Verifies all models in the system are properly configured and available.

**Usage:**
```bash
python scripts/verify_models.py
```

**What it does:**
- Checks all trained models exist and are accessible
- Identifies stub models that need training
- Verifies missing models
- Checks PRROT system models (phoneme aligner)
- Tests timbre extractor integration
- Provides recommendations

**Output:**
- Detailed report of model status
- Summary statistics
- Recommendations for next steps

**Example:**
```bash
$ python scripts/verify_models.py

================================================================================
MODEL VERIFICATION REPORT
================================================================================

✅ TRAINED MODELS:
  ✅ emotionrecognizer      emotion_embedding      Exists (4.63 MB)
  ✅ melodytransformer       melody_generation      Exists (7.36 MB)
  ✅ harmonypredictor        harmony_prediction     Exists (0.86 MB)
  ✅ dynamicsengine          dynamics_mapping       Exists (0.16 MB)
  ✅ groovepredictor         groove_prediction      Exists (0.22 MB)

⚠️  STUB MODELS (Need Training):
  ⚠️  instrumentrecognizer   audio_classification   Missing: ...
  ⚠️  emotionnodeclassifier  emotion_classification Missing: ...

...
```

---

### 2. `setup_timbre_extractor.py`

Helps set up the timbre embedding extractor with Wav2Vec2 or Whisper.

**Usage:**
```bash
# Set up with Wav2Vec2 (default)
python scripts/setup_timbre_extractor.py --model wav2vec2

# Set up with Whisper
python scripts/setup_timbre_extractor.py --model whisper

# Set up and test
python scripts/setup_timbre_extractor.py --model wav2vec2 --test
```

**What it does:**
- Checks if required dependencies are installed
- Verifies if integration is already complete
- Tests the integration
- Provides setup instructions

**Dependencies:**
- For Wav2Vec2: `pip install transformers torchaudio`
- For Whisper: `pip install openai-whisper`

**Example:**
```bash
$ python scripts/setup_timbre_extractor.py --model wav2vec2 --test

================================================================================
TIMBRE EXTRACTOR SETUP
================================================================================

Setting up Wav2Vec2 integration...
✅ transformers and torchaudio are installed
⚠️  Wav2Vec2 not yet integrated
   See docs/TIMBRE_EXTRACTOR_INTEGRATION.md for integration instructions

Testing wav2vec2 integration...
✅ Extraction successful!
   Embedding shape: (256,)
   Embedding norm: 1.0000
   Embedding range: [-0.1234, 0.5678]
   ⚠️  Warning: This appears to be a placeholder implementation
```

---

### 3. `prepare_training_data.py`

Helps prepare datasets for training stub models.

---

### 4. `test_models.py`

Tests all available models to ensure they work correctly.

**Usage:**
```bash
python scripts/test_models.py
```

**What it does:**
- Tests model registry functionality
- Verifies trained models can be loaded
- Tests timbre extractor functionality
- Tests phoneme aligner functionality
- Provides test results summary

**Example:**
```bash
$ python scripts/test_models.py

================================================================================
MODEL TESTING SUITE
================================================================================

Testing Model Registry:
  ✅ Model Registry: Registry working: 5 models, 5 discovered

Testing Trained Models:
  ✅ emotionrecognizer      : File exists, ready for inference
  ✅ melodytransformer       : File exists, ready for inference
  ✅ harmonypredictor        : File exists, ready for inference
  ✅ dynamicsengine          : File exists, ready for inference
  ✅ groovepredictor         : File exists, ready for inference

Testing Timbre Extractor:
  ✅ Timbre Extractor: Extraction successful (norm=1.0000, range=[-0.123, 0.456])

Testing Phoneme Aligner:
  ❌ Phoneme Aligner: Model file missing: ...
```

**Usage:**
```bash
# Prepare instrument recognition dataset
python scripts/prepare_training_data.py --dataset instrument

# Prepare emotion thesaurus dataset
python scripts/prepare_training_data.py --dataset emotion_thesaurus

# Specify custom output directory
python scripts/prepare_training_data.py --dataset instrument --output-dir /path/to/data
```

**What it does:**
- Creates directory structure for datasets
- Generates metadata templates
- Creates annotation templates
- Provides next steps

**Output:**
- Creates dataset directory structure:
  ```
  data/datasets/
  └── instrument_dataset_v1/
      ├── train/
      ├── val/
      ├── test/
      └── metadata/
          ├── dataset_info.json
          └── annotation_template.json
  ```

**Example:**
```bash
$ python scripts/prepare_training_data.py --dataset instrument

================================================================================
TRAINING DATA PREPARATION
================================================================================

✅ Created: data/datasets/instrument_dataset_v1/train
✅ Created: data/datasets/instrument_dataset_v1/val
✅ Created: data/datasets/instrument_dataset_v1/test
✅ Created: data/datasets/instrument_dataset_v1/metadata

Preparing instrument dataset...
✅ Created metadata template: .../metadata/dataset_info.json
✅ Created annotation template: .../metadata/annotation_template.json

================================================================================
NEXT STEPS
================================================================================

1. Add audio files to: data/datasets/instrument_dataset_v1/train/audio/
2. Create annotations.json with format from: .../annotation_template.json
3. Split data into train/val/test
4. Update dataset_info.json with actual counts
```

---

## Quick Start

### Verify Current Model Status

```bash
python scripts/verify_models.py
```

### Set Up Timbre Extractor

```bash
# Install dependencies
pip install transformers torchaudio

# Set up and test
python scripts/setup_timbre_extractor.py --model wav2vec2 --test
```

### Prepare Training Data

```bash
# For instrument recognizer
python scripts/prepare_training_data.py --dataset instrument

# For emotion node classifier
python scripts/prepare_training_data.py --dataset emotion_thesaurus
```

---

## Integration with Documentation

These scripts work alongside the documentation:

- **Model Inventory:** `docs/MODEL_INVENTORY.md`
- **Setup Guide:** `docs/MODEL_SETUP_GUIDE.md`
- **Phoneme Aligner:** `docs/PHONEME_ALIGNER_INTEGRATION.md`
- **Timbre Extractor:** `docs/TIMBRE_EXTRACTOR_INTEGRATION.md`

---

## Troubleshooting

### Script Not Found

Make sure you're running from the project root:
```bash
cd /path/to/KmiDi_FINAL
python scripts/verify_models.py
```

### Import Errors

Install required dependencies:
```bash
pip install -r requirements.txt
# Or install specific packages as needed
```

### Permission Errors

Make scripts executable:
```bash
chmod +x scripts/*.py
```

---

## Adding New Scripts

When adding new scripts:

1. Place in `scripts/` directory
2. Make executable: `chmod +x scripts/new_script.py`
3. Add shebang: `#!/usr/bin/env python3`
4. Add to this README
5. Include help text: `--help` flag

---

## See Also

- **Model Registry:** `python/penta_core/ml/model_registry.py`
- **Training Orchestrator:** `python/penta_core/ml/training_orchestrator.py`
- **Implementation Summary:** `docs/MODEL_IMPLEMENTATION_SUMMARY.md`
