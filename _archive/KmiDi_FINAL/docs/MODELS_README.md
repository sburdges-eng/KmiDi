# KmiDi Model Management

Complete guide to managing ML models in the KmiDi system.

## Quick Start

1. **Check current status:**
   ```bash
   python scripts/verify_models.py
   ```

2. **View model inventory:**
   See [MODEL_INVENTORY.md](MODEL_INVENTORY.md) for complete list

3. **Get started with training:**
   See [MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md)

4. **Quick reference:**
   See [MODEL_QUICK_REFERENCE.md](MODEL_QUICK_REFERENCE.md)

## Documentation Index

### Core Documentation

- **[MODEL_INVENTORY.md](MODEL_INVENTORY.md)** - Complete catalog of all models
  - Trained models (5)
  - Stub models (2)
  - Missing models (2)
  - Optional models (1)

- **[MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md)** - Step-by-step setup instructions
  - Training stub models
  - Integrating missing models
  - Optional model setup

- **[MODEL_QUICK_REFERENCE.md](MODEL_QUICK_REFERENCE.md)** - Quick commands and reference
  - Common commands
  - File locations
  - Integration checklist

### Integration Guides

- **[PHONEME_ALIGNER_INTEGRATION.md](PHONEME_ALIGNER_INTEGRATION.md)** - Integrate 3B Q4 model
  - Finding/obtaining model
  - Quantization
  - Code integration

- **[TIMBRE_EXTRACTOR_INTEGRATION.md](TIMBRE_EXTRACTOR_INTEGRATION.md)** - Integrate Wav2Vec2/Whisper
  - Wav2Vec2 setup
  - Whisper setup
  - Code examples

### Summary & Tracking

- **[MODEL_IMPLEMENTATION_SUMMARY.md](MODEL_IMPLEMENTATION_SUMMARY.md)** - Implementation overview
  - What was completed
  - File structure
  - Next steps

- **[MODEL_COMPLETION_CHECKLIST.md](MODEL_COMPLETION_CHECKLIST.md)** - Track progress
  - Implementation tasks
  - Verification steps
  - Progress tracking

## Helper Scripts

All scripts are in `scripts/` directory. See [scripts/README.md](../scripts/README.md) for details.

### Verification & Testing

```bash
# Verify all models
python scripts/verify_models.py

# Test model functionality
python scripts/test_models.py
```

### Setup & Preparation

```bash
# Set up timbre extractor
python scripts/setup_timbre_extractor.py --model wav2vec2 --test

# Prepare training data
python scripts/prepare_training_data.py --dataset instrument
python scripts/prepare_training_data.py --dataset emotion_thesaurus
```

## Model Status Overview

| Category | Count | Status |
|----------|-------|--------|
| **Trained** | 5 | ✅ Ready |
| **Stub** | 2 | ⚠️ Need training |
| **Missing** | 2 | ❌ Need integration |
| **Optional** | 1 | 🔵 Config only |

### Trained Models (Ready to Use)

1. **emotionrecognizer** - Emotion embedding
2. **melodytransformer** - Melody generation
3. **harmonypredictor** - Harmony prediction
4. **dynamicsengine** - Dynamics mapping
5. **groovepredictor** - Groove prediction

### Stub Models (Need Training)

1. **instrumentrecognizer** - Dual-head instrument recognition
2. **emotionnodeclassifier** - 6×6×6 emotion thesaurus

### Missing Models (Need Integration)

1. **phoneme_aligner** - 3B Q4 quantized model
2. **timbre_extractor** - Wav2Vec2/Whisper integration

## Workflow

### For Stub Models

```
1. Prepare Dataset
   └─> python scripts/prepare_training_data.py --dataset <name>

2. Add Data
   └─> Add audio files and annotations

3. Train Model
   └─> python train_integrated.py --model <name> --config configs/<name>_training.yaml

4. Verify
   └─> python scripts/verify_models.py
   └─> python scripts/test_models.py
```

### For Missing Models

```
1. Choose Integration Method
   └─> See integration guides

2. Install Dependencies
   └─> pip install -r requirements_models.txt

3. Integrate Code
   └─> Copy from integration examples
   └─> Update implementation files

4. Test
   └─> python scripts/test_models.py
```

## File Locations

### Documentation
- `docs/MODEL_*.md` - All model documentation

### Configurations
- `configs/instrument_recognizer_training.yaml`
- `configs/emotion_node_classifier_training.yaml`

### Scripts
- `scripts/verify_models.py`
- `scripts/test_models.py`
- `scripts/setup_timbre_extractor.py`
- `scripts/prepare_training_data.py`

### Models
- `ml/models/` - Trained model files
- `ml/models/registry.json` - Model registry
- `python/prrot/models/` - PRROT system models

### Integration Examples
- `python/prrot/timbre_embeddings_integration_example.py`
- `python/prrot/phoneme_aligner_integration_example.py`

## Dependencies

Install model-related dependencies:

```bash
pip install -r requirements_models.txt
```

Or install specific packages:

```bash
# For timbre extractor (Wav2Vec2)
pip install transformers torchaudio

# For timbre extractor (Whisper)
pip install openai-whisper

# For phoneme aligner
pip install llama-cpp-python
```

## Common Tasks

### Check Model Status

```bash
python scripts/verify_models.py
```

### Train a Model

```bash
cd KmiDi_TRAINING/training
python train_integrated.py \
  --model <model_name> \
  --config ../../KmiDi_FINAL/configs/<model>_training.yaml
```

### Integrate Missing Model

1. Read the relevant integration guide
2. Copy code from integration example
3. Update the implementation file
4. Test with `python scripts/test_models.py`

### Prepare Training Data

```bash
python scripts/prepare_training_data.py --dataset <dataset_name>
```

## Troubleshooting

### Model Not Found

```bash
# Run verification
python scripts/verify_models.py

# Check registry
python -c "from penta_core.ml.model_registry import get_registry; get_registry().discover()"
```

### Import Errors

```bash
# Install dependencies
pip install -r requirements_models.txt
```

### Training Issues

- Check dataset format matches config
- Verify GPU/device availability
- Reduce batch size if OOM
- Check logs in `logs/` directory

See individual guides for detailed troubleshooting.

## Next Steps

1. **Start Here:** Run `python scripts/verify_models.py` to see current status
2. **Choose Task:**
   - Training stub models? → See [MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md)
   - Integrating missing models? → See integration guides
   - Need quick reference? → See [MODEL_QUICK_REFERENCE.md](MODEL_QUICK_REFERENCE.md)
3. **Track Progress:** Use [MODEL_COMPLETION_CHECKLIST.md](MODEL_COMPLETION_CHECKLIST.md)

## Support

- **Model Registry:** `python/penta_core/ml/model_registry.py`
- **Training Orchestrator:** `python/penta_core/ml/training_orchestrator.py`
- **Scripts Help:** `scripts/README.md`

## Related Documentation

- **Multi-Language Architecture:** `docs/MULTI_LANGUAGE_ARCHITECTURE.md` - How Rust, C++, JUCE, Python work together
- **Model Cards:** `docs/model_cards/`
- **Training Guidelines:** `docs/MK_TRAINING_GUIDELINES.md`
- **PRROT System:** `PRROT_BUILD_READY.md`

---

**Last Updated:** 2026-01-22
**For the most up-to-date information, see:** [MODEL_INVENTORY.md](MODEL_INVENTORY.md)
