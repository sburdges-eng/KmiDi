# Model Quick Reference

Quick reference guide for model management in KmiDi.

## Model Status

| Model | Status | Location | Action Needed |
|-------|--------|----------|---------------|
| emotionrecognizer | ✅ Trained | `ml/models/` | None |
| melodytransformer | ✅ Trained | `ml/models/` | None |
| harmonypredictor | ✅ Trained | `ml/models/` | None |
| dynamicsengine | ✅ Trained | `ml/models/` | None |
| groovepredictor | ✅ Trained | `ml/models/` | None |
| instrumentrecognizer | ⚠️ Stub | `ml/models/` | Train model |
| emotionnodeclassifier | ⚠️ Stub | `ml/models/` | Train model |
| phoneme_aligner | ❌ Missing | `python/prrot/models/` | Obtain Q4 model |
| timbre_extractor | ❌ Placeholder | `python/prrot/` | Integrate Wav2Vec2 |
| llama_onnx | 🔵 Optional | `ml/models/` | Provide model file |

## Quick Commands

### Verify All Models
```bash
python scripts/verify_models.py
```

### Test Models
```bash
python scripts/test_models.py
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
# Instrument recognizer
python scripts/prepare_training_data.py --dataset instrument

# Emotion node classifier
python scripts/prepare_training_data.py --dataset emotion_thesaurus
```

### Train Models
```bash
cd KmiDi_TRAINING/training

# Train instrument recognizer
python train_integrated.py \
  --model instrument_recognizer \
  --config ../../KmiDi_FINAL/configs/instrument_recognizer_training.yaml

# Train emotion node classifier
python train_integrated.py \
  --model emotion_node_classifier \
  --config ../../KmiDi_FINAL/configs/emotion_node_classifier_training.yaml
```

## File Locations

### Documentation
- **Main README:** `docs/MODELS_README.md` - Start here!
- **Complete Inventory:** `docs/MODEL_INVENTORY.md`
- **Setup Guide:** `docs/MODEL_SETUP_GUIDE.md`
- **Workflow Guide:** `docs/MODEL_WORKFLOW.md` - Visual workflows
- **Phoneme Aligner:** `docs/PHONEME_ALIGNER_INTEGRATION.md`
- **Timbre Extractor:** `docs/TIMBRE_EXTRACTOR_INTEGRATION.md`
- **Implementation Summary:** `docs/MODEL_IMPLEMENTATION_SUMMARY.md`
- **Completion Checklist:** `docs/MODEL_COMPLETION_CHECKLIST.md`

### Configurations
- **Instrument Recognizer:** `configs/instrument_recognizer_training.yaml`
- **Emotion Node Classifier:** `configs/emotion_node_classifier_training.yaml`

### Scripts
- **Verify Models:** `scripts/verify_models.py`
- **Setup Timbre Extractor:** `scripts/setup_timbre_extractor.py`
- **Prepare Training Data:** `scripts/prepare_training_data.py`

### Models
- **Trained Models:** `ml/models/*.{json,onnx,mlpackage}`
- **Registry:** `ml/models/registry.json`
- **PRROT Models:** `python/prrot/models/`

## Integration Checklist

### For Stub Models
- [ ] Prepare dataset using `prepare_training_data.py`
- [ ] Add audio files and annotations
- [ ] Train model using training config
- [ ] Verify model in registry
- [ ] Test model inference

### For Phoneme Aligner
- [ ] Obtain 3B parameter model
- [ ] Quantize to Q4 format
- [ ] Place in `python/prrot/models/phoneme_aligner_q4.bin`
- [ ] Update `phoneme_aligner.py` to load model
- [ ] Test alignment functionality

### For Timbre Extractor
- [ ] Install dependencies: `pip install transformers torchaudio`
- [ ] Run setup script: `python scripts/setup_timbre_extractor.py`
- [ ] Update `timbre_embeddings.py` with Wav2Vec2/Whisper
- [ ] Test extraction functionality

## Common Tasks

### Check Model Status
```python
from penta_core.ml.model_registry import list_models

models = list_models()
for model in models:
    print(f"{model.name}: {model.task.value}")
```

### Load a Model
```python
from penta_core.ml.model_registry import get_model

model = get_model("emotionrecognizer")
print(f"Path: {model.path}")
```

### Discover Models
```python
from penta_core.ml.model_registry import get_registry

registry = get_registry()
count = registry.discover()
print(f"Discovered {count} models")
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
pip install -r requirements.txt
pip install transformers torchaudio  # For timbre extractor
```

### Training Issues
- Check dataset format matches config
- Verify GPU/device availability
- Reduce batch size if OOM
- Check logs in `logs/` directory

## Support Resources

- **Model Registry Code:** `python/penta_core/ml/model_registry.py`
- **Training Orchestrator:** `python/penta_core/ml/training_orchestrator.py`
- **Scripts README:** `scripts/README.md`

## Next Steps

1. Run `python scripts/verify_models.py` to see current status
2. Follow recommendations from verification output
3. Use setup scripts for quick integration
4. Refer to detailed guides for complex tasks

---

**Last Updated:** 2026-01-22
**For detailed information, see:** `docs/MODEL_INVENTORY.md`
