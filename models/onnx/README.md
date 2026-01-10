# ONNX Model Export Status

## ⚠️ Known Issue: PyTorch 2.9.1 + onnxscript Compatibility Bug

**Status**: ONNX export is currently blocked by a compatibility bug between PyTorch 2.9.1 and onnxscript 0.5.7.

**Error**: `TypeError: Expecting a type not f<class 'typing.Union'> for typeinfo`

**Root Cause**: The error occurs during ONNX exporter registry initialization (in `onnxscript/converter.py`), before any actual export attempt. This is a known bug in the onnxscript library when used with PyTorch 2.9.1.

## Workarounds

### Option 1: Use RTNeural JSON Format (Recommended)
All models are successfully exported to RTNeural JSON format, which is the **primary format** for C++ real-time inference:
- ✅ `emotionrecognizer.json`
- ✅ `harmonypredictor.json`
- ✅ `melodytransformer.json`
- ✅ `groovepredictor.json`
- ✅ `dynamicsengine.json`

RTNeural JSON is **better suited** for real-time audio applications than ONNX, as it's:
- Optimized for real-time C++ inference
- Lighter weight (no ONNX Runtime dependency)
- Better latency characteristics
- Already integrated in `penta_core` C++ codebase

### Option 2: Downgrade PyTorch (If ONNX is Required)
If ONNX export is specifically needed:
```bash
pip install torch==2.8.2 onnx==1.16.0 onnxruntime==1.17.1
```

Then run:
```bash
python3 scripts/export_models.py --format onnx --output-dir models/onnx
```

### Option 3: Wait for Fix
Track these issues:
- PyTorch: https://github.com/pytorch/pytorch/issues
- onnxscript: https://github.com/microsoft/onnxscript/issues

## Current Export Status

| Model | RTNeural JSON | ONNX | CoreML |
|-------|--------------|------|--------|
| emotionrecognizer | ✅ | ❌ (bug) | ✅ |
| harmonypredictor | ✅ | ❌ (bug) | ✅ |
| melodytransformer | ✅ | ❌ (bug) | ✅ |
| groovepredictor | ✅ | ❌ (bug) | ✅ |
| dynamicsengine | ✅ | ❌ (bug) | ✅ |

## Recommendation

**Use RTNeural JSON format** for production deployment. ONNX is optional and only needed for cross-platform inference outside the C++ codebase. Since all models work with RTNeural JSON (which is the primary target), ONNX export is not a blocker for production.
