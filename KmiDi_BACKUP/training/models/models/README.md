# Kelly ML Models

This directory contains trained neural network models for Kelly's emotion-driven MIDI generation.

## Model Architecture

Kelly uses a 5-model pipeline (~1M params, ~4MB total, <10ms inference):

| Model | Input → Output | Params | Format | Purpose |
|-------|----------------|--------|--------|---------|
| `emotionrecognizer.json` | 128 → 64 | ~500K | RTNeural | Audio features → Emotion embedding |
| `melodytransformer.json` | 64 → 128 | ~400K | RTNeural | Emotion → Note probabilities |
| `harmonypredictor.json` | 128 → 64 | ~100K | RTNeural | Context → Chord predictions |
| `dynamicsengine.json` | 32 → 16 | ~20K | RTNeural | Emotion → Expression params |
| `groovepredictor.json` | 64 → 32 | ~25K | RTNeural | Emotion → Groove/timing |

## Model Formats

### RTNeural (JSON)
Primary format for C++ inference. Models are stored as JSON with layer weights.

```json
{
  "layers": [
    {
      "type": "dense",
      "shape": [128, 256],
      "weights": [...],
      "bias": [...]
    }
  ]
}
```

### ONNX (Optional)
Cross-platform format for flexible deployment. Enable with `ENABLE_ONNX_RUNTIME=ON`.

#### LLaMA ONNX (optional, text → music control ideas)
- Config file: `models/llama_onnx.json`
- Required: replace `model_path` with your LLaMA ONNX file.
- Provider: set to `cpu`, `coreml`, or `cuda` depending on your runtime.
- Used by: `music_brain/intelligence/suggestion_engine.py` when `current_state["use_llama"]` is true or `current_state["llama_prompt"]` is provided.

## Training

Models are trained using Python (see `penta_core/ml/`):

```bash
# Train emotion recognizer
python -m penta_core.ml.train --model emotion_recognizer --data data/emotion_dataset

# Export to RTNeural JSON
python -m penta_core.ml.export --model emotion_recognizer --format rtneural
```

## Fallback Mode

If models are not present, Kelly uses heuristic fallbacks:
- EmotionRecognizer: RMS → valence, spectral centroid → arousal
- MelodyTransformer: Scale-based note probabilities
- HarmonyPredictor: Circle of fifths relationships
- DynamicsEngine: Envelope following
- GroovePredictor: Tempo-based swing estimation

## Model Specifications

### EmotionRecognizer
```
Input:  128 floats (audio features: MFCC, spectral, temporal)
Output: 64 floats (emotion embedding)
        [0-15]:  Valence components
        [16-31]: Arousal components  
        [32-47]: Dominance components
        [48-63]: Complexity components
```

### MelodyTransformer
```
Input:  64 floats (emotion embedding)
Output: 128 floats (note probabilities, MIDI 0-127)
```

### HarmonyPredictor
```
Input:  128 floats (64 emotion + 64 audio context)
Output: 64 floats (chord/harmony predictions)
```

### DynamicsEngine
```
Input:  32 floats (compressed emotion)
Output: 16 floats (velocity, timing, expression)
```

### GroovePredictor
```
Input:  64 floats (emotion embedding)
Output: 32 floats (swing, humanization, accents)
```

