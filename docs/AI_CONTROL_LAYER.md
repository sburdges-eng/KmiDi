# AI Control Layer Architecture

**Date:** January 18, 2026  
**Purpose:** Define AI/ML placement and boundaries in the KmiDi architecture  
**Reference:** Architectural Boundary Compliance Report, ML Frameworks Evaluation

## Core Principle

**"If this AI crashes, does the music still play?"**

If the answer is **NO**, the AI is in the wrong place.

## AI Does NOT Live in DSP

### ❌ Forbidden AI Operations

1. **Real-Time Audio Generation**
   ```cpp
   // WRONG: AI generating audio in real-time
   void processBlock(...) {
       auto samples = aiModel.generateSamples(...);  // FORBIDDEN
   }
   ```

2. **Audio Thread Inference**
   ```cpp
   // WRONG: Running ML model in audio thread
   void processBlock(...) noexcept {
       auto prediction = model.predict(audioBuffer);  // FORBIDDEN
   }
   ```

3. **Dynamic Memory Allocation**
   ```cpp
   // WRONG: AI allocating during processing
   void processBlock(...) {
       auto features = extractFeatures(buffer);  // May allocate
       auto result = model.predict(features);  // FORBIDDEN
   }
   ```

## AI as Control/Analysis Layer

### ✅ Correct AI Placement

AI operates **above** DSP, not beside it:

```
[ UI Layer ]
   ↑        ↓
[ AI CONTROL / ANALYSIS ]
   ↑        ↓
[ PARAMETER / STATE LAYER ]
   ↑
[ DSP CORE ]
```

### AI Responsibilities

1. **Intent Interpretation**
   - User text → emotional intent
   - Emotional intent → musical parameters
   - Natural language → structured commands

2. **Pattern Analysis**
   - Analyze existing audio/MIDI
   - Detect patterns and structures
   - Extract features for parameter mapping

3. **Suggestion Generation**
   - Generate parameter suggestions
   - Create automation curves
   - Propose structural decisions

4. **High-Level Instructions**
   - Design chord progressions
   - Plan song structure
   - Suggest rule-breaking applications

## AI Input/Output Contract

### ✅ Allowed AI Inputs

1. **Offline Audio**
   ```python
   # AI can analyze pre-rendered audio
   audio_file = load_audio("reference.wav")
   features = extract_features(audio_file)
   intent = ai_model.analyze_emotion(features)
   ```

2. **Pre-computed Features**
   ```python
   # AI receives spectral summaries, not live streams
   spectrum = compute_spectrum(audio_buffer)  # Pre-computed
   embedding = ai_model.embed(spectrum)
   ```

3. **MIDI Abstractions**
   ```python
   # AI works with symbolic music
   midi_sequence = load_midi("song.mid")
   structure = ai_model.analyze_structure(midi_sequence)
   ```

4. **Emotional/Harmonic Descriptors**
   ```python
   # AI works with high-level descriptors
   emotion_vector = [valence, arousal, dominance]
   parameters = ai_model.emotion_to_parameters(emotion_vector)
   ```

5. **User History & Metadata**
   ```python
   # AI can use session context
   user_preferences = load_user_history()
   suggestion = ai_model.suggest_parameters(user_preferences)
   ```

### ✅ Allowed AI Outputs

1. **Parameter Targets**
   ```python
   # AI outputs parameter values
   parameters = {
       'cutoff': 1200.0,
       'resonance': 0.7,
       'tempo': 90
   }
   ```

2. **Automation Curves**
   ```python
   # AI outputs parameter automation
   automation = [
       {'time': 0.0, 'cutoff': 500},
       {'time': 2.0, 'cutoff': 2000},
       {'time': 4.0, 'cutoff': 500}
   ]
   ```

3. **Structural Decisions**
   ```python
   # AI outputs high-level structure
   structure = {
       'sections': ['intro', 'verse', 'chorus'],
       'chord_progression': ['I', 'V', 'vi', 'IV'],
       'rule_breaks': ['modal_interchange']
   }
   ```

4. **High-Level Instructions**
   ```python
   # AI outputs instructions for generators
   instructions = {
       'harmony': generate_harmony_progression(),
       'melody': generate_melody_contour(),
       'groove': generate_groove_pattern()
   }
   ```

### ❌ Forbidden AI Outputs

1. **Audio Samples**
   ```python
   # WRONG: AI generating audio samples
   samples = ai_model.generate_audio(...)  # FORBIDDEN
   ```

2. **Real-Time Buffers**
   ```python
   # WRONG: AI outputting real-time audio
   buffer = ai_model.process_realtime(...)  # FORBIDDEN
   ```

## AI Module Structure

### Directory Organization

```
ai/
├── harmony_model/
│   ├── model.py           # Model definition
│   ├── inference.py        # Inference logic
│   └── parameters.py       # Input/output types
├── emotion_model/
│   ├── model.py
│   ├── inference.py
│   └── parameters.py
├── gesture_model/
│   ├── model.py
│   ├── inference.py
│   └── parameters.py
└── structure_model/
    ├── model.py
    ├── inference.py
    └── parameters.py
```

### Module Requirements

Each AI module must have:

1. **Clear Inputs** - Well-defined input types
2. **Clear Outputs** - Well-defined output types
3. **No UI Code** - Pure inference logic
4. **No DSP Code** - No audio processing
5. **No Host Assumptions** - Framework-agnostic

### Module Interface

```python
# ai/harmony_model/inference.py
from typing import Protocol
from dataclasses import dataclass

@dataclass
class HarmonyInput:
    """Input to harmony model"""
    emotion_vector: list[float]
    key: str
    mode: str
    duration_bars: int

@dataclass
class HarmonyOutput:
    """Output from harmony model"""
    chord_progression: list[str]
    voicings: list[list[int]]
    rule_breaks: list[str]

class HarmonyModel(Protocol):
    """Harmony model interface"""
    
    def predict(self, input: HarmonyInput) -> HarmonyOutput:
        """Generate harmony from input"""
        ...
    
    def load(self, checkpoint_path: str) -> None:
        """Load model weights"""
        ...
```

## AI Execution Context

### Plugin Context

**AI is offline or deferred:**

```cpp
// plugins/PluginProcessor.cpp
class PluginProcessor {
    // AI runs on background thread
    std::thread ai_thread_;
    RTQueue<AISuggestion> suggestions_;
    
    void onPresetLoad() {
        // Trigger AI analysis (non-real-time)
        ai_thread_ = std::thread([this]() {
            auto suggestion = ai_model.analyzePreset();
            suggestions_.push(suggestion);
        });
    }
    
    void processBlock(...) noexcept {
        // Audio thread: check for AI suggestions (non-blocking)
        if (auto suggestion = suggestions_.try_pop()) {
            applySuggestion(*suggestion);  // Update parameters
        }
        // Continue audio processing regardless of AI
    }
};
```

### Standalone App Context

**AI can run continuously:**

```python
# apps/macOS/AIController.swift (via Python bridge)
class AIController {
    func analyzeSession() {
        // Continuous analysis (non-real-time)
        let audioFeatures = extractFeaturesFromSession()
        let emotion = emotionModel.predict(audioFeatures)
        let parameters = parameterMapper.map(emotion)
        
        // Update parameters (async, non-blocking)
        audioEngine.setParameters(parameters)
    }
    
    func generateStructure() {
        // Long-running reasoning (background)
        let structure = structureModel.generate(
            emotion: currentEmotion,
            duration: songDuration
        )
        
        // Apply structure (user approval required)
        presentStructureToUser(structure)
    }
}
```

## Real-Time Safety

### ✅ Safe AI Patterns

1. **Offline Analysis**
   ```python
   # AI analyzes pre-rendered audio
   audio = load_audio_file("reference.wav")
   analysis = model.analyze(audio)  # Safe, offline
   ```

2. **Background Thread**
   ```cpp
   // AI runs on separate thread
   std::thread([this]() {
       auto result = model.predict(input);
       // Push result to lock-free queue
       results_queue.push(result);
   });
   ```

3. **Deferred Processing**
   ```python
   # AI processes when user triggers
   def onUserRequest():
       # Non-real-time, user-initiated
       result = model.generate()
       apply_result(result)
   ```

### ❌ Unsafe AI Patterns

1. **Audio Thread Inference**
   ```cpp
   // WRONG: AI in audio thread
   void processBlock(...) noexcept {
       auto result = model.predict(...);  // FORBIDDEN
   }
   ```

2. **Real-Time Generation**
   ```python
   # WRONG: AI generating in real-time
   def process_audio_stream(buffer):
       samples = model.generate_samples(buffer)  # FORBIDDEN
   ```

3. **Blocking Operations**
   ```cpp
   // WRONG: AI blocking audio thread
   void processBlock(...) {
       auto result = model.predict(...);  // May block
       // FORBIDDEN
   }
   ```

## Model Size Constraints

### 16GB Mac Constraints

Based on architectural guidance:

| Model Size | Fit on 16GB | Use Case |
|-----------|-------------|----------|
| 7B-8B (quantized) | ✅ Yes | General purpose, chat, code |
| 13B (quantized) | ⚠️ Borderline | Advanced inference |
| 20B+ (unquantized) | ❌ No | Not realistic |

### Recommended Models

1. **Mistral 7B** (Apache-2.0) - ✅ Fits comfortably
2. **LLaMA 3 8B** (quantized) - ✅ Fits comfortably
3. **Quantized 13B** - ⚠️ Usable with constraints

### Tools for Local Inference

- **llama.cpp** - Lightweight inference engine
- **Ollama** - Local model management
- **LM Studio** - UI with Metal/MLX acceleration
- **MLX** - Apple's ML framework

## AI Model Integration

### Model Loading

```python
# ai/model_loader.py
class ModelLoader:
    def load_harmony_model(self) -> HarmonyModel:
        """Load harmony prediction model"""
        model = HarmonyModel()
        model.load("models/harmony_predictor.json")
        return model
    
    def load_emotion_model(self) -> EmotionModel:
        """Load emotion recognition model"""
        model = EmotionModel()
        model.load("models/emotion_recognizer.mlpackage")
        return model
```

### Model Inference

```python
# ai/inference_engine.py
class InferenceEngine:
    def __init__(self):
        self.harmony_model = ModelLoader().load_harmony_model()
        self.emotion_model = ModelLoader().load_emotion_model()
    
    def process_intent(self, intent: UserIntent) -> Parameters:
        """Convert user intent to parameters"""
        # AI interprets intent
        emotion = self.emotion_model.predict(intent.text)
        
        # AI generates parameters
        parameters = self.harmony_model.predict(emotion)
        
        return parameters
```

## Testing AI Boundaries

### Test 1: AI Crash Doesn't Stop Audio

```python
# Test: Audio continues if AI crashes
def test_ai_crash_resilience():
    # Simulate AI crash
    ai_model.crash()
    
    # Audio should still play
    assert audio_engine.is_playing()
    assert audio_engine.has_output()
```

### Test 2: AI Never in Audio Thread

```cpp
// Test: No AI calls in audio thread
void test_no_ai_in_audio_thread() {
    // Compile-time check: AI functions not marked noexcept
    static_assert(!noexcept(ai_model.predict(...)));
    
    // Runtime check: No AI calls in processBlock
    // (Use static analysis or runtime instrumentation)
}
```

### Test 3: AI Outputs Parameters Only

```python
# Test: AI outputs parameters, not samples
def test_ai_output_format():
    result = ai_model.predict(input)
    
    # Should be parameters/structure
    assert hasattr(result, 'parameters')
    assert hasattr(result, 'structure')
    
    # Should NOT be audio samples
    assert not hasattr(result, 'samples')
    assert not hasattr(result, 'audio_buffer')
```

## Migration Checklist

If existing AI violates boundaries:

- [ ] Move all AI out of audio thread
- [ ] Verify AI never generates audio samples
- [ ] Ensure AI outputs parameters/structure only
- [ ] Test: AI crash doesn't stop audio
- [ ] Document AI input/output contracts
- [ ] Implement background thread execution
- [ ] Add error handling for AI failures

## References

- `ARCHITECTURAL_BOUNDARY_COMPLIANCE_REPORT.md` - Full boundary analysis
- `ml/ML_FRAMEWORKS_EVALUATION.md` - ML framework evaluation
- `05_AI_ML_VISIBILITY.md` - AI visibility specifications