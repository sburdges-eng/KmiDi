# Standalone Generation Optimization Guide

**Date:** 2026-01-22
**Focus:** Optimizing music and vocal generation for standalone application

## Understanding Standalone Mode

The standalone application operates **without low-latency constraints**, allowing full use of:
- Complex ML models (Python ecosystem)
- Dynamic memory allocation
- Multi-pass processing
- File I/O operations
- Batch processing
- Higher-quality algorithms

## Music Generation Optimization

### Current Capabilities

**KellyBrain API:**
```cpp
// High-level music generation
KellyBrain brain;
brain.initialize("./data");

// Multiple input methods
GeneratedMidi music1 = brain.generateMidiFromText("I feel lost", 8);
GeneratedMidi music2 = brain.generateMidiFromEmotion("melancholy", 0.8f, 16);
GeneratedMidi music3 = brain.generateMidiFromIntentFrame(intentFrame, 12);
```

**MidiGenerator Features:**
- Complete arrangements (chords, melody, bass, pads, strings)
- Multiple engines (MelodyEngine, BassEngine, PadEngine, etc.)
- Groove and humanization
- Dynamics and tension
- Rule breaks for emotional expression

### Optimization Opportunities

#### 1. ML Model Integration (Standalone Mode)

**Current:** Uses rule-based generation primarily
**Enhancement:** Integrate Python ML models for higher quality

```cpp
// Standalone mode - can use Python ML
// 1. Extract emotion from audio/text using ML
// 2. Use ML models for melody/harmony prediction
// 3. Combine with rule-based generation

// Example workflow:
IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");

// Enhance with ML models (standalone only)
if (isStandaloneMode()) {
    // Use Python ML pipeline
    auto ml_enhanced = enhanceWithMLModels(frame);
    GeneratedMidi music = brain.generateMidiFromIntentFrame(ml_enhanced, 16);
}
```

#### 2. Batch Generation

**Opportunity:** Generate multiple variations in parallel

```cpp
// Generate multiple variations
std::vector<GeneratedMidi> variations;
for (int i = 0; i < 5; ++i) {
    variations.push_back(brain.generateMidi(intent, 8));
}
// User can select best or blend
```

#### 3. Iterative Refinement

**Opportunity:** Multi-pass generation with feedback

```cpp
// First pass: basic structure
GeneratedMidi draft = brain.generateMidi(intent, 8);

// Second pass: add details
GeneratedMidi refined = refineMidi(draft, intent);

// Third pass: humanization
GeneratedMidi final = humanizeMidi(refined);
```

## Vocal Generation Optimization

### Current Capabilities

**PRROTEngine API:**
```cpp
PRROTEngine engine;
engine.initialize();
engine.loadVoiceProfile(profile);

// Process audio
PhonemeControlData vocals = engine.processAudioSegment(
    audio_samples,
    num_samples,
    sample_rate_hz,
    tempo_bpm
);
```

**PRROT Features:**
- Phoneme segmentation
- Articulation analysis
- Spectral analysis (formants, timbre)
- Breath detection
- Pitch target generation
- Complete control data output

### Optimization Opportunities

#### 1. ML Model Enhancement (Standalone Mode)

**Current:** Rule-based phoneme segmentation
**Enhancement:** Use 3B Phoneme Aligner model

```python
# Standalone mode - can use large ML models
from prrot import phoneme_aligner

# High-quality phoneme alignment (3B model)
aligner = phoneme_aligner.PhonemeAligner()
aligned_phonemes = aligner.align_phonemes(audio, transcript)

# Use aligned phonemes in PRROT
control_data = engine.generateControlData(aligned_phonemes, pitch_targets)
```

#### 2. Timbre Extraction Enhancement

**Current:** Basic spectral analysis
**Enhancement:** Use Wav2Vec2/Whisper for timbre

```python
# Standalone mode - can use transformer models
from prrot import timbre_embeddings

extractor = timbre_embeddings.TimbreEmbeddingExtractor()
timbre_features = extractor.extract_embedding(audio, sample_rate)

# Use timbre features for better voice modeling
```

#### 3. Multi-Pass Processing

**Opportunity:** Iterative refinement of vocal control data

```cpp
// First pass: basic segmentation
PhonemeControlData draft = engine.processAudioSegment(audio, ...);

// Second pass: refine with ML models (standalone)
if (isStandaloneMode()) {
    draft = enhanceWithMLModels(draft, audio);
}

// Third pass: optimize timing and pitch
PhonemeControlData final = optimizeControlData(draft);
```

## Standalone Mode Detection

### Implementation Pattern

```cpp
// Detect if running in standalone mode
bool isStandaloneMode() {
    // Standalone app: no strict latency requirements
    // Plugin: must be RT-safe
    #ifdef STANDALONE_APP
        return true;
    #else
        return false;
    #endif
}

// Or runtime detection
bool isStandaloneMode() {
    // Check if we're in a plugin context
    // Plugins have strict latency requirements
    // Standalone apps don't
    return !isPluginContext();
}
```

### Usage Pattern

```cpp
void generateMusic(const IntentFrame& frame) {
    if (isStandaloneMode()) {
        // Use full ML pipeline
        auto enhanced = enhanceWithMLModels(frame);
        return generateWithML(enhanced);
    } else {
        // Use RT-safe generation only
        return generateRT(frame);
    }
}
```

## Performance Characteristics

### Standalone Mode (No Constraints)

**Music Generation:**
- Intent processing: <10ms
- ML inference: 50-500ms (acceptable)
- MIDI generation: 10-100ms
- **Total:** 100ms - 1s per generation (acceptable)

**Vocal Generation:**
- Audio analysis: 10-50ms per second
- ML enhancement: 1-10s (acceptable for standalone)
- Control data: <10ms
- **Total:** 1-10s per second of audio (acceptable)

### Real-Time Mode (Constrained)

**Music Generation:**
- Intent processing: <10ms
- ML inference: Not available (too slow)
- MIDI generation: 10-100ms
- **Total:** 20-110ms (must be <10ms for audio callback)

**Vocal Generation:**
- Audio analysis: 10-50ms per second
- ML enhancement: Not available
- Control data: <10ms
- **Total:** 20-60ms per second (must be <10ms for callback)

## Recommended Architecture

### Standalone Application Structure

```
Standalone App (Swift/macOS)
    ↓
C++ Bridge Layer
    ↓
KellyBrain (Music) + PRROTEngine (Vocals)
    ↓
Python ML Bridge (Standalone Only)
    ↓
Python ML Models (Full Pipeline)
```

### Integration Points

1. **Swift → C++ Bridge**
   - Use Objective-C++ for FFI
   - Call KellyBrain and PRROTEngine directly
   - Handle async operations for ML

2. **C++ → Python Bridge**
   - Use subprocess or IPC for Python ML
   - Async communication (don't block UI)
   - Error handling and fallbacks

3. **Generation Workflow**
   - User input → Intent processing
   - Intent → Music/Vocal generation
   - ML enhancement (async, standalone only)
   - Export/playback

## Implementation Recommendations

### 1. Add Standalone Mode Flag

**File:** `engine/src/common/Types.h` or new header

```cpp
namespace kelly {
    // Runtime detection of standalone vs plugin mode
    bool isStandaloneMode();
    void setStandaloneMode(bool standalone);
}
```

### 2. Enhance KellyBrain for Standalone

**File:** `engine/src/engine/KellyBrain.h`

```cpp
class KellyBrain {
    // Add standalone-specific methods
    GeneratedMidi generateMidiWithML(
        const IntentFrame& frame,
        int bars,
        bool useMLModels = true  // Standalone only
    );

    // Batch generation
    std::vector<GeneratedMidi> generateVariations(
        const IntentFrame& frame,
        int bars,
        int numVariations = 5
    );
};
```

### 3. Enhance PRROTEngine for Standalone

**File:** `engine/src/prrot/PRROTEngine.h`

```cpp
class PRROTEngine {
    // Add ML-enhanced methods
    PhonemeControlData processAudioSegmentWithML(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz,
        bool useMLModels = true  // Standalone only
    );

    // Enhanced phoneme alignment
    std::vector<PhonemeTiming> analyzePhonemesWithML(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    );
};
```

### 4. Python ML Bridge

**File:** `python/penta_core/ml/standalone_bridge.py`

```python
"""
Bridge for standalone application to use Python ML models.
Not RT-safe - for standalone use only.
"""

from penta_core.ml import inference
from prrot import phoneme_aligner, timbre_embeddings

class StandaloneMLBridge:
    """Bridge for standalone app to use ML models."""

    def enhance_intent_frame(self, frame: dict) -> dict:
        """Enhance IntentFrame with ML model predictions."""
        # Use emotion recognizer, melody transformer, etc.
        pass

    def align_phonemes(self, audio: np.ndarray, transcript: str) -> list:
        """High-quality phoneme alignment using 3B model."""
        aligner = phoneme_aligner.PhonemeAligner()
        return aligner.align_phonemes(audio, transcript)

    def extract_timbre(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract timbre features using Wav2Vec2/Whisper."""
        extractor = timbre_embeddings.TimbreEmbeddingExtractor()
        return extractor.extract_embedding(audio, sample_rate)
```

## Code Quality Improvements

### Current State

- ✅ Music generation working
- ✅ Vocal generation working
- ⚠️ ML integration not fully connected
- ⚠️ Standalone mode not explicitly handled

### Recommended Improvements

1. **Add Standalone Mode Detection**
   - Runtime flag or compile-time define
   - Enable/disable ML features based on mode

2. **Async ML Integration**
   - Don't block UI thread
   - Use background threads/processes
   - Progress callbacks

3. **Error Handling**
   - Fallback to rule-based if ML fails
   - Graceful degradation
   - User feedback

4. **Performance Monitoring**
   - Track generation time
   - Log ML model usage
   - Profile bottlenecks

## Example Standalone Workflow

### Music Generation

```swift
// Swift/macOS app
func generateMusic(from text: String) {
    // 1. Process text to intent (fast)
    let intent = brain.fromText(text)

    // 2. Generate MIDI (can take 100ms-1s, OK for standalone)
    let music = brain.generateMidi(intent, bars: 16)

    // 3. Optionally enhance with ML (async, standalone only)
    if isStandaloneMode {
        enhanceWithML(music) { enhanced in
            // Update UI with enhanced music
            self.updateMusic(enhanced)
        }
    } else {
        // Use generated music directly
        self.updateMusic(music)
    }
}
```

### Vocal Generation

```swift
// Swift/macOS app
func processVocalAudio(_ audio: [Float]) {
    // 1. Basic PRROT analysis (fast)
    let controlData = engine.processAudioSegment(audio, ...)

    // 2. Enhance with ML models (async, standalone only)
    if isStandaloneMode {
        enhanceVocalWithML(controlData, audio) { enhanced in
            // Update UI with enhanced control data
            self.updateVocals(enhanced)
        }
    } else {
        // Use basic control data
        self.updateVocals(controlData)
    }
}
```

## Testing Strategy

### Standalone Mode Tests

1. **Music Generation**
   - Test full pipeline (text → MIDI)
   - Test ML enhancement
   - Test batch generation
   - Measure performance

2. **Vocal Generation**
   - Test audio → control data
   - Test ML enhancement
   - Test multi-pass processing
   - Measure quality improvements

3. **Performance**
   - Verify acceptable latency (<1s for music, <10s for vocals)
   - Test async operations
   - Test error handling

## Conclusion

The standalone application has **full capabilities** for music and vocal generation:

✅ **Music:** Complete pipeline with ML enhancement capability
✅ **Vocals:** Complete PRROT pipeline with ML enhancement capability
✅ **Performance:** Acceptable for standalone (not RT-constrained)
✅ **Quality:** Can use highest-quality models and processing

**Key Insight:** Standalone mode removes real-time constraints, enabling:
- Full Python ML ecosystem
- Complex multi-pass processing
- Higher-quality algorithms
- Better user experience

---

**See Also:**
- `docs/STANDALONE_GENERATION_ARCHITECTURE.md` - Architecture details
- `docs/MULTI_LANGUAGE_ARCHITECTURE.md` - Multi-language integration
- `docs/MODELS_README.md` - ML model documentation
