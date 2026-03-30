# Full Pipeline Documentation

**Date:** 2026-01-22
**Purpose:** Complete documentation of music and vocal generation pipelines

## Overview

This document describes the complete pipelines from user input to final output for both music and vocal generation in the KmiDi system.

---

## Music Generation Pipeline

### Complete Flow

```
User Input
    ↓
[Text/Emotion/Wound/Journey]
    ↓
IntentPipeline::process()
    ↓
IntentResult (Legacy) OR IntentFrame (IR v1)
    ↓
KellyBrain::generateMidi()
    ↓
MidiGenerator::generate()
    ↓
[Multiple Engines]
    ├── ChordGenerator
    ├── MelodyEngine
    ├── BassEngine
    ├── PadEngine
    ├── StringEngine
    ├── CounterMelodyEngine
    ├── RhythmEngine
    ├── DrumGrooveEngine
    ├── FillEngine
    ├── TensionEngine
    ├── DynamicsEngine
    └── GrooveEngine
    ↓
GeneratedMidi
    ├── chords (vector<Chord>)
    ├── melody (vector<MidiNote>)
    ├── bass (vector<MidiNote>)
    ├── pads (vector<MidiNote>)
    ├── strings (vector<MidiNote>)
    ├── counterMelody (vector<MidiNote>)
    ├── rhythm (vector<MidiNote>)
    ├── drums (vector<MidiNote>)
    ├── fills (vector<MidiNote>)
    ├── tempoBpm (float)
    ├── bars (int)
    ├── key (int)
    ├── mode (int)
    └── lengthInBeats (double)
    ↓
Export/Render
    ├── MIDI File (.mid)
    ├── Audio Render
    └── DAW Integration
```

### Detailed Steps

#### 1. Input Processing

**Entry Points:**
```cpp
// Text input
IntentResult result = brain.fromText("I feel lost and alone");
IntentFrame frame = brain.fromTextToIntentFrame("I feel lost and alone");

// Emotion input
IntentResult result = brain.fromEmotion("melancholy", 0.8f);
IntentFrame frame = brain.fromEmotionToIntentFrame("melancholy", 0.8f);

// Wound input
IntentResult result = brain.fromWound(wound);
IntentFrame frame = brain.fromWoundToIntentFrame(wound);

// Journey input (SideA → SideB)
IntentResult result = brain.fromJourney(current, desired);
IntentFrame frame = brain.fromJourneyToIntentFrame(current, desired);
```

**Internal Processing:**
```
Text/Emotion/Wound
    ↓
EmotionThesaurus::lookup()  // Maps to emotion nodes
    ↓
IntentPipeline::process()   // Processes emotion → musical parameters
    ↓
IntentResult OR IntentFrame
```

#### 2. Intent Processing

**IntentPipeline Components:**
- **Emotion Analysis:** Maps emotion to VAD (Valence, Arousal, Dominance)
- **Rule Breaks:** Identifies musical rule violations for expression
- **Musical Parameters:** Derives tempo, key, mode, complexity, etc.
- **Constraints:** Applies user constraints and preferences

**Output:**
- `IntentResult` (legacy) - Concrete musical parameters
- `IntentFrame` (IR v1) - Bias-based representation

#### 3. IntentFrame Preparation (IR v1)

**Validation and Clamping:**
```cpp
void prepareIntentFrame(IntentFrame& frame) {
    // Uses Rust FFI validator
    clamp_intent_frame_ffi(&frame);  // Clamps all values to valid ranges

    // Optional: Validate after clamping
    // int validation_result = validate_intent_frame_ffi(&frame);
}
```

**What Gets Clamped:**
- Emotion: valence [-1.0, 1.0], arousal [0.0, 1.0], dominance [0.0, 1.0]
- Musical Intent: All biases normalized to [-1.0, 1.0] or [0.0, 1.0]
- Time Scope: Duration constraints
- Constraints: User-defined limits

#### 4. MIDI Generation

**KellyBrain::generateMidi()**
```cpp
GeneratedMidi generateMidi(const IntentResult& intent, int bars = 8);
GeneratedMidi generateMidiFromIntentFrame(const IntentFrame& frame, int bars = 8);
```

**Internal Processing:**
```cpp
// Derive complexity from intent
float melodic_complexity = (intent.melodicRange + intent.leapProbability) / 2.0f;
float rule_break_complexity = std::min(static_cast<float>(intent.ruleBreaks.size()) / 5.0f, 1.0f);
float harmonic_complexity = intent.allowChromaticism ? 0.7f : 0.3f;
const float complexity = (melodic_complexity * 0.4f + rule_break_complexity * 0.3f + harmonic_complexity * 0.3f);

// Derive feel from syncopation and swing
const float feel = std::clamp((intent.syncopationLevel * 0.6f + intent.swingAmount * 0.4f), 0.0f, 1.0f);

// Generate MIDI
GeneratedMidi result = midiGenerator_->generate(
    intentForGenerator, bars, complexity, humanize, feel, dynamics);
```

#### 5. MidiGenerator Orchestration

**MidiGenerator::generate()** orchestrates multiple engines:

1. **ChordGenerator**
   - Generates chord progressions
   - Based on key, mode, harmonic tension
   - Applies rule breaks for expression

2. **MelodyEngine**
   - Generates melodic lines
   - Based on melodic activity, contour variance
   - Respects key and mode constraints

3. **BassEngine**
   - Generates bass lines
   - Follows chord progressions
   - Applies groove and rhythm

4. **PadEngine**
   - Generates pad textures
   - Based on texture density
   - Harmonic support

5. **StringEngine**
   - Generates string arrangements
   - Orchestral textures
   - Dynamic expression

6. **CounterMelodyEngine**
   - Generates counter-melodies
   - Complements main melody
   - Harmonic interaction

7. **RhythmEngine**
   - Generates rhythmic patterns
   - Based on rhythmic density
   - Groove and feel

8. **DrumGrooveEngine**
   - Generates drum patterns
   - Based on groove strength
   - Humanization

9. **FillEngine**
   - Generates fills and transitions
   - Between sections
   - Dynamic variation

10. **TensionEngine**
    - Manages harmonic tension
    - Builds and releases
    - Emotional arc

11. **DynamicsEngine**
    - Manages dynamic range
    - Expression curves
    - Emotional intensity

12. **GrooveEngine**
    - Applies groove and humanization
    - Timing variations
    - Feel and swing

#### 6. Output Assembly

**GeneratedMidi Structure:**
```cpp
struct GeneratedMidi {
    // Layers
    std::vector<Chord> chords;
    std::vector<MidiNote> melody;
    std::vector<MidiNote> bass;
    std::vector<MidiNote> pads;
    std::vector<MidiNote> strings;
    std::vector<MidiNote> counterMelody;
    std::vector<MidiNote> rhythm;
    std::vector<MidiNote> drums;
    std::vector<MidiNote> fills;

    // Metadata
    float tempoBpm;
    int bars;
    int key;
    int mode;
    double lengthInBeats;
    float bpm;
};
```

#### 7. Export/Render

**Options:**
- **MIDI File:** Export to `.mid` format
- **Audio Render:** Render to audio file
- **DAW Integration:** Send to DAW via MIDI/OSC
- **Real-time Playback:** Play directly

---

## Vocal Generation Pipeline

### Complete Flow

```
Audio Input
    ↓
[Audio Samples + Sample Rate + Tempo]
    ↓
PRROTEngine::processAudioSegment()
    ↓
[Analysis Components]
    ├── SpectralAnalyzer (FFT)
    ├── PhonemeSegmenter
    ├── ArticulationAnalyzer
    ├── BreathDetector
    └── PitchTracker
    ↓
PhonemeControlData
    ├── phoneme_sequence (vector<PhonemeTiming>)
    ├── pitch_targets (vector<PitchTarget>)
    ├── vibrato_events (vector<VibratoParameters>)
    ├── breath_markers (vector<BreathMarker>)
    ├── midi_notes (vector<MidiNote>)
    ├── automation_envelopes (vector<AutomationEnvelope>)
    └── articulation_envelopes (vector<ArticulationEnvelope>)
    ↓
Export/DAW Integration
    ├── MIDI Export
    ├── Automation Curves
    ├── Control Data (JSON)
    └── DAW Plugin Control
```

### Detailed Steps

#### 1. Audio Input

**Entry Point:**
```cpp
PhonemeControlData processAudioSegment(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz,
    float tempo_bpm = 120.0f
) noexcept;
```

**Input Requirements:**
- Audio samples: Mono or stereo float array
- Sample rate: Typically 44.1kHz or 48kHz
- Tempo: BPM for timing calculations

#### 2. Spectral Analysis

**SpectralAnalyzer::analyze()**
```cpp
// Uses JUCE FFT (optimized, real-time safe)
juce::dsp::FFT fft(11);  // 2^11 = 2048 samples

// Compute FFT
fft.performRealOnlyForwardTransform(fft_data.data(), false);

// Extract features
- Formants (F1, F2, F3)
- Spectral centroid
- Spectral rolloff
- Zero crossing rate
- Energy distribution
```

**Output:**
- Formant frequencies (vowel identification)
- Spectral envelope (timbre)
- Energy distribution (articulation)

#### 3. Phoneme Segmentation

**PhonemeSegmenter::segment()**
```cpp
std::vector<PhonemeTiming> analyzePhonemes(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept;
```

**Process:**
1. **Spectral Analysis:** FFT for each frame
2. **Feature Extraction:** Formants, energy, zero-crossings
3. **Phoneme Classification:** Map features to phonemes
4. **Boundary Detection:** Identify phoneme boundaries
5. **Timing Calculation:** Start/end times for each phoneme

**Output:**
```cpp
struct PhonemeTiming {
    std::string phoneme;      // e.g., "AA", "IH", "S"
    float start_time_ms;      // Start time in milliseconds
    float duration_ms;        // Duration in milliseconds
    float confidence;         // Detection confidence [0.0, 1.0]
};
```

#### 4. Articulation Analysis

**ArticulationAnalyzer::analyze()**
- Analyzes articulation characteristics
- Extracts attack, sustain, release envelopes
- Identifies consonant/vowel transitions
- Measures articulation strength

**Output:**
```cpp
struct ArticulationEnvelope {
    float attack_time_ms;     // Attack phase duration
    float sustain_level;       // Sustain level [0.0, 1.0]
    float release_time_ms;     // Release phase duration
    float articulation_strength;  // Overall articulation [0.0, 1.0]
};
```

#### 5. Breath Detection

**BreathDetector::detect()**
```cpp
std::vector<BreathMarker> detectBreathMarkers(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept;
```

**Process:**
1. **Energy Analysis:** Detect low-energy regions
2. **Spectral Analysis:** Identify breath characteristics
3. **Timing Detection:** Find breath locations
4. **Intensity Calculation:** Measure breath intensity

**Output:**
```cpp
struct BreathMarker {
    float time_ms;             // Breath location
    float intensity;           // Breath intensity [0.0, 1.0]
    float duration_ms;         // Breath duration
    float confidence;          // Detection confidence
};
```

#### 6. Pitch Tracking

**PitchTracker::track()**
- Extracts fundamental frequency (F0)
- Converts to MIDI notes
- Calculates cents offset
- Tracks pitch over time

**Output:**
```cpp
struct PitchTarget {
    float time_ms;             // Target time
    int midi_note;             // MIDI note number
    float cents_offset;        // Cents offset from note
    float confidence;          // Tracking confidence
};
```

#### 7. Control Data Assembly

**PhonemeControlData Structure:**
```cpp
struct PhonemeControlData {
    // Metadata
    std::string output_id;
    std::string voice_profile_id;
    float tempo_bpm;
    float sample_rate_hz;

    // Phoneme sequence
    std::vector<PhonemeTiming> phoneme_sequence;

    // Pitch targets
    std::vector<PitchTarget> pitch_targets;

    // Vibrato parameters
    std::vector<std::pair<float, VibratoParameters>> vibrato_events;

    // Breath markers
    std::vector<BreathMarker> breath_markers;

    // MIDI notes
    std::vector<MidiNote> midi_notes;

    // Automation envelopes
    std::vector<AutomationEnvelope> automation_envelopes;

    // Articulation envelopes
    std::vector<ArticulationEnvelope> articulation_envelopes;
};
```

#### 8. Export/DAW Integration

**Options:**
- **MIDI Export:** Export pitch targets as MIDI
- **Automation Curves:** Export envelopes for DAW automation
- **Control Data (JSON):** Export complete control data
- **DAW Plugin Control:** Send to vocal synthesizer plugins

---

## Combined Pipeline (Music + Vocals)

### Full Workflow

```
User Input
    ├── Text: "I feel lost and alone"
    └── Audio: Vocal recording
    ↓
Parallel Processing
    ├── Music Pipeline
    │   └── → GeneratedMidi
    └── Vocal Pipeline
        └── → PhonemeControlData
    ↓
Synchronization
    ├── Align MIDI and vocal timing
    ├── Match tempo
    └── Coordinate musical and vocal expression
    ↓
Combined Output
    ├── MIDI (music + vocal pitch)
    ├── Control Data (vocal articulation)
    └── Synchronized playback
```

### Example: Full Song Generation

```cpp
// 1. Generate music from text
KellyBrain brain;
brain.initialize("./data");

IntentFrame musicIntent = brain.fromTextToIntentFrame("I feel lost and alone");
GeneratedMidi music = brain.generateMidiFromIntentFrame(musicIntent, 16);

// 2. Process vocal audio
PRROTEngine engine;
engine.initialize();
engine.loadVoiceProfile(voiceProfile);

std::vector<float> vocalAudio = loadAudioFile("vocal.wav");
PhonemeControlData vocals = engine.processAudioSegment(
    vocalAudio.data(),
    vocalAudio.size(),
    44100.0f,
    music.tempoBpm
);

// 3. Synchronize and combine
synchronizeMusicAndVocals(music, vocals);

// 4. Export
exportCombinedOutput(music, vocals, "output.mid");
```

---

## Performance Characteristics

### Music Generation

**Timing:**
- Intent processing: <10ms
- MIDI generation: 10-100ms
- ML enhancement (standalone): 50-500ms
- **Total:** 100ms - 1s (acceptable for standalone)

**Real-time Constraints:**
- Plugin mode: Must be <10ms (limited features)
- Standalone mode: No constraints (full features)

### Vocal Generation

**Timing:**
- Audio analysis: 10-50ms per second of audio
- Phoneme segmentation: 5-20ms per second
- ML enhancement (standalone): 1-10s per second
- **Total:** 1-10s per second of audio (acceptable for standalone)

**Real-time Constraints:**
- Plugin mode: Must be <10ms per buffer (basic analysis)
- Standalone mode: No constraints (full analysis)

---

## ML Model Integration Points

### Music Generation

**Available Models (Standalone):**
1. **Emotion Recognizer** - Audio/text → Emotion embedding
2. **Melody Transformer** - Emotion → Note probabilities
3. **Harmony Predictor** - Context → Chord predictions
4. **Dynamics Engine** - Emotion → Expression parameters
5. **Groove Predictor** - Emotion → Groove/timing

**Integration:**
```python
# Python side (standalone only)
from penta_core.ml import inference

# Enhance IntentFrame with ML
enhanced_frame = enhance_with_ml_models(intent_frame)
```

### Vocal Generation

**Available Models (Standalone):**
1. **Phoneme Aligner** - 3B Q4 model for high-quality alignment
2. **Timbre Extractor** - Wav2Vec2/Whisper for timbre features

**Integration:**
```python
# Python side (standalone only)
from prrot import phoneme_aligner, timbre_embeddings

# Enhanced phoneme alignment
aligner = phoneme_aligner.PhonemeAligner()
aligned = aligner.align_phonemes(audio, transcript)

# Timbre extraction
extractor = timbre_embeddings.TimbreEmbeddingExtractor()
timbre = extractor.extract_embedding(audio, sample_rate)
```

---

## Code Examples

### Music Generation Example

```cpp
#include "engine/KellyBrain.h"

using namespace kelly;

// Initialize
KellyBrain brain;
brain.initialize("./data");

// Generate from text
IntentFrame frame = brain.fromTextToIntentFrame("I feel joyful and energetic");
prepareIntentFrame(frame);  // Validate and clamp

GeneratedMidi music = brain.generateMidiFromIntentFrame(frame, 16);

// Access generated content
for (const auto& chord : music.chords) {
    std::cout << "Chord: " << chord.root << " " << chord.quality << std::endl;
}

for (const auto& note : music.melody) {
    std::cout << "Note: " << note.pitch << " @ " << note.startTime << std::endl;
}

// Export
exportMidiToFile(music, "output.mid");
```

### Vocal Generation Example

```cpp
#include "prrot/PRROTEngine.h"

using namespace prrot;

// Initialize
PRROTEngine engine;
engine.initialize();
engine.loadVoiceProfile(voiceProfile);

// Process audio
std::vector<float> audio = loadAudioFile("vocal.wav");
PhonemeControlData vocals = engine.processAudioSegment(
    audio.data(),
    audio.size(),
    44100.0f,
    120.0f
);

// Access control data
for (const auto& phoneme : vocals.phoneme_sequence) {
    std::cout << "Phoneme: " << phoneme.phoneme
              << " @ " << phoneme.start_time_ms << "ms" << std::endl;
}

for (const auto& pitch : vocals.pitch_targets) {
    std::cout << "Pitch: MIDI " << pitch.midi_note
              << " @ " << pitch.time_ms << "ms" << std::endl;
}

// Export
exportControlData(vocals, "vocal_control.json");
```

### Combined Example

```cpp
// Generate music
KellyBrain brain;
brain.initialize("./data");
GeneratedMidi music = brain.generateMidiFromText("I feel lost", 16);

// Process vocals
PRROTEngine engine;
engine.initialize();
PhonemeControlData vocals = engine.processAudioSegment(audio, size, 44100.0f, music.tempoBpm);

// Synchronize
synchronizeMusicAndVocals(music, vocals);

// Export combined
exportCombinedOutput(music, vocals, "song.mid");
```

---

## Pipeline Components Summary

### Music Generation

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| IntentPipeline | Process emotion → intent | Text/Emotion/Wound | IntentResult/IntentFrame |
| KellyBrain | High-level API | IntentResult/IntentFrame | GeneratedMidi |
| MidiGenerator | Orchestrate engines | IntentResult/IntentFrame | GeneratedMidi |
| ChordGenerator | Generate chords | Intent | Chords |
| MelodyEngine | Generate melody | Intent | Melody notes |
| BassEngine | Generate bass | Intent | Bass notes |
| PadEngine | Generate pads | Intent | Pad notes |
| StringEngine | Generate strings | Intent | String notes |
| CounterMelodyEngine | Generate counter-melody | Intent | Counter-melody notes |
| RhythmEngine | Generate rhythm | Intent | Rhythm notes |
| DrumGrooveEngine | Generate drums | Intent | Drum notes |
| FillEngine | Generate fills | Intent | Fill notes |
| TensionEngine | Manage tension | Intent | Tension curves |
| DynamicsEngine | Manage dynamics | Intent | Dynamic curves |
| GrooveEngine | Apply groove | Intent | Timing variations |

### Vocal Generation

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| PRROTEngine | Main orchestrator | Audio samples | PhonemeControlData |
| SpectralAnalyzer | FFT analysis | Audio samples | Formants, spectral features |
| PhonemeSegmenter | Segment phonemes | Audio samples | Phoneme sequence |
| ArticulationAnalyzer | Analyze articulation | Audio samples | Articulation envelopes |
| BreathDetector | Detect breaths | Audio samples | Breath markers |
| PitchTracker | Track pitch | Audio samples | Pitch targets |

---

## Conclusion

The KmiDi system provides complete pipelines for both music and vocal generation:

✅ **Music:** Text/Emotion → Intent → MIDI (complete arrangements)
✅ **Vocals:** Audio → Analysis → Control Data (complete vocal control)
✅ **Combined:** Synchronized music + vocals
✅ **Standalone:** Full ML model integration
✅ **Real-time:** RT-safe components for plugins

Both pipelines are production-ready and can be used independently or combined for full song generation.

---

**See Also:**
- `docs/STANDALONE_GENERATION_ARCHITECTURE.md` - Architecture details
- `docs/STANDALONE_GENERATION_OPTIMIZATION.md` - Optimization guide
- `docs/TEST_INTEGRATION_COMPLETE.md` - Test integration
