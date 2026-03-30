# Pipeline Visual Summary

**Date:** 2026-01-22
**Quick Reference:** Visual overview of complete pipelines

---

## Music Generation Pipeline (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│  Text: "I feel lost"  |  Emotion: "melancholy"  |  Wound        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTENT PROCESSING                             │
│  IntentPipeline::process()                                       │
│  ├── EmotionThesaurus lookup                                    │
│  ├── Emotion → VAD mapping                                      │
│  ├── Rule breaks identification                                 │
│  └── Musical parameters derivation                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTENT REPRESENTATION                         │
│  IntentResult (Legacy)  OR  IntentFrame (IR v1)                  │
│  └── prepareIntentFrame() - Validate & clamp                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MIDI GENERATION                               │
│  KellyBrain::generateMidi()                                      │
│  └── MidiGenerator::generate()                                   │
│      ├── Derive complexity from intent                          │
│      ├── Derive feel from syncopation/swing                     │
│      └── Orchestrate engines                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MIDI ENGINES                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ ChordGen     │  │ MelodyEngine │  │ BassEngine   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ PadEngine    │  │ StringEngine│  │ RhythmEngine │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ DrumEngine   │  │ FillEngine   │  │ TensionEngine│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ DynamicsEng  │  │ GrooveEngine │                           │
│  └──────────────┘  └──────────────┘                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATED MIDI                                │
│  GeneratedMidi                                                   │
│  ├── chords, melody, bass, pads, strings                        │
│  ├── counterMelody, rhythm, drums, fills                        │
│  └── tempoBpm, bars, key, mode, lengthInBeats                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│  MIDI File  |  Audio Render  |  DAW Integration                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Vocal Generation Pipeline (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                               │
│  Audio Samples + Sample Rate + Tempo                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRROT ENGINE                                  │
│  PRROTEngine::processAudioSegment()                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS COMPONENTS                           │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ SpectralAnalyzer │  │ PhonemeSegmenter │                    │
│  │ (JUCE FFT)       │  │ (Segmentation)    │                    │
│  └──────────────────┘  └──────────────────┘                    │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Articulation     │  │ BreathDetector   │                    │
│  │ Analyzer         │  │ (Breath markers) │                    │
│  └──────────────────┘  └──────────────────┘                    │
│  ┌──────────────────┐                                           │
│  │ PitchTracker     │                                           │
│  │ (F0 extraction)  │                                           │
│  └──────────────────┘                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL DATA ASSEMBLY                         │
│  PhonemeControlData                                              │
│  ├── phoneme_sequence (timing, confidence)                      │
│  ├── pitch_targets (MIDI notes, cents offset)                    │
│  ├── vibrato_events (time-varying parameters)                   │
│  ├── breath_markers (location, intensity)                       │
│  ├── midi_notes (shaped from phonemes)                          │
│  ├── automation_envelopes (formant, timbre)                     │
│  └── articulation_envelopes (attack, sustain, release)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│  MIDI Export  |  Automation Curves  |  Control Data (JSON)     │
│  DAW Plugin Control                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Combined Pipeline (Music + Vocals)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│  Text: "I feel lost"  +  Audio: Vocal recording                 │
└────────────┬──────────────────────────────┬─────────────────────┘
             │                              │
             ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │ Music Pipeline  │            │ Vocal Pipeline  │
    │ (see above)     │            │ (see above)     │
    └────────┬────────┘            └────────┬────────┘
             │                              │
             ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │ GeneratedMidi   │            │ PhonemeControl  │
    │                 │            │ Data            │
    └────────┬────────┘            └────────┬────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │      SYNCHRONIZATION           │
            │  ├── Align timing              │
            │  ├── Match tempo               │
            │  └── Coordinate expression     │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │      COMBINED OUTPUT           │
            │  ├── MIDI (music + vocal pitch)│
            │  ├── Control Data (articulation)│
            │  └── Synchronized playback     │
            └───────────────────────────────┘
```

---

## Data Flow Diagram

### Music Generation Data Flow

```
Text/Emotion
    │
    ├─→ EmotionThesaurus → EmotionNode
    │
    ├─→ IntentPipeline → IntentResult/IntentFrame
    │
    ├─→ prepareIntentFrame() → Validated IntentFrame
    │
    ├─→ KellyBrain::generateMidi() → GeneratedMidi
    │
    └─→ Export/Render
```

### Vocal Generation Data Flow

```
Audio Samples
    │
    ├─→ SpectralAnalyzer → Formants, Spectral Features
    │
    ├─→ PhonemeSegmenter → Phoneme Sequence
    │
    ├─→ ArticulationAnalyzer → Articulation Envelopes
    │
    ├─→ BreathDetector → Breath Markers
    │
    ├─→ PitchTracker → Pitch Targets
    │
    └─→ PRROTEngine → PhonemeControlData
```

---

## Component Interaction

### Music Generation Components

```
KellyBrain (High-level API)
    │
    ├─→ IntentPipeline (Intent processing)
    │   └─→ EmotionThesaurus (Emotion lookup)
    │
    └─→ MidiGenerator (MIDI generation)
        ├─→ ChordGenerator
        ├─→ MelodyEngine
        ├─→ BassEngine
        ├─→ PadEngine
        ├─→ StringEngine
        ├─→ CounterMelodyEngine
        ├─→ RhythmEngine
        ├─→ DrumGrooveEngine
        ├─→ FillEngine
        ├─→ TensionEngine
        ├─→ DynamicsEngine
        └─→ GrooveEngine
```

### Vocal Generation Components

```
PRROTEngine (Main orchestrator)
    │
    ├─→ SpectralAnalyzer (FFT analysis)
    │   └─→ JUCE FFT (optimized)
    │
    ├─→ PhonemeSegmenter (Phoneme detection)
    │
    ├─→ ArticulationAnalyzer (Articulation analysis)
    │
    ├─→ BreathDetector (Breath detection)
    │
    ├─→ PitchTracker (Pitch tracking)
    │
    └─→ MidiShaper (MIDI note shaping)
```

---

## Performance Timeline

### Music Generation (Standalone)

```
0ms ──────────────────────────────────────────────────> 1000ms
│
├─ 0-10ms:   Intent processing
├─ 10-110ms: MIDI generation
├─ 110-610ms: ML enhancement (optional, standalone)
└─ Total:    100ms - 1s (acceptable)
```

### Vocal Generation (Standalone)

```
0ms ──────────────────────────────────────────────────> 10000ms
│
├─ 0-50ms:    Audio analysis (per second)
├─ 50-70ms:   Phoneme segmentation (per second)
├─ 70-10070ms: ML enhancement (optional, standalone, per second)
└─ Total:     1-10s per second of audio (acceptable)
```

---

## Key Functions

### Music Generation

```cpp
// Entry points
KellyBrain::fromText("I feel lost")
KellyBrain::fromEmotion("melancholy", 0.8f)
KellyBrain::fromWound(wound)
KellyBrain::fromJourney(current, desired)

// Generation
KellyBrain::generateMidi(intent, bars)
KellyBrain::generateMidiFromIntentFrame(frame, bars)

// Preparation
prepareIntentFrame(frame)  // Validate & clamp
```

### Vocal Generation

```cpp
// Entry point
PRROTEngine::processAudioSegment(audio, size, sample_rate, tempo)

// Analysis
PRROTEngine::analyzePhonemes(audio, size, sample_rate)
PRROTEngine::detectBreathMarkers(audio, size, sample_rate)

// Generation
PRROTEngine::generateControlData(phonemes, pitch_targets, tempo)
```

---

## Output Formats

### Music Output

```cpp
GeneratedMidi {
    // Layers
    vector<Chord> chords;
    vector<MidiNote> melody;
    vector<MidiNote> bass;
    // ... more layers

    // Metadata
    float tempoBpm;
    int bars;
    int key;
    int mode;
}
```

### Vocal Output

```cpp
PhonemeControlData {
    // Sequence
    vector<PhonemeTiming> phoneme_sequence;
    vector<PitchTarget> pitch_targets;

    // Events
    vector<BreathMarker> breath_markers;
    vector<VibratoParameters> vibrato_events;

    // Control
    vector<MidiNote> midi_notes;
    vector<AutomationEnvelope> automation_envelopes;
    vector<ArticulationEnvelope> articulation_envelopes;
}
```

---

## Quick Reference

### Music: Text → MIDI
```
Text → IntentPipeline → IntentFrame → KellyBrain → MidiGenerator → GeneratedMidi
```

### Vocals: Audio → Control Data
```
Audio → PRROTEngine → Analysis → PhonemeControlData
```

### Combined: Text + Audio → Synchronized Output
```
Text → Music Pipeline → GeneratedMidi ──┐
                                        ├─→ Synchronize → Combined Output
Audio → Vocal Pipeline → Control Data ──┘
```

---

**See Also:**
- `docs/FULL_PIPELINE_DOCUMENTATION.md` - Detailed documentation
- `docs/STANDALONE_GENERATION_ARCHITECTURE.md` - Architecture details
