# Intent IR v1 Migration Example

## Example: Migrating MidiKompanionBrain

This example shows how to add `IntentFrame` support to an existing class while maintaining backward compatibility.

### Step 1: Update Header File

**File**: `engine/src/engine/MidiKompanionBrain.h`

```cpp
#pragma once

#include "common/KellyTypes.h"
#include "common/IntentIRAdapter.h"  // ADD THIS
#include "shared/include/kmidi/IntentIR.h"  // ADD THIS

// ... existing includes ...

class MidiKompanionBrain {
public:
    // ... existing methods ...

    // NEW: IntentFrame-based methods
    IntentFrame fromWoundToIntentFrame(const Wound& wound);
    IntentFrame fromTextToIntentFrame(const std::string& description);
    IntentFrame fromEmotionToIntentFrame(const std::string& emotionName, float intensity = 0.7f);
    IntentFrame fromJourneyToIntentFrame(const SideA& current, const SideB& desired);

    GeneratedMidi generateMidiFromIntentFrame(const IntentFrame& frame, int bars = 8);

    // EXISTING: Keep for backward compatibility
    IntentResult fromWound(const Wound& wound);
    IntentResult fromText(const std::string& description);
    IntentResult fromEmotion(const std::string& emotionName, float intensity = 0.7f);
    IntentResult fromJourney(const SideA& current, const SideB& desired);
    GeneratedMidi generateMidi(const IntentResult& intent, int bars = 8);

private:
    std::unique_ptr<IntentPipeline> pipeline_;
    // ... existing members ...
};
```

### Step 2: Update Implementation File

**File**: `engine/src/engine/MidiKompanionBrain.cpp`

```cpp
#include "engine/MidiKompanionBrain.h"
#include "engine/IntentPipeline.h"
#include "common/IntentIRAdapter.h"  // ADD THIS
#include "shared/include/kmidi/IntentIR.h"  // ADD THIS

// ... existing code ...

// NEW: IntentFrame-based implementation
IntentFrame MidiKompanionBrain::fromWoundToIntentFrame(const Wound& wound) {
    if (!pipeline_) {
        // Return default frame if pipeline not initialized
        IntentFrame frame;
        frame.meta.ir_version = INTENT_IR_VERSION;
        return frame;
    }

    // Use new IntentPipeline method
    return pipeline_->processToIntentFrame(wound, getCurrentSessionId());
}

IntentFrame MidiKompanionBrain::fromTextToIntentFrame(const std::string& description) {
    Wound wound;
    wound.description = description;
    wound.intensity = 0.7f;  // Default intensity
    return fromWoundToIntentFrame(wound);
}

IntentFrame MidiKompanionBrain::fromEmotionToIntentFrame(const std::string& emotionName, float intensity) {
    Wound wound;
    wound.primaryEmotion = emotionName;
    wound.intensity = intensity;
    return fromWoundToIntentFrame(wound);
}

IntentFrame MidiKompanionBrain::fromJourneyToIntentFrame(const SideA& current, const SideB& desired) {
    if (!pipeline_) {
        IntentFrame frame;
        frame.meta.ir_version = INTENT_IR_VERSION;
        return frame;
    }

    return pipeline_->processJourneyToIntentFrame(current, desired, getCurrentSessionId());
}

GeneratedMidi MidiKompanionBrain::generateMidiFromIntentFrame(const IntentFrame& frame, int bars) {
    // Validate and clamp frame before use
    prepareIntentFrame(frame);

    // Use MidiGenerator with IntentFrame
    // (Assuming MidiGenerator has been updated)
    MidiGenerator generator;
    return generator.generate(frame, bars, 0.5f, 0.4f, 0.5f, 0.6f);
}

// EXISTING: Update to use new methods internally
IntentResult MidiKompanionBrain::fromWound(const Wound& wound) {
    // Convert IntentFrame to IntentResult for backward compatibility
    IntentFrame frame = fromWoundToIntentFrame(wound);
    return convertIntentIRToIntentResult(frame);
}

IntentResult MidiKompanionBrain::fromText(const std::string& description) {
    IntentFrame frame = fromTextToIntentFrame(description);
    return convertIntentIRToIntentResult(frame);
}

IntentResult MidiKompanionBrain::fromEmotion(const std::string& emotionName, float intensity) {
    IntentFrame frame = fromEmotionToIntentFrame(emotionName, intensity);
    return convertIntentIRToIntentResult(frame);
}

IntentResult MidiKompanionBrain::fromJourney(const SideA& current, const SideB& desired) {
    IntentFrame frame = fromJourneyToIntentFrame(current, desired);
    return convertIntentResultToIntentIR(frame);
}

GeneratedMidi MidiKompanionBrain::generateMidi(const IntentResult& intent, int bars) {
    // Convert IntentResult to IntentFrame
    IntentFrame frame = convertIntentResultToIntentIR(intent);
    return generateMidiFromIntentFrame(frame, bars);
}
```

### Step 3: Update Callers (Gradual Migration)

**Before**:
```cpp
MidiKompanionBrain brain;
brain.initialize("./data");

IntentResult result = brain.fromText("I feel lost");
GeneratedMidi midi = brain.generateMidi(result, 8);
```

**After** (New Code):
```cpp
MidiKompanionBrain brain;
brain.initialize("./data");

IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");
prepareIntentFrame(frame);  // Validate + clamp (do this once)
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

**During Migration** (Hybrid):
```cpp
MidiKompanionBrain brain;
brain.initialize("./data");

// Use old method, convert to new format
IntentResult result = brain.fromText("I feel lost");
IntentFrame frame = convertIntentResultToIntentIR(result);
prepareIntentFrame(frame);

// Use new method
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

## Example: Migrating MidiGenerator

### Step 1: Add IntentFrame Overload

**File**: `engine/src/midi/MidiGenerator.h`

```cpp
class MidiGenerator {
public:
    // NEW: IntentFrame-based method
    GeneratedMidi generate(const IntentFrame& frame, int bars,
                          float complexity, float humanize,
                          float feel, float dynamics);

    // EXISTING: Keep for backward compatibility
    GeneratedMidi generate(const IntentResult& intent, int bars,
                          float complexity, float humanize,
                          float feel, float dynamics);
};
```

### Step 2: Implement New Method

**File**: `engine/src/midi/MidiGenerator.cpp`

```cpp
#include "midi/MidiGenerator.h"
#include "common/IntentIRAdapter.h"
#include "common/IntentIRExtractor.h"
#include "shared/include/kmidi/IntentIR.h"

GeneratedMidi MidiGenerator::generate(const IntentFrame& frame, int bars,
                                      float complexity, float humanize,
                                      float feel, float dynamics) {
    GeneratedMidi result;

    // Extract parameters from IntentFrame
    EmotionParams emotion = EmotionParams::fromIntentFrame(frame);
    HarmonyParams harmony = HarmonyParams::fromIntentFrame(frame);
    RhythmParams rhythm = RhythmParams::fromIntentFrame(frame);
    DynamicsParams dynamicsParams = DynamicsParams::fromIntentFrame(frame);

    // Map IntentFrame biases to concrete parameters
    std::string mode = (harmony.mode_preference > 0) ? "major" : "minor";
    int tempoBpm = tempoBiasToBPM(frame.music.tempo_bias);
    std::string key = "C";  // Default

    // Use EmotionMusicMapper for emotion-based parameters
    auto musicalParams = EmotionMusicMapper::mapEmotion(
        emotion.valence, emotion.arousal, emotion.dominance
    );
    tempoBpm = musicalParams.tempo;
    if (!musicalParams.detailedMode.empty()) {
        mode = musicalParams.detailedMode;
    }

    result.bpm = static_cast<float>(tempoBpm);
    result.lengthInBeats = bars * BEATS_PER_BAR;

    // Generate arrangement if needed
    std::optional<ArrangementOutput> arrangementOpt;
    if (bars >= 8) {
        // Create temporary IntentResult for arrangement generation
        // (ArrangementEngine not yet migrated)
        IntentResult tempIntent = convertIntentIRToIntentResult(frame);
        ArrangementOutput arrangement = generateArrangement(tempIntent, bars);
        arrangementOpt = arrangement;
    }

    // Determine layers
    LayerFlags layers = determineLayersFromIntentFrame(frame, complexity, bars);

    // Generate harmonic foundation
    result.chords = generateChordsFromIntentFrame(frame, bars);
    std::vector<std::string> chordStrings = chordsToStrings(result.chords);

    // Generate melodic layers using IntentFrame
    if (layers.melody) {
        MelodyEngine melodyEngine;
        MelodyOutput melodyOutput = melodyEngine.generateFromIntentFrame(
            frame, key, bars, tempoBpm
        );
        result.melody = melodyOutputToMidiNotes(melodyOutput);
    }

    if (layers.bass) {
        BassEngine bassEngine;
        BassOutput bassOutput = bassEngine.generateFromIntentFrame(
            frame, chordStrings, key, bars, tempoBpm
        );
        result.bass = bassOutputToMidiNotes(bassOutput);
    }

    // ... continue with other layers ...

    return result;
}

// EXISTING: Update to use new method
GeneratedMidi MidiGenerator::generate(const IntentResult& intent, int bars,
                                      float complexity, float humanize,
                                      float feel, float dynamics) {
    // Convert IntentResult to IntentFrame
    IntentFrame frame = convertIntentResultToIntentIR(intent);
    return generate(frame, bars, complexity, humanize, feel, dynamics);
}
```

## Best Practices

### 1. Always Validate Before Use
```cpp
IntentFrame frame = getIntentFrame();
prepareIntentFrame(frame);  // Validate + clamp
// Now safe to use
```

### 2. Use Const References in Audio Thread
```cpp
void processBlock(..., const IntentFrame& frame) {
    // Safe: frame is const, no allocation
    float bias = frame.music.tempo_bias;
}
```

### 3. Convert at Boundaries
```cpp
// At API boundary: convert once
IntentFrame frame = fromTextToIntentFrame("I feel lost");
prepareIntentFrame(frame);

// Pass const reference to engines
melodyEngine.generateFromIntentFrame(frame, ...);
bassEngine.generateFromIntentFrame(frame, ...);
```

### 4. Keep Backward Compatibility
```cpp
// Old API (deprecated but still works)
IntentResult result = fromText("I feel lost");

// New API (preferred)
IntentFrame frame = fromTextToIntentFrame("I feel lost");
```

## Testing Migration

### Unit Test Example
```cpp
TEST(MidiKompanionBrain, IntentFrameSupport) {
    MidiKompanionBrain brain;
    brain.initialize("./data");

    // Test new IntentFrame method
    IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");
    EXPECT_EQ(frame.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_LT(frame.emotion.valence, 0.0f);  // Negative for "lost"

    // Test backward compatibility
    IntentResult result = brain.fromText("I feel lost");
    EXPECT_FALSE(result.emotion.empty());

    // Test conversion
    IntentFrame frame2 = convertIntentResultToIntentIR(result);
    EXPECT_EQ(frame2.meta.ir_version, INTENT_IR_VERSION);
}
```

## Common Pitfalls

### ❌ Don't Validate in Audio Thread
```cpp
void processBlock(..., IntentFrame& frame) {
    prepareIntentFrame(frame);  // BAD: Uses alloc, not audio-thread safe
}
```

### ✅ Validate Before Audio Thread
```cpp
// UI thread
IntentFrame frame = createIntentFrame();
prepareIntentFrame(frame);  // OK: UI thread
validatedFrame_ = frame;

// Audio thread
void processBlock(...) {
    const IntentFrame& frame = validatedFrame_;  // OK: Already validated
}
```

### ❌ Don't Use JSON in Audio Thread
```cpp
void processBlock(...) {
    char* json = intent_frame_to_json(&frame);  // BAD: Allocates
}
```

### ✅ Use JSON Only for Logging
```cpp
// UI thread
char* json = intent_frame_to_json(&frame);
logger.info(json);
free(json);
```
