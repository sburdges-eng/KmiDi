# Intent IR v1 Quick Start Guide

## 5-Minute Quick Start

### 1. Create IntentFrame from Text

```cpp
#include "engine/KellyBrain.h"
#include "common/IntentIRAdapter.h"
#include "shared/include/kmidi/IntentIR.h"

KellyBrain brain;
brain.initialize("./data");

// Create IntentFrame
IntentFrame frame = brain.fromTextToIntentFrame("I feel joyful");

// IMPORTANT: Validate and clamp before use
prepareIntentFrame(frame);

// Generate MIDI
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

### 2. Create IntentFrame from Emotion

```cpp
IntentFrame frame = brain.fromEmotionToIntentFrame("grief", 0.8f);
prepareIntentFrame(frame);
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 4);
```

### 3. Use MidiGenerator Directly

```cpp
MidiGenerator generator;
IntentFrame frame = brain.fromTextToIntentFrame("I feel creative");
prepareIntentFrame(frame);

GeneratedMidi midi = generator.generate(frame, 8, 0.5f, 0.4f, 0.0f, 0.75f);
```

## Key Points

### ✅ Always Validate Before Use
```cpp
IntentFrame frame = /* ... create frame ... */;
prepareIntentFrame(frame);  // DO THIS FIRST
// Now safe to use
```

### ✅ Use Const References in Audio Thread
```cpp
void processBlock(..., const IntentFrame& frame) {
    // Safe: frame is const, no allocation
    float bias = frame.music.tempo_bias;
}
```

### ✅ Convert at Boundaries
```cpp
// Old code uses IntentResult
IntentResult result = brain.fromText("I feel sad");

// Convert to IntentFrame for new code
IntentFrame frame = convertIntentResultToIntentIR(result);
prepareIntentFrame(frame);

// Use with new methods
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

## Common Patterns

### Pattern 1: Text → IntentFrame → MIDI
```cpp
IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");
prepareIntentFrame(frame);
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

### Pattern 2: Emotion → IntentFrame → MIDI
```cpp
IntentFrame frame = brain.fromEmotionToIntentFrame("joy", 0.7f);
prepareIntentFrame(frame);
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 4);
```

### Pattern 3: Journey → IntentFrame → MIDI
```cpp
SideA current{"I feel anxious", 0.7f};
SideB desired{"I want to feel calm", 0.6f};
IntentFrame frame = brain.fromJourneyToIntentFrame(current, desired);
prepareIntentFrame(frame);
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

### Pattern 4: IntentFrame → JSON (for logging)
```cpp
IntentFrame frame = brain.fromTextToIntentFrame("I feel happy");
prepareIntentFrame(frame);

char* json = intent_frame_to_json(&frame);
logger.info("IntentFrame", json);
free(json);
```

## Migration Checklist

When updating existing code:

- [ ] Include `IntentIR.h` and `IntentIRAdapter.h`
- [ ] Use `fromTextToIntentFrame()` instead of `fromText()`
- [ ] Call `prepareIntentFrame()` before using frame
- [ ] Use `generateMidiFromIntentFrame()` instead of `generateMidi(IntentResult)`
- [ ] Pass `const IntentFrame&` to audio thread (not non-const)
- [ ] Convert IntentResult → IntentFrame at boundaries if needed

## See Also

- `examples/intent_ir_usage_example.cpp` - Full working examples
- `docs/INTENT_IR_V1_USAGE.md` - Detailed usage guide
- `docs/INTENT_IR_V1_PERFORMANCE.md` - Performance & thread safety
- `docs/INTENT_IR_V1_MIGRATION_EXAMPLE.md` - Migration patterns
