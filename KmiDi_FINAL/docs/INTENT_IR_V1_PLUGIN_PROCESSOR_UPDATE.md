# PluginProcessor IntentFrame Update Example

## Current Implementation

The `PluginProcessor::generateMidi()` method currently uses `IntentResult`:

```cpp
void PluginProcessor::generateMidi() {
    // ... parameter extraction ...

    // Current code (around line 737-768)
    if (hasJourney) {
        IntentResult kellyIntent = kellyBrain_->fromJourney(sideA, sideB);
    } else {
        IntentResult kellyIntent = kellyBrain_->fromWound(kellyWound);
    }

    GeneratedMidi midi = kellyBrain_->generateMidi(kellyIntent, bars);
    // ... use midi ...
}
```

## Updated Implementation (Using IntentFrame)

Here's how to update it to use IntentFrame:

```cpp
// In PluginProcessor.h - Add member variable
class PluginProcessor {
    // ... existing members ...

    // NEW: Store validated IntentFrame for audio thread
    IntentFrame validatedFrame_;
    std::atomic<bool> frameReady_{false};
    std::mutex frameMutex_;  // Protect frame updates
};

// In PluginProcessor.cpp - Update generateMidi()
void PluginProcessor::generateMidi() {
    if (isGenerating_.exchange(true)) {
        return;
    }

    // ... existing parameter extraction ...

    // NEW: Create IntentFrame instead of IntentResult
    IntentFrame frame;

    if (hasJourney) {
        frame = kellyBrain_->fromJourneyToIntentFrame(sideA, sideB);
    } else {
        frame = kellyBrain_->fromWoundToIntentFrame(kellyWound);
    }

    // Validate and clamp (UI thread - safe)
    prepareIntentFrame(frame);

    // Store validated frame
    {
        std::lock_guard<std::mutex> lock(frameMutex_);
        validatedFrame_ = frame;
        frameReady_.store(true);
    }

    // Generate MIDI from IntentFrame
    GeneratedMidi midi = kellyBrain_->generateMidiFromIntentFrame(frame, bars);

    // ... rest of method ...
}
```

## Audio Thread Usage

If you need to use the frame in `processBlock()`:

```cpp
// In PluginProcessor.h
class PluginProcessor {
    IntentFrame validatedFrame_;  // Validated frame
    std::atomic<bool> frameReady_{false};
    std::mutex frameMutex_;
};

// In processBlock() - Audio thread
void PluginProcessor::processBlock(..., juce::MidiBuffer& midiMessages) {
    // ... existing code ...

    // Check if new frame is ready
    if (frameReady_.load()) {
        // Get const reference (safe for audio thread)
        IntentFrame frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex_);
            frame = validatedFrame_;  // Copy (fast, ~80 bytes)
            frameReady_.store(false);
        }

        // Safe: frame is const, no allocation
        float tempoBias = frame.music.tempo_bias;
        float valence = frame.emotion.valence;

        // Use frame parameters for real-time adjustments
        // ...
    }
}
```

## Hybrid Approach (During Migration)

If you want to keep both paths working during migration:

```cpp
void PluginProcessor::generateMidi() {
    // ... parameter extraction ...

    // Option 1: Use IntentFrame (new way)
    if (useIntentFrame_) {  // Feature flag
        IntentFrame frame;
        if (hasJourney) {
            frame = kellyBrain_->fromJourneyToIntentFrame(sideA, sideB);
        } else {
            frame = kellyBrain_->fromWoundToIntentFrame(kellyWound);
        }
        prepareIntentFrame(frame);
        GeneratedMidi midi = kellyBrain_->generateMidiFromIntentFrame(frame, bars);
        // ... use midi ...
    } else {
        // Option 2: Use IntentResult (old way, still works)
        IntentResult result;
        if (hasJourney) {
            result = kellyBrain_->fromJourney(sideA, sideB);
        } else {
            result = kellyBrain_->fromWound(kellyWound);
        }
        GeneratedMidi midi = kellyBrain_->generateMidi(result, bars);
        // ... use midi ...
    }
}
```

## Benefits of Update

### Performance
- ✅ Faster frame copying (~80 bytes vs variable-size IntentResult)
- ✅ No heap allocation in audio thread
- ✅ Predictable memory usage

### Safety
- ✅ Audio-thread safe (const reference)
- ✅ Immutable once validated
- ✅ Type-safe (C struct)

### Maintainability
- ✅ Single canonical format
- ✅ Versioned for future changes
- ✅ Better for serialization/logging

## Migration Checklist

When updating PluginProcessor:

- [ ] Add `#include "common/IntentIRAdapter.h"` and `#include "shared/include/kmidi/IntentIR.h"`
- [ ] Add `validatedFrame_` member variable
- [ ] Update `generateMidi()` to use IntentFrame methods
- [ ] Call `prepareIntentFrame()` before use
- [ ] Use `const IntentFrame&` in audio thread
- [ ] Test thoroughly
- [ ] Remove old IntentResult path (after verification)

## Testing After Update

1. **Compile**: Ensure code compiles
2. **Test**: Run plugin in DAW
3. **Verify**: Check that MIDI generation works
4. **Profile**: Verify no performance regressions
5. **Check**: Ensure no allocations in audio thread
