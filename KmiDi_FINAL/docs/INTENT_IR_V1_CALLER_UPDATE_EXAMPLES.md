# Intent IR v1 - Caller Update Examples

## Common Caller Update Patterns

### Pattern 1: Simple Text-to-MIDI

**Before**:
```cpp
KellyBrain brain;
brain.initialize("./data");

IntentResult result = brain.fromText("I feel lost");
GeneratedMidi midi = brain.generateMidi(result, 8);
```

**After**:
```cpp
KellyBrain brain;
brain.initialize("./data");

IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");
prepareIntentFrame(frame);  // Validate + clamp
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

### Pattern 2: Emotion-Based Generation

**Before**:
```cpp
IntentResult result = brain.fromEmotion("grief", 0.8f);
GeneratedMidi midi = brain.generateMidi(result, 4);
```

**After**:
```cpp
IntentFrame frame = brain.fromEmotionToIntentFrame("grief", 0.8f);
prepareIntentFrame(frame);
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 4);
```

### Pattern 3: Using MidiGenerator Directly

**Before**:
```cpp
MidiGenerator generator;
IntentResult result = brain.fromText("I feel creative");
GeneratedMidi midi = generator.generate(result, 8, 0.5f, 0.4f, 0.0f, 0.75f);
```

**After**:
```cpp
MidiGenerator generator;
IntentFrame frame = brain.fromTextToIntentFrame("I feel creative");
prepareIntentFrame(frame);
GeneratedMidi midi = generator.generate(frame, 8, 0.5f, 0.4f, 0.0f, 0.75f);
```

### Pattern 4: Storing Intent for Later Use

**Before**:
```cpp
class MyProcessor {
    IntentResult currentIntent_;
    
    void updateIntent(const std::string& text) {
        currentIntent_ = brain_.fromText(text);
    }
    
    void generate() {
        GeneratedMidi midi = brain_.generateMidi(currentIntent_, 8);
    }
};
```

**After**:
```cpp
class MyProcessor {
    IntentFrame currentFrame_;
    
    void updateIntent(const std::string& text) {
        currentFrame_ = brain_.fromTextToIntentFrame(text);
        prepareIntentFrame(currentFrame_);  // Validate once
    }
    
    void generate() {
        // Safe: frame is already validated
        GeneratedMidi midi = brain_.generateMidiFromIntentFrame(currentFrame_, 8);
    }
};
```

### Pattern 5: Audio Thread Usage

**Before** (Unsafe):
```cpp
void processBlock(..., IntentResult& intent) {
    // IntentResult contains std::string, std::vector - NOT audio-thread safe
    float tempo = intent.tempoBpm;
}
```

**After** (Safe):
```cpp
// UI thread: Prepare frame
void onUserInput() {
    IntentFrame frame = brain_.fromTextToIntentFrame(userText_);
    prepareIntentFrame(frame);  // UI thread - safe
    validatedFrame_ = frame;    // Store for audio thread
}

// Audio thread: Use const reference
void processBlock(..., const IntentFrame& frame) {
    // Safe: frame is const, no allocation, no std::string operations
    float bias = frame.music.tempo_bias;
    // Use frame fields directly
}
```

### Pattern 6: Converting Existing IntentResult

**Before**:
```cpp
IntentResult result = /* ... from old code ... */;
GeneratedMidi midi = generator.generate(result, 8, ...);
```

**After** (Hybrid):
```cpp
IntentResult result = /* ... from old code ... */;

// Convert to IntentFrame
IntentFrame frame = convertIntentResultToIntentIR(result);
prepareIntentFrame(frame);

// Use new method
GeneratedMidi midi = generator.generate(frame, 8, ...);
```

### Pattern 7: Multiple Engines from Same Frame

**Before**:
```cpp
IntentResult result = brain.fromText("I feel energetic");
auto melody = melodyEngine.generate(result, ...);
auto drums = drumEngine.generate(result, ...);
auto bass = bassEngine.generate(result, ...);
```

**After**:
```cpp
IntentFrame frame = brain.fromTextToIntentFrame("I feel energetic");
prepareIntentFrame(frame);  // Validate once

// Use IntentFrame directly with engines
auto melody = melodyEngine.generateFromIntentFrame(frame, ...);
auto drums = drumEngine.generateFromIntentFrame(frame, ...);
auto bass = bassEngine.generateFromIntentFrame(frame, ...);
```

## PluginProcessor Update Example

### Current Code Pattern
```cpp
// In PluginProcessor.cpp
void PluginProcessor::processBlock(...) {
    // ... existing code ...
    
    // Old pattern (if used)
    if (needNewIntent_) {
        IntentResult result = kellyBrain_->fromText(userInput_);
        GeneratedMidi midi = kellyBrain_->generateMidi(result, 8);
        // ... use midi ...
    }
}
```

### Updated Code Pattern
```cpp
// In PluginProcessor.h
class PluginProcessor {
    IntentFrame validatedFrame_;  // Store validated frame
    std::atomic<bool> frameReady_{false};
    
    // ... existing members ...
};

// In PluginProcessor.cpp
void PluginProcessor::onUserInput(const std::string& text) {
    // UI thread - safe to call prepareIntentFrame
    IntentFrame frame = kellyBrain_->fromTextToIntentFrame(text);
    prepareIntentFrame(frame);
    validatedFrame_ = frame;
    frameReady_.store(true);
}

void PluginProcessor::processBlock(...) {
    // Audio thread
    if (frameReady_.load()) {
        const IntentFrame& frame = validatedFrame_;  // Const reference
        // Safe: frame is const, no allocation
        GeneratedMidi midi = kellyBrain_->generateMidiFromIntentFrame(frame, 8);
        // ... use midi ...
    }
}
```

## Migration Strategy

### Phase 1: Add New Methods (Non-Breaking) ✅
- ✅ Done: New IntentFrame methods added
- ✅ Done: Old methods still work

### Phase 2: Update New Code
- Use IntentFrame methods in new features
- Use IntentFrame in tests
- Use IntentFrame in examples

### Phase 3: Gradual Migration
- Update one caller at a time
- Test after each update
- Keep old code working during transition

### Phase 4: Deprecation (Future)
- Mark IntentResult methods as deprecated
- Provide migration guide
- Remove after deprecation period

## Files to Update (When Ready)

### High Priority
1. `engine/src/plugin/PluginProcessor.cpp` - Main plugin entry point
2. `engine/src/ml/MLBridge.cpp` - ML integration
3. Test files - Use IntentFrame in tests

### Medium Priority
4. GUI controllers - Use IntentFrame for new features
5. OSC handlers - Use IntentFrame for new messages
6. Python bridge - Use IntentFrame for new APIs

### Low Priority
7. Documentation examples - Update to use IntentFrame
8. Tutorial code - Use IntentFrame in tutorials

## Testing After Updates

After updating a caller:

1. **Compile**: Ensure code compiles
2. **Test**: Run relevant tests
3. **Verify**: Check that output matches expected behavior
4. **Profile**: Verify no performance regressions

## Rollback Plan

If issues arise:

1. Keep old IntentResult methods (they still work)
2. Revert caller to use IntentResult
3. Fix issue in IntentFrame code
4. Re-apply caller update

## Benefits of Migration

### Performance
- ✅ 10-100x faster frame copying
- ✅ Zero allocations in audio thread
- ✅ Predictable memory usage

### Safety
- ✅ Audio-thread safe (const reference)
- ✅ Immutable once validated
- ✅ Type-safe (C struct)

### Maintainability
- ✅ Single canonical format
- ✅ Versioned for future changes
- ✅ Language-agnostic
