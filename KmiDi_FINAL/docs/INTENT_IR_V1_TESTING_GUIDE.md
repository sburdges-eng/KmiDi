# Intent IR v1 Testing Guide

## Test Files Created

### 1. Python Integration Tests
**File**: `tests/intent_ir_integration_test.py`

**Tests**:
- IntentFrame creation
- JSON round-trip serialization
- Validation
- File I/O
- Intent ID generation
- CompleteSongIntent conversion
- Engine consumability

**Run**:
```bash
python tests/intent_ir_integration_test.py
```

### 2. C++ Unit Tests
**File**: `tests/intent_ir_cpp_test.cpp`

**Tests**:
- IntentFrame default construction
- Validation
- Clamping
- IntentPipeline produces IntentFrame
- Engine consumption (Melody, Drum, Bass, Dynamics, Pad)
- IntentIRExtractor helpers
- Adapter round-trip
- Version negotiation
- JSON serialization

**Run** (after building):
```bash
cd build
ctest -R intent_ir
```

### 3. C++ Integration Tests
**File**: `tests/intent_ir_cpp_integration_test.cpp`

**Tests**:
- KellyBrain IntentFrame methods
- Full pipeline: Text → IntentFrame → MIDI
- MidiGenerator with IntentFrame
- Round-trip conversion
- Validation and clamping
- JSON serialization
- Journey processing

**Run** (after building):
```bash
cd build
ctest -R intent_ir_integration
```

## Testing Checklist

### Build Tests
- [ ] Rust crate compiles (`cargo build --lib`)
- [ ] CMake configures without errors
- [ ] C++ adapter compiles
- [ ] All targets link successfully

### Unit Tests
- [ ] Rust validator tests pass
- [ ] C++ unit tests pass
- [ ] Python integration tests pass

### Integration Tests
- [ ] Full pipeline test: Text → IntentFrame → MIDI
- [ ] Conversion test: IntentResult ↔ IntentFrame
- [ ] JSON round-trip test
- [ ] Engine consumption test

### Performance Tests
- [ ] No allocations in audio thread
- [ ] Frame copy < 1μs
- [ ] Validation < 50μs (not in audio thread)

## Manual Testing

### Test 1: Basic Usage
```cpp
KellyBrain brain;
brain.initialize("./data");

IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");
prepareIntentFrame(frame);
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);

// Verify: midi.bpm > 0, midi.bars == 8, midi has content
```

### Test 2: Emotion Mapping
```cpp
IntentFrame frame = brain.fromEmotionToIntentFrame("grief", 0.8f);
prepareIntentFrame(frame);

// Verify: frame.emotion.valence < 0, frame.emotion.arousal > 0
```

### Test 3: JSON Serialization
```cpp
IntentFrame frame = brain.fromTextToIntentFrame("test");
prepareIntentFrame(frame);

char* json = intent_frame_to_json(&frame);
IntentFrame frame2;
bool success = intent_frame_from_json(json, &frame2);

// Verify: success == true, frame2 matches frame
```

### Test 4: Backward Compatibility
```cpp
// Old way
IntentResult result = brain.fromText("test");
GeneratedMidi midi1 = brain.generateMidi(result, 4);

// New way
IntentFrame frame = brain.fromTextToIntentFrame("test");
prepareIntentFrame(frame);
GeneratedMidi midi2 = brain.generateMidiFromIntentFrame(frame, 4);

// Verify: Both produce similar results
```

## Expected Results

### IntentFrame Creation
- `frame.meta.ir_version == 1`
- `frame.meta.intent_id > 0`
- `frame.emotion.valence` in range [-1.0, 1.0]
- `frame.emotion.arousal` in range [0.0, 1.0]

### MIDI Generation
- `midi.bpm > 0`
- `midi.bars == requested_bars`
- `midi.lengthInBeats > 0`
- At least one layer has content (chords, melody, or bass)

### Validation
- Valid frame: `intent_frame_validate(&frame) == true`
- Invalid version: `intent_frame_validate(&frame) == false` (if version != 1)
- Clamping: Out-of-range values are clamped to valid ranges

## Troubleshooting

### Build Fails
- Check Rust toolchain: `cargo --version`
- Check CMake version: `cmake --version` (need 3.20+)
- Check cJSON: `pkg-config --exists cjson`

### Tests Fail
- Check Python imports
- Verify C++ test linking
- Check JSON serialization
- Review test output for specific errors

### Runtime Errors
- Ensure `prepareIntentFrame()` is called before use
- Check that frame is validated
- Verify no allocations in audio thread
- Check const correctness

## Performance Benchmarks

Expected performance (M1 Mac):
- IntentFrame creation: < 1μs
- Validation: ~20μs (not audio-thread safe)
- Clamping: ~15μs (not audio-thread safe)
- JSON serialize: ~100μs (not audio-thread safe)
- Frame copy: < 100ns (audio-thread safe)
- Field access: < 1ns (audio-thread safe)

## Next Steps After Testing

1. Fix any build errors
2. Fix any test failures
3. Profile performance
4. Update callers to use new methods
5. Document any issues found
