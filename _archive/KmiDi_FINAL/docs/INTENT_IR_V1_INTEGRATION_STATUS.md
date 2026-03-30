# Intent IR v1 Integration Status

## ✅ Completed Integration Points

### Core Infrastructure
- [x] `IntentPipeline::processToIntentFrame()` - Produces IntentFrame
- [x] `IntentPipeline::processJourneyToIntentFrame()` - Journey support
- [x] `IntentIRAdapter` - Bidirectional conversion IntentFrame ↔ IntentResult
- [x] `IntentIRExtractor` - Helper utilities for engines

### Engines (5/14 migrated)
- [x] `MelodyEngine::generateFromIntentFrame()`
- [x] `DrumGrooveEngine::generateFromIntentFrame()`
- [x] `BassEngine::generateFromIntentFrame()`
- [x] `DynamicsEngine::applyFromIntentFrame()`
- [x] `PadEngine::generateFromIntentFrame()`

## ✅ Completed Integration Points

### High Priority - COMPLETE ✅

#### 1. `KellyBrain` / `MidiKompanionBrain` ✅
**Status**: Complete
**Files Updated**:
- `engine/src/engine/KellyBrain.h` - Added IntentFrame method declarations
- `engine/src/engine/KellyBrain.cpp` - Implemented IntentFrame methods
- `engine/src/engine/MidiKompanionBrain.h` - Added IntentFrame method declarations

**New Methods**:
- `fromWoundToIntentFrame()` - Process wound to IntentFrame
- `fromJourneyToIntentFrame()` - Process journey to IntentFrame
- `fromTextToIntentFrame()` - Process text to IntentFrame
- `fromEmotionToIntentFrame()` - Process emotion to IntentFrame
- `generateMidiFromIntentFrame()` - Generate MIDI from IntentFrame

#### 2. `MidiGenerator` ✅
**Status**: Complete
**Files Updated**:
- `engine/src/midi/MidiGenerator.h` - Added IntentFrame overload
- `engine/src/midi/MidiGenerator.cpp` - Implemented IntentFrame generate() method

**New Method**:
- `generate(const IntentFrame& frame, ...)` - Generate MIDI from IntentFrame

#### 3. `AdaptiveGenerator` ✅
**Status**: Complete
**Files Updated**:
- `engine/src/engine/AdaptiveGenerator.h` - Added IntentFrame method declaration
- `engine/src/engine/AdaptiveGenerator.cpp` - Implemented IntentFrame method

**New Method**:
- `generateMidiFromIntentFrame()` - Generate MIDI with adaptive adjustments from IntentFrame

## 🔄 Integration Points Needing Updates

### Medium Priority

#### 4. `PluginProcessor` (engine/src/plugin/PluginProcessor.h/cpp)
**Current**: Uses `KellyBrain` and `MLIntentPipeline` which use `IntentResult`
**Needs**: Update to use `IntentFrame` where appropriate

**Note**: This is complex due to ML integration. May need adapter layer.

**Files**:
- `engine/src/plugin/PluginProcessor.h`
- `engine/src/plugin/PluginProcessor.cpp`

#### 5. `MLBridge` (engine/src/ml/MLBridge.h/cpp)
**Current**: Uses `IntentResult`
**Needs**: Add `IntentFrame` support for ML inference results

**Files**:
- `engine/src/ml/MLBridge.h`
- `engine/src/ml/MLBridge.cpp`

### Low Priority (Optional)

#### 6. Remaining Engines (9 engines)
- `ArrangementEngine`
- `CounterMelodyEngine`
- `FillEngine`
- `GrooveEngine`
- `RhythmEngine`
- `StringEngine`
- `TensionEngine`
- `TransitionEngine`
- `VariationEngine`

**Pattern**: Follow `BassEngine` example - add `generateFromIntentFrame()` method

## Migration Strategy

### Phase 1: Add New Methods (Non-Breaking)
1. Add `IntentFrame`-based methods alongside existing `IntentResult` methods
2. Keep old methods for backward compatibility
3. Use adapter functions for conversion

### Phase 2: Update Callers (Gradual)
1. Update new code to use `IntentFrame` methods
2. Update tests to use `IntentFrame`
3. Update documentation

### Phase 3: Deprecate Old Methods (Future)
1. Mark `IntentResult` methods as deprecated
2. Provide migration guide
3. Remove after deprecation period

## Example Migration

### Before (IntentResult)
```cpp
MidiKompanionBrain brain;
IntentResult result = brain.fromText("I feel lost");
GeneratedMidi midi = brain.generateMidi(result, 8);
```

### After (IntentFrame)
```cpp
MidiKompanionBrain brain;
IntentFrame frame = brain.fromTextToIntentFrame("I feel lost");
prepareIntentFrame(frame);  // Validate + clamp
GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
```

### Hybrid (During Migration)
```cpp
MidiKompanionBrain brain;
IntentResult result = brain.fromText("I feel lost");  // Old method
IntentFrame frame = convertIntentResultToIntentIR(result);  // Convert
GeneratedMidi midi = midiGenerator.generate(frame, 8, ...);  // New method
```

## Integration Checklist

### Immediate (This Week) ✅
- [x] Add `fromWoundToIntentFrame()` to `KellyBrain`/`MidiKompanionBrain`
- [x] Add `generate(IntentFrame)` to `MidiGenerator`
- [x] Add `generateMidiFromIntentFrame()` to `AdaptiveGenerator`
- [x] Create integration tests
- [ ] Test the build (CMake configuration)
- [ ] Run integration tests

### Short-term (This Month)
- [ ] Update `PluginProcessor` to use `IntentFrame` where possible
- [ ] Update `MLBridge` to support `IntentFrame`
- [ ] Create migration examples
- [ ] Update documentation

### Long-term (As Needed)
- [ ] Migrate remaining engines
- [ ] Deprecate `IntentResult` methods
- [ ] Remove `IntentResult` entirely

## Testing Strategy

### Unit Tests
- Test new `IntentFrame` methods
- Test adapter conversions
- Test backward compatibility

### Integration Tests
- Test full pipeline: Wound → IntentFrame → Engine → MIDI
- Test JSON serialization round-trip
- Test performance (no allocations in audio thread)

### Regression Tests
- Ensure old `IntentResult` methods still work
- Ensure adapter conversions are correct
- Ensure no breaking changes

## Performance Considerations

### Audio Thread Safety
- ✅ `IntentFrame` is safe for audio thread (const reference)
- ❌ Validation/clamping must happen before audio thread
- ❌ JSON serialization must not happen in audio thread

### Memory
- `IntentFrame`: ~80 bytes (stack-allocated)
- `IntentResult`: Contains `std::string`, `std::vector` (heap-allocated)
- **Benefit**: 10-100x faster copying, zero allocations

## Next Steps

1. **Start with `MidiKompanionBrain`** - Core integration point
2. **Update `MidiGenerator`** - Used by most code paths
3. **Update `AdaptiveGenerator`** - Used by plugin
4. **Test thoroughly** - Ensure backward compatibility
5. **Document changes** - Update API docs

## Questions to Resolve

1. Should we keep `IntentResult` indefinitely or plan removal?
2. How to handle ML inference results (convert to `IntentFrame`)?
3. Should `PluginProcessor` use `IntentFrame` directly or via adapter?
4. Timeline for deprecating `IntentResult`?
