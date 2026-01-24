# Intent IR v1 - Next Steps

## Immediate Actions (Do First)

### 1. Verify Build Setup ✅
```bash
cd KmiDi_FINAL
./scripts/verify_intent_ir_build.sh
```

**Expected**: All components found, Rust toolchain available

### 2. Test Rust Crate Build
```bash
cd engine/intent_ir
cargo build --lib
cargo test
```

**Expected**: Crate compiles, tests pass

### 3. Configure CMake
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
```

**Check for**:
- ✅ Rust crate found
- ✅ cbindgen generates header
- ✅ intent_ir_adapter target created
- ✅ KellyCore links intent_ir_adapter

### 4. Build Project
```bash
cmake --build . -j$(nproc)
```

**Watch for**:
- Rust crate compiles
- FFI header generated (`build/include/intent_ir_ffi.h`)
- C++ adapter compiles
- KellyCore links successfully

### 5. Run Integration Tests
```bash
# Python tests
python tests/intent_ir_integration_test.py

# C++ tests (if built)
cd build
ctest -R intent_ir
```

## Integration Tasks (Short-term)

### 6. Update IntentPipeline Callers

**Find all usages**:
```bash
grep -r "processToIntent\|IntentResult" engine/src --include="*.cpp" --include="*.h"
```

**Migration pattern**:
```cpp
// OLD
IntentResult result = pipeline.process(wound, sessionId);

// NEW
IntentFrame frame = pipeline.processToIntentFrame(wound, sessionId);
// Use frame directly, or convert if needed:
IntentResult result = convertIntentIRToIntentResult(frame);
```

**Files to check**:
- Plugin processors
- GUI controllers
- OSC handlers
- Test files

### 7. Update Plugin Processors

**Location**: `engine/src/plugin/`

**Changes needed**:
1. Include `IntentIR.h` and `IntentIRAdapter.h`
2. Replace `IntentResult` with `IntentFrame` where appropriate
3. Use `prepareIntentFrame()` before audio thread
4. Pass const reference to audio thread

**Example**:
```cpp
// UI thread
IntentFrame frame = pipeline.processToIntentFrame(wound, sessionId);
prepareIntentFrame(frame);  // Validate + clamp
validatedFrame_ = frame;

// Audio thread
void processBlock(..., const IntentFrame& frame) {
    // Safe: frame is const, no allocation
    auto melody = melodyEngine.generateFromIntentFrame(frame, ...);
}
```

### 8. Update Python Integration Points

**Find Python code that creates/uses intent**:
```bash
grep -r "CompleteSongIntent\|IntentResult" python/ --include="*.py"
```

**Migration**:
```python
# OLD
intent = CompleteSongIntent(...)
# ... use intent directly

# NEW
intent = CompleteSongIntent(...)
frame = convert_complete_song_intent_to_ir(intent, session_id=123)
# Use frame for serialization/logging
# Pass to C++ via JSON or FFI
```

## Testing & Validation (Medium-term)

### 9. End-to-End Testing

**Create test scenario**:
1. Python creates `CompleteSongIntent`
2. Convert to `IntentFrame`
3. Serialize to JSON
4. C++ deserializes JSON
5. Pass to `IntentPipeline`
6. Engines consume `IntentFrame`
7. Verify output

**Test file**: `tests/intent_ir_e2e_test.py`

### 10. Performance Profiling

**Profile IntentFrame operations**:
```cpp
// Add timing around critical paths
auto start = std::chrono::high_resolution_clock::now();
IntentFrame frame = pipeline.processToIntentFrame(wound, sessionId);
auto end = std::chrono::high_resolution_clock::now();
// Log duration
```

**Verify**:
- ✅ No allocations in audio thread
- ✅ Frame copy < 1μs
- ✅ Validation < 50μs (not in audio thread)

### 11. Memory Safety Verification

**Use Valgrind/AddressSanitizer**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build .
# Run tests with sanitizer
```

**Check for**:
- No leaks in IntentFrame operations
- No use-after-free
- No buffer overflows

## Migration Tasks (Long-term)

### 12. Migrate Remaining Engines (Optional)

**Engines to migrate** (as needed):
- `ArrangementEngine`
- `CounterMelodyEngine`
- `FillEngine`
- `GrooveEngine`
- `RhythmEngine`
- `StringEngine`
- `TensionEngine`
- `TransitionEngine`
- `VariationEngine`

**Pattern** (see `BassEngine` for example):
1. Add `#include "../common/IntentIRExtractor.h"`
2. Add `generateFromIntentFrame()` method
3. Extract params using `IntentIRExtractor`
4. Map IR biases to engine config
5. Call existing `generate()` method

### 13. Deprecate IntentResult (Future)

**Once all engines migrated**:
1. Mark `IntentResult` as deprecated
2. Update all callers to use `IntentFrame`
3. Remove `IntentResult` after deprecation period

**Timeline**: After 6+ months of stable usage

## Documentation Tasks

### 14. Update API Documentation

**Files to update**:
- `docs/API.md` (if exists)
- Engine documentation
- Plugin developer guide

**Add examples**:
- Creating IntentFrame from Python
- Using IntentFrame in C++ engines
- JSON serialization for logging

### 15. Create Migration Guide

**Document**:
- Step-by-step migration from IntentResult
- Common pitfalls
- Performance considerations
- Thread safety guidelines

## Monitoring & Maintenance

### 16. Add Logging

**Log IntentFrame operations**:
```cpp
// Log frame creation
logger.info("IntentFrame created", {
    {"intent_id", frame.meta.intent_id},
    {"valence", frame.emotion.valence},
    {"tempo_bias", frame.music.tempo_bias}
});
```

**Use cases**:
- Debugging intent flow
- Performance monitoring
- User analytics

### 17. Version Compatibility

**Plan for IR v2** (if needed):
- Add version negotiation
- Support multiple versions simultaneously
- Migration path from v1 to v2

## Priority Order

**High Priority** (Do First):
1. ✅ Build verification
2. ✅ CMake configuration
3. ✅ Run tests
4. Fix any build errors

**Medium Priority** (This Week):
5. Update IntentPipeline callers
6. Update plugin processors
7. End-to-end testing

**Low Priority** (As Needed):
8. Migrate remaining engines
9. Performance optimization
10. Documentation updates

## Success Criteria

**Build Success**:
- [ ] Rust crate compiles
- [ ] CMake configures without errors
- [ ] All targets build successfully
- [ ] Tests pass

**Integration Success**:
- [ ] IntentPipeline produces IntentFrame
- [ ] Engines consume IntentFrame
- [ ] Python → C++ flow works
- [ ] JSON serialization works

**Performance Success**:
- [ ] No allocations in audio thread
- [ ] Frame operations < 1μs
- [ ] Memory usage acceptable

## Troubleshooting

**If build fails**:
1. Check Rust toolchain: `cargo --version`
2. Check CMake version: `cmake --version` (need 3.20+)
3. Check cJSON: `pkg-config --exists cjson`
4. Review build errors carefully

**If tests fail**:
1. Check Python imports
2. Verify C++ test linking
3. Check JSON serialization
4. Review test output

**If performance issues**:
1. Profile with Tracy/Instruments
2. Check for accidental allocations
3. Verify const references
4. Review thread safety

## Getting Help

**Documentation**:
- `docs/INTENT_IR_V1_USAGE.md` - Usage guide
- `docs/INTENT_IR_V1_PERFORMANCE.md` - Performance guide
- `docs/INTENT_IR_V1_BUILD_GUIDE.md` - Build guide

**Code Examples**:
- `engine/src/engines/BassEngine.cpp` - Engine migration example
- `engine/src/engine/IntentPipeline.cpp` - Pipeline integration
- `python/music_brain/session/intent_ir_converter.py` - Python conversion
