# Intent IR v1 Build Verification Checklist

## Pre-Build Verification

### 1. Rust Toolchain ✅
```bash
cargo --version
# Expected: cargo 1.70+ or newer
```

### 2. CMake Version ✅
```bash
cmake --version
# Expected: cmake version 3.20.0 or newer
```

### 3. Component Verification ✅
```bash
./scripts/verify_intent_ir_build.sh
# Should show: ✓ All Intent IR v1 components found!
```

## Build Steps

### Step 1: Configure CMake
```bash
cd KmiDi_FINAL
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
```

**Expected Output**:
- ✅ `-- Rust toolchain found`
- ✅ `-- Building intent_ir Rust crate`
- ✅ `-- Generating IntentIR FFI header`
- ✅ `-- Configuring intent_ir_adapter`
- ✅ `-- Configuring done`

**Watch For**:
- Rust crate configuration
- cbindgen header generation
- intent_ir_adapter target creation
- KellyCore linking intent_ir_adapter

### Step 2: Build Project
```bash
cmake --build . -j$(nproc)
```

**Expected Targets**:
- `intent_ir_rust_lib` - Rust static library
- `intent_ir_adapter` - C++ adapter library
- `KellyCore` - Main library (should link intent_ir_adapter)

**Watch For**:
- Rust crate compiles successfully
- FFI header generated: `build/include/intent_ir_ffi.h`
- C++ adapter compiles
- No linker errors

### Step 3: Verify Generated Files
```bash
# Check FFI header was generated
ls -la build/include/intent_ir_ffi.h

# Check Rust library was built
find build -name "libintent_ir*.a" -o -name "libintent_ir*.so"

# Check adapter library
find build -name "libintent_ir_adapter*.a"
```

## Common Build Issues

### Issue 1: Rust Dependencies Not Found
**Error**: `error: failed to get 'cbindgen' as a dependency`

**Solution**:
```bash
cargo install cbindgen
# Or let CMake install it automatically
```

### Issue 2: CMake Can't Find Rust
**Error**: `CARGO_EXECUTABLE not found`

**Solution**:
```bash
# Ensure cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"
which cargo
```

### Issue 3: FFI Header Not Generated
**Error**: Missing `intent_ir_ffi.h`

**Solution**:
- Check that cbindgen is installed: `cargo install cbindgen`
- Check CMake output for cbindgen errors
- Verify `cbindgen.toml` exists in `engine/intent_ir/`

### Issue 4: Linker Errors
**Error**: `undefined reference to clamp_intent_frame_ffi`

**Solution**:
- Verify Rust library is built: `find build -name "*intent_ir*.a"`
- Check that `intent_ir_adapter` links `${INTENT_IR_RUST_LIB}`
- Ensure Rust library path is correct in CMake

### Issue 5: cJSON Not Found
**Error**: `cJSON.h: No such file or directory`

**Solution**:
```bash
# macOS
brew install cjson

# Or disable JSON features (optional)
# JSON is only for debugging/logging
```

## Build Verification Checklist

After successful build, verify:

- [ ] Rust crate compiled (`libintent_ir.a` or `.so` exists)
- [ ] FFI header generated (`build/include/intent_ir_ffi.h` exists)
- [ ] C++ adapter compiled (`libintent_ir_adapter.a` exists)
- [ ] KellyCore links successfully
- [ ] No linker errors
- [ ] All targets build without errors

## Post-Build Testing

### Quick Test: IntentFrame Creation
```cpp
#include "shared/include/kmidi/IntentIR.h"
#include "common/IntentIRAdapter.h"

IntentFrame frame;
frame.meta.ir_version = INTENT_IR_VERSION;
frame.emotion.valence = 0.5f;

prepareIntentFrame(frame);
bool valid = intent_frame_validate(&frame);
// Should be true
```

### Quick Test: JSON Serialization
```cpp
IntentFrame frame;
frame.emotion.valence = 0.5f;
prepareIntentFrame(frame);

char* json = intent_frame_to_json(&frame);
// Should produce valid JSON
free(json);
```

## Success Criteria

✅ **Build Success**:
- All targets compile
- No linker errors
- FFI header generated
- Libraries linked correctly

✅ **Runtime Success**:
- IntentFrame can be created
- Validation works
- JSON serialization works
- Engines can consume IntentFrame

## Next Steps After Successful Build

1. Run unit tests: `ctest -R intent_ir`
2. Run integration tests: `python tests/intent_ir_integration_test.py`
3. Test example code: `examples/intent_ir_usage_example.cpp`
4. Start using IntentFrame in new code
