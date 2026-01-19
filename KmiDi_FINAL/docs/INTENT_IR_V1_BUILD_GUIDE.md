# Intent IR v1 Build Guide

## Prerequisites

1. **Rust toolchain** (for validator)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **CMake 3.20+** (for build system)

3. **cJSON** (optional, for JSON serialization)
   ```bash
   # macOS
   brew install cjson
   
   # Or build from source: https://github.com/DaveGamble/cJSON
   ```

## Build Steps

### 1. Verify Setup

```bash
cd KmiDi_FINAL
./scripts/verify_intent_ir_build.sh
```

### 2. Configure CMake

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

The build system will:
- Build Rust crate (`engine/intent_ir`)
- Generate C FFI header via cbindgen
- Build C++ adapter (`intent_ir_adapter`)
- Link into `KellyCore`

### 3. Build

```bash
cmake --build . -j$(nproc)
```

### 4. Verify Build

Check that these targets were built:
- `intent_ir_rust_lib` (Rust static library)
- `intent_ir_adapter` (C++ adapter library)
- `KellyCore` (should link intent_ir_adapter)

## Troubleshooting

### Rust Crate Won't Build

**Error**: `cargo: command not found`
- **Solution**: Install Rust toolchain

**Error**: `cbindgen: command not found`
- **Solution**: `cargo install cbindgen` or let CMake install it

### CMake Can't Find Rust

**Error**: `CARGO_EXECUTABLE not found`
- **Solution**: Ensure `cargo` is in PATH
- **Check**: `which cargo`

### FFI Header Not Generated

**Error**: Missing `intent_ir_ffi.h`
- **Solution**: Check that cbindgen ran successfully
- **Location**: Should be in `build/include/intent_ir_ffi.h`

### JSON Serialization Fails

**Error**: `cJSON.h: No such file`
- **Solution**: Install cJSON or disable JSON features
- **Workaround**: JSON is optional - IntentFrame works without it

### Link Errors

**Error**: `undefined reference to clamp_intent_frame_ffi`
- **Solution**: Ensure Rust library is linked
- **Check**: `target_link_libraries(intent_ir_adapter ${INTENT_IR_RUST_LIB})`

## Build Verification Checklist

- [ ] Rust toolchain installed (`cargo --version`)
- [ ] CMake finds cargo (`cmake ..` shows no errors)
- [ ] Rust crate builds (`cargo build` in `engine/intent_ir`)
- [ ] FFI header generated (`build/include/intent_ir_ffi.h` exists)
- [ ] C++ adapter compiles
- [ ] KellyCore links successfully
- [ ] Tests compile (if BUILD_TESTS=ON)

## Development Workflow

### Making Changes to Rust Validator

1. Edit Rust code in `engine/intent_ir/src/`
2. Run `cargo build` in `engine/intent_ir/` to test
3. CMake will rebuild automatically on next build

### Making Changes to C++ Adapter

1. Edit `engine/src/common/IntentIRAdapter.*`
2. Rebuild: `cmake --build build`

### Testing Changes

```bash
# Python tests
python tests/intent_ir_integration_test.py

# C++ tests (if BUILD_TESTS=ON)
cd build
ctest -R intent_ir
```

## Performance Verification

After building, verify audio thread safety:

```cpp
// In your audio callback
void processBlock(..., const IntentFrame& frame) {
    // This should compile and run without allocations
    float bias = frame.music.tempo_bias;
    // ...
}
```

Use memory profiler to verify:
- No `new`/`malloc` calls
- No `std::string`/`std::vector` operations
- Frame is const reference
