# Quick Build Checklist

Quick verification checklist after improvements.

## Pre-Build Checks

- [ ] Rust toolchain installed (`cargo --version`)
- [ ] CMake 3.27+ installed (`cmake --version`)
- [ ] JUCE available in `build/external/JUCE`
- [ ] Qt6 installed (for desktop build)

## Build Steps

```bash
cd /Users/seanburdges/KmiDi-1/KmiDi_FINAL
mkdir -p build && cd build
cmake .. -DBUILD_KMIDI_CORE=ON -DBUILD_TESTS=ON
cmake --build . -j$(sysctl -n hw.ncpu)
```

## Verification Checklist

### Rust Integration
- [ ] Rust library builds: `libintent_ir.a` generated
- [ ] FFI header generated: `include/intent_ir_ffi.h` exists
- [ ] C++ adapter links to Rust library
- [ ] No undefined symbol errors

### JUCE FFT
- [ ] JUCE DSP links to `prrot_core`
- [ ] SpectralAnalyzer compiles with JUCE FFT
- [ ] No FFT-related linker errors

### Code Quality
- [ ] No TODO warnings in critical paths
- [ ] IntentIRAdapter uses Rust validator (check code)
- [ ] KellyBrain derives complexity/feel (check code)

### Build Output
- [ ] No compilation errors
- [ ] No linker errors
- [ ] All targets build successfully

## Quick Test

After build, verify:

```bash
# Check Rust library exists
ls build/rust_target/*/release/libintent_ir.a

# Check FFI header generated
ls build/include/intent_ir_ffi.h

# Check binaries built
ls build/KellyCore*  # or appropriate target name
```

## Common Issues

### FFI Header Not Found
- **Fix:** Ensure Rust builds before C++ adapter
- **Check:** `add_dependencies(intent_ir_adapter intent_ir_rust_lib)`

### JUCE FFT Not Found
- **Fix:** Ensure `juce::juce_dsp` linked to `prrot_core`
- **Check:** CMakeLists.txt line ~186

### Rust Build Fails
- **Fix:** Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Check:** `cargo --version`

## Success Criteria

✅ All targets build
✅ No undefined symbols
✅ FFI integration works
✅ FFT compiles and links
✅ No critical TODOs remain

---

**See:** `docs/BUILD_ANALYSIS_AND_IMPROVEMENTS.md` for detailed analysis
