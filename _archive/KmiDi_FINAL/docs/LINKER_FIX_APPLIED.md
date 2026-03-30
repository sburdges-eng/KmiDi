# Linker Fix Applied

**Date:** 2026-01-22
**Issue:** Linker errors for `KellyPlugin` and `KellyApp` - `ld: library 'intent_ir_rust' not found`

## Problem

The Rust static library (`libintent_ir.a`) was built successfully, but the final executables (`KellyPlugin` and `KellyApp`) couldn't find it during linking.

**Root Cause:**
- `intent_ir_rust` is an IMPORTED library
- The library directory wasn't in the linker search path for final executables
- Transitive dependencies through static libraries need explicit handling

## Solution Applied

### 1. Fixed Duplicate Entries in KellyCore

**Before:**
```cmake
target_link_libraries(KellyCore PUBLIC
    intent_ir_adapter
    # ... other libs ...
    intent_ir_adapter  # Duplicate!
    prrot_core  # Duplicate!
)
```

**After:**
```cmake
target_link_libraries(KellyCore PUBLIC
    intent_ir_adapter  # Intent IR adapter with Rust validator (includes intent_ir_rust)
    # ... other libs ...
)
```

### 2. Added Library Directory to Linker Search Path

**Added to `intent_ir_adapter`:**
```cmake
# Ensure Rust library directory is in linker search path
if(TARGET ${INTENT_IR_RUST_LIB})
    get_target_property(RUST_LIB_LOCATION ${INTENT_IR_RUST_LIB} IMPORTED_LOCATION)
    if(RUST_LIB_LOCATION)
        get_filename_component(RUST_LIB_DIR ${RUST_LIB_LOCATION} DIRECTORY)
        target_link_directories(intent_ir_adapter PUBLIC ${RUST_LIB_DIR})
    endif()
endif()
```

This ensures that the Rust library directory is added to the linker search path and propagated to all targets that link to `intent_ir_adapter`.

### 3. Added Explicit Dependencies

**For `KellyApp`:**
```cmake
target_link_libraries(KellyApp PRIVATE
    KellyCore  # Includes intent_ir_adapter and intent_ir_rust transitively
    Qt6::Core
    Qt6::Widgets
)

# Ensure Rust library is linked (transitive dependency through KellyCore)
add_dependencies(KellyApp intent_ir_rust_lib)
```

**For `KellyPlugin`:**
```cmake
target_link_libraries(KellyPlugin PRIVATE
    KellyCore  # Includes intent_ir_adapter and intent_ir_rust transitively
    juce::juce_audio_plugin_client
)

# Ensure Rust library is linked (transitive dependency through KellyCore)
add_dependencies(KellyPlugin intent_ir_rust_lib)
```

## Changes Made

### Files Modified

1. **`CMakeLists.txt`**
   - Removed duplicate entries in `KellyCore` link libraries
   - Added `target_link_directories` to `intent_ir_adapter`
   - Added explicit dependencies on `intent_ir_rust_lib` for `KellyApp` and `KellyPlugin`

### Dependency Chain

```
intent_ir_rust_lib (custom target)
    ↓
intent_ir_rust (IMPORTED library)
    ↓
intent_ir_adapter (static library)
    ↓
KellyCore (static library)
    ↓
KellyApp / KellyPlugin (executables)
```

## Verification

To verify the fix:

1. **Clean build:**
   ```bash
   rm -rf build/
   cmake -B build -S . -DBUILD_KMIDI_CORE=ON -DBUILD_DESKTOP=ON -DBUILD_PLUGINS=ON
   cmake --build build
   ```

2. **Check for linker errors:**
   - Should see successful linking of `KellyApp`
   - Should see successful linking of `KellyPlugin`

3. **Verify Rust library is found:**
   ```bash
   # Check that library exists
   ls -la build/rust_target/*/release/libintent_ir.a

   # Check linker command includes library path
   cmake --build build --verbose 2>&1 | grep intent_ir
   ```

## Expected Behavior

After this fix:
- ✅ `intent_ir_rust_lib` builds successfully
- ✅ `intent_ir_adapter` links to Rust library
- ✅ `KellyCore` links to `intent_ir_adapter`
- ✅ `KellyApp` and `KellyPlugin` can find Rust library during linking
- ✅ All executables build successfully

## Technical Details

### Why This Works

1. **`target_link_directories` with PUBLIC:**
   - Adds the Rust library directory to the linker search path
   - `PUBLIC` visibility ensures it propagates to all dependents
   - This is necessary because IMPORTED libraries don't automatically add their directory to the search path

2. **Explicit Dependencies:**
   - `add_dependencies` ensures `intent_ir_rust_lib` is built before linking
   - This is a build-time dependency, not a link-time dependency
   - Ensures the library file exists when the linker tries to find it

3. **Transitive Linking:**
   - Static libraries in CMake don't automatically propagate all dependencies
   - By using `PUBLIC` visibility and explicit directory paths, we ensure the Rust library is found

## Alternative Approaches Considered

1. **Using `target_link_options`:**
   - Could add `-L${RUST_LIB_DIR}` directly
   - Less portable across platforms
   - `target_link_directories` is more CMake-idiomatic

2. **Linking Rust library directly to executables:**
   - Would work but violates encapsulation
   - Makes the dependency chain less clear
   - Current approach is cleaner

3. **Using `INTERFACE` library:**
   - Could create an interface library for the Rust library
   - More complex, current approach is sufficient

## Testing

After applying this fix, the build should complete successfully:

```bash
# Full build
cmake -B build -S . \
    -DBUILD_KMIDI_CORE=ON \
    -DBUILD_DESKTOP=ON \
    -DBUILD_PLUGINS=ON

cmake --build build

# Should see:
# [100%] Built target KellyApp
# [100%] Built target KellyPlugin_VST3
```

## Status

✅ **Fix Applied** - Ready for testing

---

**See Also:**
- `docs/BUILD_RESULTS.md` - Original build results
- `docs/BUILD_VERIFICATION_STATUS.md` - Build verification checklist
