# KmiDi Development Guide

**Version:** 1.0  
**Updated:** 2026-01-18  
**Target Audience:** Developers working on KmiDi

## Quick Start

### Prerequisites

- **macOS 10.15+** or **Linux** (Ubuntu 20.04+)
- **8GB RAM** minimum, 16GB recommended
- **10GB free disk space** for development environment
- **Internet connection** for dependency downloads

### One-Command Setup

From repo root (after cloning and with CMake 3.27+, Node 20+, Rust, Python 3.11+ available):

```bash
./scripts/dev-setup.sh
```

This runs:
- **bootstrap.sh** — JUCE submodule sync, CMake/Node version checks, pybind11 hint
- **npm install** — React and Tauri dependencies
- **pip install -e .** — `music_brain` and tools (sync_entities, tests)

To install system tools (CMake, Ninja, Node, Rust, Python) use your OS package manager or [tauri.app](https://tauri.app) / [vitejs.dev](https://vitejs.dev) docs.

### Start Development

```bash
# Full stack: React dev server + Tauri app + Music Brain API (port 8000)
npm run dev:all

# Or run individually:
npm run dev          # React only (http://localhost:1420)
npm run dev:tauri    # Tauri desktop app (uses React dev server)
npm run dev:python   # Music Brain API (http://localhost:8000, docs at /docs)
```

Open http://localhost:1420 in the browser or use the Tauri desktop window.

## Development Environment Details

### System Requirements

**macOS Development:**
- Xcode Command Line Tools
- Homebrew package manager
- macOS SDK 10.15+

**Linux Development:**
- GCC 9+ or Clang 10+
- ALSA/JACK development headers
- X11 development libraries

### Expected tools

Have these available (install via OS package manager or official installers). `dev-setup.sh` uses them; it does not install them:

- **CMake** 3.27+, **Ninja**, **Rust** (stable), **Node** 20+, **Python** 3.11+
- **npm install** (run by dev-setup) adds Tauri CLI, Vite, concurrently, etc.
- **pip install -e .** (run by dev-setup) adds `music_brain`, pybind11, pydantic, uvicorn

## Development Workflows

### Full-Stack Development

Detailed integration/build matrix (React -> Tauri -> KellyFFI -> KellyCore) lives in [`docs/FULL_STACK_BUILD.md`](FULL_STACK_BUILD.md).

**Start All Services:**
```bash
npm run dev:all
```

This starts:
- React development server (port 1420)
- Python Music Brain API server (port 8000)
- Tauri desktop application

(C++ KellyFFI must be built separately when using pipeline B; re-run CMake build after C++ changes.)

**Development URLs:**
- React Frontend: http://localhost:1420
- Python API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Technology-Specific Development

**React frontend only:** `npm run dev:react` (or `npm run dev`) — hot reload, http://localhost:1420

**Tauri desktop only:** `npm run dev:tauri` — launches app; uses React dev server when running together

**Python API only:** `npm run dev:python` — Music Brain API at http://localhost:8000, Swagger at /docs

**C++ (Kelly FFI):** Build from repo root: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_KELLY_CORE=ON -DKMIDI_BUILD_JUCE_UI=ON -DBUILD_PLUGINS=ON` then `cmake --build build --target KellyFFI`. Re-run after C++ changes; Tauri picks up the library from `build/`, `build/debug/`, or `build/release/` per `engine/intent_ir/build.rs`.

**Plugin notes:** Root project uses `BUILD_PLUGINS`/`KMIDI_BUILD_JUCE_UI` (Kelly VST3 path). Legacy `DAIW_BUILD_VST3`/`DAIW_BUILD_AU` options belong to `KmiDi_FINAL/engine/cpp_music_brain` only.

### Build Workflows

Two V1 build paths:

- **V1 pipeline A (penta_core + PyInstaller + Tauri):** from repo root: `./scripts/build_v1.sh`. Builds: sync entities → C++ penta_core / Python bindings → PyInstaller-packaged Music Brain API → Tauri app. Requires Python (pybind11, pyinstaller), Node, Rust, CMake. Does not build KellyFFI.
- **V1 pipeline B (KellyFFI + Tauri):** See [FULL_STACK_BUILD.md](FULL_STACK_BUILD.md) and `./scripts/build-full-stack.sh`. Builds KellyFFI (and optional KellyPlugin_VST3) for native desktop integration (React → Tauri → KellyFFI → KellyCore).

**Tauri desktop app only** (after C++ KellyFFI is built for pipeline B):
```bash
npm ci && npm run tauri build
```

**Component builds:**
- `npm run build` — React frontend only
- CMake: `cmake --build build --target KellyFFI` — FFI library for Tauri (pipeline B)
- `./scripts/build-all.sh` — Full multi-technology stack (see script for options)

**API/schema (UI–engine contract):** The single source of truth is `shared_schemas/CompleteSongIntentRequest.json`. Run `python3 scripts/sync_entities.py` after schema changes; CI verifies no drift between JSON, `src/types/Intent.ts`, and `engine/intent_ir/src/generated/intent.rs`. Python validation: `pytest tests/unit/test_api_schema.py`.

## Code Organization

### C++ Development

**Directory Structure:**
```
src/
├── engine/          # Kelly Brain AI system
│   ├── KellyBrain.h/.cpp
│   ├── EmotionThesaurus.h/.cpp
│   └── [50+ AI components]
├── engines/         # Specialized music engines
│   ├── MelodyEngine.h/.cpp
│   ├── BassEngine.h/.cpp
│   └── [22 more engines]
├── dsp/            # DSP primitives
├── audio/          # Audio I/O
├── bridge/         # FFI interface
└── plugin/         # Plugin implementations
```

**Coding Standards:**
- C++20 standard with modern features
- RAII for resource management
- `const`-correctness everywhere
- No exceptions in audio thread
- Comprehensive documentation

**Key Files:**
- `src/engine/KellyBrain.h` - Main AI interface
- `src/bridge/kelly_ffi.h` - C FFI interface
- `CMakeLists.txt` - Build configuration

### Rust Development

**Directory Structure:**
```
engine/intent_ir/src/
├── lib.rs           # Library entry
├── ffi.rs           # FFI exports
├── builder.rs       # Intent builder
├── types.rs         # Core types
├── validator.rs     # Validation
└── generated/       # Auto-generated intent types
```

**Coding Standards:**
- Idiomatic Rust patterns
- Error handling with `Result` types
- Comprehensive documentation
- Memory safety enforced by compiler
- Async/await for non-blocking operations

**Key Files:**
- `engine/intent_ir/src/ffi.rs` - FFI exports
- `engine/intent_ir/src/bridge/kelly_ffi.rs` - Safe FFI wrappers
- `engine/intent_ir/build.rs` - Build configuration

### React Development

**Directory Structure:**
```
src/
├── components/      # React components
│   ├── EmotionWheel.tsx
│   ├── GuideNav.tsx
│   └── SpectoCloudPanel.tsx
├── hooks/          # Custom hooks
│   ├── useKellyBrain.ts
│   └── useMusicBrain.ts
├── App.tsx         # Main application
└── main.tsx        # Entry point
```

**Coding Standards:**
- Functional components with hooks
- TypeScript strict mode
- Tailwind CSS for styling
- Props interface definitions
- Comprehensive error handling

**Key Files:**
- `src/App.tsx` - Main application component
- `src/hooks/useKellyBrain.ts` - C++ backend integration
- `src/hooks/useMusicBrain.ts` - Hybrid API integration

## Debugging Guide

### C++ Debugging

**Setup Debugging:**
```bash
# Build with debug symbols
cd build/debug
cmake ../../ -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j4

# Run with debugger
lldb ./KellyTests  # macOS
gdb ./KellyTests   # Linux
```

**VS Code Debugging:**
1. Set breakpoints in C++ files
2. Use "Debug C++ Tests" launch configuration
3. Step through code with full symbol information

**Common Issues:**
- **Null pointer crashes:** Check FFI parameter validation
- **Memory leaks:** Run with AddressSanitizer (`KMIDI_ENABLE_ASAN=ON`)
- **Audio glitches:** Profile with Tracy (`ENABLE_TRACY=ON`)

### Rust Debugging

**Environment Variables:**
```bash
RUST_LOG=debug              # Detailed logging
RUST_BACKTRACE=1           # Stack traces on panic
RUST_LOG=tauri=trace       # Tauri-specific debugging
```

**VS Code Debugging:**
1. Use rust-analyzer extension
2. Set breakpoints in Rust files
3. Debug Tauri commands directly

**Common Issues:**
- **FFI crashes:** Check C++ library availability
- **Linking errors:** Verify `build.rs` configuration
- **Command failures:** Check parameter serialization

### React Debugging

**Browser DevTools:**
- Use React DevTools extension
- Monitor component state and props
- Profile render performance
- Network tab for Tauri command calls

**Console Debugging:**
```javascript
// Enable detailed logging
localStorage.setItem('debug', 'kelly:*');

// Monitor hook state
const { state, error } = useKellyBrain();
console.log('KellyBrain state:', state);
```

**Common Issues:**
- **Hook errors:** Check Tauri command availability
- **State sync issues:** Monitor event listeners
- **Performance:** Use React Profiler

### Integration Debugging

**FFI Boundary:**
```bash
# Check library loading
otool -L engine/intent_ir/resources/libKellyFFI.dylib  # macOS
ldd engine/intent_ir/resources/libKellyFFI.so          # Linux

# Test FFI directly
cd build/debug && ./KellyTests                  # C++ side
cd engine/intent_ir && cargo test kelly_ffi           # Rust side
```

**Event System:**
```bash
# Monitor events in browser console
# Events will appear as Tauri events
# Check event frequency and data

# Check state synchronization
curl http://localhost:1420/api/state  # If debug endpoint available
```

## Performance Optimization

### Profiling

**C++ Profiling:**
```bash
# Enable Tracy profiling
cmake -DENABLE_TRACY=ON
make -j4

# Run with profiler
./KellyApp  # Tracy will connect automatically
```

**Rust Profiling:**
```bash
# CPU profiling
cargo build --release
perf record ./target/release/idaw  # Linux
Instruments.app                    # macOS

# Memory profiling
valgrind --tool=massif ./target/release/idaw
```

**React Profiling:**
```bash
# React DevTools Profiler
# Measure component render times
# Identify expensive re-renders
# Optimize with useMemo/useCallback
```

### Optimization Strategies

**C++ Optimizations:**
- Enable AVX2 SIMD (default on supported platforms; see DSP code in `libs/daiw/` and `include/penta/`)
- Release builds: `CMAKE_BUILD_TYPE=Release`
- Profile-guided optimization (PGO)
- Cache-friendly data structures

**Rust Optimizations:**
- Release builds with LTO: `cargo build --release`
- FFI call minimization
- Batch state updates
- Async operation optimization

**React Optimizations:**
- Memoization with `useMemo`/`useCallback`
- State update batching
- Lazy loading for large components
- Efficient event handler cleanup

## Testing Guide

### Running Tests

**Python (repo test suite):**
```bash
python3 -m pytest tests/
```

**Schema/API contract (UI–engine):**
```bash
python3 -m pytest tests/unit/test_api_schema.py
```

**Rust/Tauri:** from repo root, `cd engine/intent_ir && cargo test`.

**C++ (when BUILD_TESTS=ON):** `ctest --test-dir build --output-on-failure`.

### Writing Tests

**C++ Tests (Catch2):**
```cpp
#include <catch2/catch.hpp>
#include "engine/KellyBrain.h"

TEST_CASE("KellyBrain initialization") {
    kelly::KellyBrain brain;
    REQUIRE(brain.initialize("./test-data"));
}
```

**Rust Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kelly_ffi_creation() {
        let brain = KellyBrain::new();
        assert!(brain.is_ok());
    }
}
```

**React Tests (Vitest):**
```typescript
import { describe, it, expect } from 'vitest';
import { useKellyBrain } from '../hooks/useKellyBrain';

describe('useKellyBrain', () => {
  it('should initialize correctly', () => {
    // Test hook behavior
  });
});
```

### Integration Testing

**FFI Integration:**
- Test C++ ↔ Rust communication
- Validate memory management
- Check error handling
- Performance benchmarking

**Command Integration:**
- Test Tauri commands end-to-end
- Validate parameter serialization
- Check async operation handling
- Error propagation testing

## Troubleshooting

### Build Issues

**CMake Configuration Errors:**
```bash
# Clear CMake cache and reconfigure
rm -rf build/CMakeCache.txt
cd build && cmake ..

# Check for missing dependencies
cmake .. -DCMAKE_FIND_DEBUG_MODE=ON
```

**Rust Compilation Errors:**
```bash
# Update Rust toolchain
rustup update

# Clear Cargo cache
cargo clean

# Verbose compilation
cargo build --verbose
```

**Node Build Errors:**
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Runtime Issues

**Library Loading Errors:**
```bash
# Check library paths (macOS)
otool -L engine/intent_ir/resources/libKellyFFI.dylib
export DYLD_LIBRARY_PATH=./build/debug:$DYLD_LIBRARY_PATH

# Check library paths (Linux)
ldd engine/intent_ir/resources/libKellyFFI.so
export LD_LIBRARY_PATH=./build/debug:$LD_LIBRARY_PATH
```

**Audio Issues:**
- Check audio device permissions
- Verify Core Audio/ALSA configuration
- Test with simple audio applications first
- Check sample rate compatibility

**Plugin Issues:**
```bash
# Check plugin validation
pluginval KellyPlugin.vst3  # If available

# Test in simple host
# Use JUCE AudioPluginHost for initial testing

# Check plugin installation paths
ls ~/Library/Audio/Plug-Ins/VST3/  # macOS
ls ~/.vst3/                        # Linux
```

### Performance Issues

**Identify Bottlenecks:**
```bash
# Profile C++ code
# Enable Tracy or use system profilers

# Profile Rust code
cargo build --release
perf record ./target/release/idaw

# Profile React code
# Use browser DevTools Profiler
```

**Common Performance Fixes:**
- Reduce FFI call frequency
- Batch state updates
- Use appropriate data structures
- Enable compiler optimizations

## Contributing

### Code Style

**C++ Style:**
- Follow Google C++ Style Guide
- Use clang-format with project configuration
- Comprehensive documentation with Doxygen
- Unit tests for all public APIs

**Rust Style:**
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow Rust API Guidelines
- Comprehensive documentation with `cargo doc`

**TypeScript Style:**
- Use Prettier for formatting
- Follow React/TypeScript best practices
- Use ESLint for code quality
- Props interfaces for all components

### Pull Request Process

1. **Setup Development Environment:**
   ```bash
   ./scripts/dev-setup.sh
   ```

2. **Create Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development:**
   ```bash
   # Work on your feature
   npm run dev:all
   
   # Test your changes (Python)
   python3 -m pytest tests/
   # Optional: cd engine/intent_ir && cargo test
   ```

4. **Build Verification:**
   ```bash
   # Frontend build
   npm run build
   # Native (pipeline B): ./scripts/build-full-stack.sh
   # Plugins: cmake --build build --target KellyPlugin_VST3
   ```

5. **Submit Pull Request:**
   - Include comprehensive description
   - Add tests for new functionality
   - Update documentation if needed
   - Verify CI passes

### Testing Guidelines

**Unit Tests Required:**
- All public C++ APIs
- All Rust FFI functions
- All Tauri commands
- React hooks and utilities

**Integration Tests Required:**
- C++ ↔ Rust FFI integration
- Tauri commands end-to-end
- React ↔ Backend communication
- Plugin loading in DAWs

**Performance Tests Required:**
- Audio processing latency
- UI responsiveness
- Memory usage
- FFI call overhead

## Advanced Topics

### Adding New C++ Features

1. **Implement in C++:**
   ```cpp
   // src/engine/YourNewFeature.h
   class YourNewFeature {
   public:
       void processAudio(float* buffer, int samples);
   };
   ```

2. **Add FFI Wrapper:**
   ```cpp
   // src/bridge/kelly_ffi.h
   extern "C" {
       int your_new_feature_process(KellyBrain* brain, float* data, int size);
   }
   
   // src/bridge/kelly_ffi.cpp
   int your_new_feature_process(KellyBrain* brain, float* data, int size) {
       // Implementation
   }
   ```

3. **Add Rust Binding:**
   ```rust
   // engine/intent_ir/src/bridge/kelly_ffi.rs
   extern "C" {
       fn your_new_feature_process(brain: *mut KellyBrainHandle, data: *mut f32, size: c_int) -> c_int;
   }
   
   impl KellyBrain {
       pub fn process_new_feature(&mut self, data: &mut [f32]) -> KellyResult<()> {
           // Safe wrapper
       }
   }
   ```

4. **Add Tauri Command:**
   ```rust
   // engine/intent_ir/src/commands.rs
   #[command]
   pub async fn process_new_feature(data: Vec<f32>) -> Result<Vec<f32>, String> {
       // Implementation
   }
   ```

5. **Add React Hook:**
   ```typescript
   // src/hooks/useKellyBrain.ts
   const processNewFeature = useCallback(async (data: number[]) => {
       return await invoke('process_new_feature', { data });
   }, []);
   ```

### Adding New UI Features

1. **Create React Component:**
   ```typescript
   // src/components/YourNewComponent.tsx
   interface YourNewComponentProps {
       // Props definition
   }
   
   export const YourNewComponent: React.FC<YourNewComponentProps> = (props) => {
       // Implementation
   };
   ```

2. **Add to Main App:**
   ```typescript
   // src/App.tsx
   import { YourNewComponent } from './components/YourNewComponent';
   
   // Add to JSX
   ```

3. **Style with Tailwind:**
   ```css
   /* Use existing design tokens */
   className="bg-bg-primary text-text-primary border-border-light"
   ```

### Plugin Development

**Create New Plugin Type:**
1. Update CMakeLists.txt with new plugin target
2. Implement JUCE plugin processor
3. Add plugin-specific UI
4. Test in target DAW
5. Update installation scripts

**Plugin Testing Workflow:**
```bash
# Build plugin (from repo root, after CMake configure with BUILD_PLUGINS=ON)
cmake --build build --target KellyPlugin_VST3

# Install locally
cp build/KellyPlugin_artefacts/Release/VST3/*.vst3 ~/Library/Audio/Plug-Ins/VST3/

# Test in DAW
open /Applications/Logic\ Pro.app
# or
open /Applications/Reaper.app
```

## External sources and dataset layout

External assets (datasets, weights, benchmarks) and the canonical dataset volume layout are documented and driven by manifest + scripts so they cooperate with existing structure.

- **Dataset layout:** Set `KMIDI_DATASETS_PATH` to the Datasets root that contains `by_source/`, `by_domain/`, etc. See [docs/DATASETS_LAYOUT.md](DATASETS_LAYOUT.md) for the full layout (by_source per source, downloads/raw/processed, and how prepare_datasets vs acquisition scripts use it).
- **Source manifest:** [config/source_manifest.yaml](../config/source_manifest.yaml) lists external source items (verification status, storage path, license, adoption). No downloads until primary URLs and licenses are verified.
- **Briefings:** One briefing per source item in [docs/research/sources/](research/sources/) (exact template, verification_basis).
- **Acquisition script:** `python scripts/acquire/acquire_from_manifest.py --list` and `--resolve-paths` / `--dry-run` resolve storage paths from the manifest; dataset-like items use `$KMIDI_DATASETS_PATH/by_source/<source_item>/downloads`. No fetch is performed until URLs are in the manifest and approved. See [scripts/acquire/README.md](../scripts/acquire/README.md).
- **Plan and phases:** [docs/SOURCE_INTEGRATION_PLAN.md](SOURCE_INTEGRATION_PLAN.md) describes integration and download phases; Phase 3 (this wiring) is low-risk only.

## Environment Variables

### Build Configuration

```bash
# Build system
BUILD_TYPE=Debug|Release        # C++ build type
BUILD_TESTS=ON|OFF             # Enable test building
BUILD_PLUGINS=ON|OFF           # Enable plugin building
CLEAN_BUILD=true|false         # Clean before building
PARALLEL_JOBS=N                # Number of parallel build jobs

# Runtime configuration
RUST_LOG=debug                 # Rust logging level
DYLD_LIBRARY_PATH=./build      # macOS library path
LD_LIBRARY_PATH=./build        # Linux library path

# Development
TAURI_DEV_HOST=localhost       # Tauri development host
HOT_RELOAD=true               # Enable hot reload
```

### API Configuration

```bash
# Python API
MUSIC_BRAIN_API_HOST=127.0.0.1
MUSIC_BRAIN_API_PORT=8000
MUSIC_BRAIN_DATA_PATH=./data

# Kelly Brain
KELLY_DATA_PATH=./data
KELLY_LOG_LEVEL=info
KELLY_ENABLE_PROFILING=false
```

## IDE Configuration

### Visual Studio Code

The development setup creates `.vscode/` configuration:

**Extensions Recommended:**
- rust-analyzer (Rust support)
- C++ extension pack (C++ support)
- Tauri (Tauri development)
- ES7+ React snippets (React development)

**Configured Tasks:**
- Build All (Debug/Release)
- Start Dev Servers
- Build C++ Only
- Run Tests

**Debug Configurations:**
- Debug Tauri App
- Debug C++ Tests
- Debug Rust Tests

### Alternative IDEs

**CLion (C++ focused):**
- Open CMakeLists.txt as project
- CMake integration works automatically
- Excellent debugging capabilities
- Built-in profiling tools

**IntelliJ IDEA (Rust plugin):**
- Rust plugin for advanced Rust support
- Integrated terminal for multi-technology work
- Database tools for data analysis

## Security Considerations

### Development Security

**API Access:**
- Python API runs on localhost only
- No external network access by default
- Tauri CSP configured for development

**File Access:**
- Tauri file system API restricted
- User must grant permissions
- No automatic file system access

**Plugin Security:**
- Audio thread isolation
- Parameter validation
- Host compatibility verification
- No network access from plugins

### Production Security

**Code Signing (macOS):**
```bash
# Sign application
export DEVELOPER_ID="Your Developer ID"
./scripts/build-all.sh --sign

# Notarize for distribution  
export APPLE_ID="your@email.com"
export APPLE_TEAM_ID="TEAMID"
./scripts/build-all.sh --sign --notarize
```

**Distribution:**
- All libraries embedded in application
- No external dependencies required
- Secure update mechanisms
- User data kept local

---

This guide covers the essential aspects of KmiDi development. For specific technical questions, refer to:

- `docs/ARCHITECTURE.md` - Overall system design
- `docs/API.md` - API reference documentation
- Individual component READMEs
- Inline code documentation