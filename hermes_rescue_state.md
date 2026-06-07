# Hermes Rescue State

Branch: hermes-overnight-rescue
Repo: /Users/seanburdges/Dev/KmiDi
Intent IR rule: preserve Intent IR semantics and C ABI behavior; only harden ownership/lifetime/resource handling.

Validated rescue commits so far:
- f7afd209 rescue(intent-ir): remove transient builder heap churn and fix Rust staticlib relink
- 50e6883f rescue(kelly-ffi): tighten C ABI lifetime handling with RAII helpers
- 03b10efc rescue(kelly-ffi): normalize wrapper access through helper casts
- 0525c2ec rescue(plugin-state): encapsulate DynamicObject allocation in preset JSON serialization
- 656b5757 rescue(plugin-ui): make editor ownership transfer explicit
- 972d294c rescue(bridge-project): wrap JSON and PyObject lifetimes with local RAII helpers
- 86f9eb4d rescue(plugin-ir): guard async inspector callbacks against dangling component lifetime
- pending-next: bridge Python subclass RAII hardening (ContextBridge, IntentBridge)

Validated builds/tests so far:
- engine/intent_ir: cargo test passing
- build-rescue: cmake --build build-rescue --target KellyFFI -j4 passing
- build: cmake --build build --target KellyCore -j4 passing
- build: cmake --build build --target KellyPlugin -j4 passing

FFI boundaries already secured:
- Rust intent_ir FFI handle now stores IntentFrameBuilder inline, using core::mem::take ownership transitions
- src/bridge/intent_ir_ffi.cpp no longer defines local rust_eh_personality stub; staticlib relink is clean
- src/bridge/kelly_ffi.cpp now uses RAII for malloc-owned strings and explicit wrapper helper casts

Current scan mode:
- broad regex sweeps exhausted easy allocation hits
- pivoted to file-by-file inspection in small batches

Target file inventory:
- src/project: ProjectFile.cpp, ProjectManager.cpp, ProjectManager.h
- src/plugin cpp: PluginProcessor.cpp, PluginState.cpp, PluginEditor.cpp, MasterEQProcessor.cpp, PluginLogger.cpp, PluginTestHarness.cpp, PluginIRInspector.cpp, HostDebugger.cpp
- src/plugin h: PluginProcessor.h, PluginState.h, PluginEditor.h, MasterEQProcessor.h, PluginTestHarness.h, PluginLogger.h, PluginIRInspector.h, plugin_processor.h, plugin_editor.h, HostDebugger.h
- src/bridge cpp: kelly_ffi.cpp, intent_ir_ffi.cpp, OrchestratorBridge.cpp, SuggestionBridge.cpp, StateBridge.cpp, PythonBridgeBase.cpp, PreferenceBridge.cpp, IntentBridge.cpp, EngineIntelligenceBridge.cpp, ContextBridge.cpp, OSCBridge.cpp, kelly_bridge.cpp, BridgeBase.cpp, CacheManager.cpp
- src/bridge h: OrchestratorBridge.h, kelly_ffi.h, StateBridge.h, PythonBridgeBase.h, PyGILGuard.h, OSCBridge.h, SuggestionBridge.h, PreferenceBridge.h, IntentBridge.h, intent_ir_ffi.h, EngineIntelligenceBridge.h, ContextBridge.h, CacheManager.h, BridgeBase.h

Current position in file-by-file scan:
1. PluginProcessor.cpp inspected and hardened at createEditor boundary
2. PluginState.cpp inspected and hardened around DynamicObject creation helper
3. ProjectFile.cpp inspected and hardened around DynamicObject creation helper
4. PythonBridgeBase.cpp inspected and hardened with local PyObject RAII in callPythonFunction
5. PluginIRInspector.cpp inspected and hardened: SafePointer replaces raw this in async UI callback; syntax-checked with plugin compile flags after fixing JUCE FontOptions usage
6. HostDebugger.cpp and PluginLogger.cpp inspected; no new ownership fixes required in this pass
7. PluginTestHarness.cpp and MasterEQProcessor.cpp inspected; no ownership fixes required in this pass
8. ContextBridge.cpp and IntentBridge.cpp inspected and hardened with local PyObject RAII around tuple/result construction and Python call return paths
9. StateBridge.cpp and SuggestionBridge.cpp inspected and hardened with local PyObject RAII around tuple/result construction and worker-thread Python dispatch
10. PreferenceBridge.cpp and EngineIntelligenceBridge.cpp inspected and hardened with local PyObject RAII around module/class/method acquisition, tuple construction, and Python call return paths
11. OrchestratorBridge.cpp inspected and hardened with local PyObject RAII around module import, execute/status/cancel tuple construction, and return-path cleanup
12. OSCBridge.cpp inspected and hardened by funneling DynamicObject allocation through a local helper for request/error payload construction
13. kelly_bridge.cpp inspected; pybind11 ownership appears declarative and clean in this pass
14. kelly_ffi.cpp inspected and hardened: wrapper initialization state is now atomic for cross-thread FFI reads; rebuilt KellyFFI successfully
15. intent_ir_ffi.cpp inspected and hardened: music update validation/clamp now occurs outside the state mutex, mirroring emotion update and reducing cross-FFI deadlock risk while preserving Intent IR semantics
16. ProjectManager.cpp inspected and hardened: JSON builders now funnel DynamicObject allocation through a local helper across project, MIDI, note, chord, and vocal-note serialization paths
17. Current search frontier: residual plugin/project ownership patterns after ProjectManager cleanup (createPluginFilter API exception still expected)

Inspection heuristics for next batches:
- raw pointer ownership hidden behind typedefs or factory methods
- manual lock/unlock or resource acquire/release without guard objects
- owning pointers passed by reference without clear lifetime contract
- malloc/free/new/delete hidden in helper methods or third-party wrapper usage
- Python C API refcount edges lacking RAII/GIL guards
- JUCE object creation APIs that require raw returns: isolate allocation in unique_ptr and release only at boundary
