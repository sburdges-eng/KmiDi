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
- 68d2eecf rescue(biometric-voice): replace manual bridge/JSON heap ownership with RAII
- 923b507f rescue(core-queue): wrap transient queue-node allocation in RAII before ownership handoff
- 7cc4648f rescue(ui-tooltip): guard singleton creation with RAII before shutdown-managed handoff
- 8172fa28 rescue(penta-rtlogger): make worker lifecycle idempotent across secondary native core
- 4ebbabe2 rescue(penta-worker): make audio worker lifecycle idempotent before thread handoff
- acb92b30 rescue(penta-ml): make inference worker lifecycle restart-safe
- 365dd5b1 rescue(audio-emotion): make primary inference runner lifecycle restart-safe
- 331c8651 rescue(raw-boundaries): harden queue teardown, pool pointer validation, and state worker lifecycle
- e7d7e4d9 rescue(async-retention): make ML and orchestrator worker pruning completion-aware
- 4c40f764 rescue(thread-helpers): normalize inference-thread restart and drop dead biometric thread state
- pending-next: RT/FFI boundary coherence cleanup (Intent IR session sync + OSC pending-request send-failure hygiene)

Validated builds/tests so far:
- engine/intent_ir: cargo test passing
- build-rescue: cmake --build build-rescue --target KellyFFI -j4 passing
- build: cmake --build build --target KellyCore -j4 passing
- build: cmake --build build --target KellyPlugin -j4 passing
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after biometric/voice refactor
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after core/memory queue RAII hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after tooltip singleton RAII hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after src_penta-core RTLogger lifecycle hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after penta AudioWorkerThread lifecycle hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after penta MLInterface lifecycle hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after AudioEmotionRunner lifecycle hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after LockFreeQueue/RTMemoryPool/StateBridge raw-boundary hardening
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after async-thread retention cleanup in MLBridge/OrchestratorBridge
- build-rescue: cmake --build build-rescue --target KellyCore -j4 passing after remaining thread-owner cleanup in InferenceThreadManager/BiometricInput
- build-rescue: cmake --build build-rescue --target KellyCore KellyFFI -j4 passing after Intent IR session coherence and OSC send-failure cleanup

FFI boundaries already secured:
- Rust intent_ir FFI handle now stores IntentFrameBuilder inline, using core::mem::take ownership transitions
- src/bridge/intent_ir_ffi.cpp no longer defines local rust_eh_personality stub; staticlib relink is clean
- src/bridge/kelly_ffi.cpp now uses RAII for malloc-owned strings and explicit wrapper helper casts
- kelly_ffi malloc/free contract rechecked this pass: header docs match implementation; returned char* paths funnel through string_to_c_str() and are released by kelly_free_string()

Current scan mode:
- broad regex sweeps exhausted easy allocation hits
- pivoted to file-by-file inspection in small batches
- widened sweep into src_penta-core and adjacent penta-backed src/ support code after main src/ hot paths cooled
- now working through intentional raw-boundary sites one by one, only changing them when lifetime safety improves without changing ABI or RT semantics

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
17. RTLogger.cpp inspected and hardened: start/stop are now idempotent and protected against double-start thread ownership hazards
18. AudioEmotionRunner.cpp first pass inspected; worker lifecycle looked paired
19. BiometricInput.h/.cpp/.mm inspected and hardened: bridge ownership moved from void* + manual delete to std::unique_ptr across both C++ and Objective-C++ implementations
20. VoiceCloner.cpp inspected and hardened: saveProfile now uses stack-based JUCE arrays plus a local DynamicObject helper instead of manual heap allocation/deletion
21. core/memory.cpp inspected and hardened: LockFreeQueue sentinel and pushed nodes now use local std::unique_ptr during allocation/ownership handoff before raw atomic linkage
22. TooltipComponent.cpp inspected and hardened: singleton creation now uses local std::unique_ptr before DeletedAtShutdown ownership handoff
23. ONNXInference.cpp/.h inspected; existing unique_ptr-based owner wrappers and reset ordering already look clean in this pass
24. ScoreEntryPanel.cpp/.h inspected; widespread reset(new) patterns are already unique_ptr-owned synchronous widget setup, no immediate lifetime fix applied in this pass
25. src_penta-core/common/RTLogger.cpp inspected and hardened: start/stop are now idempotent and protected against double-start/restart hazards, mirroring the main RTLogger rescue
26. src_penta-core/ml/MLInterface.cpp first pass inspected; worker lifecycle looked mostly paired
27. include/penta/rt/AudioWorkerThread.h inspected and hardened: inline start/stop now use compare_exchange/exchange guards plus joinable cleanup to prevent duplicate-start or stale-thread ownership hazards
28. src_penta-core/osc/RTMessageQueue.cpp inspected; unique_ptr-backed queue ownership already clean in this pass
29. src_penta-core/osc/OSCMessage.cpp inspected; value semantics only, no ownership fix needed in this pass
30. src_penta-core/diagnostics/PerformanceMonitor.cpp and AudioAnalyzer.cpp inspected; atomics/preallocated buffers already look ownership-safe in this pass
31. src/osc/OSCServer.cpp, OSCClient.cpp, OSCHub.cpp inspected as penta-backed support code; ownership is primarily unique_ptr-based, no memory-safety fix applied in this pass
32. src/common/RTMemoryPool.cpp and include/penta/common/RTMemoryPool.h inspected; placement-new pool contract remains intentional for RT allocation, no ownership rewrite applied in this pass
33. TempoEstimator.cpp, RhythmQuantizer.cpp, ScaleDetector.cpp, ChordAnalyzer.cpp inspected; no new ownership/lifetime hazards surfaced in this pass
34. src_penta-core/ml/MLInterface.cpp second pass hardened: start/stop now use compare_exchange/exchange guards and join any stale joinable worker before relaunch, aligning with the other rescued worker-thread modules
35. src/ml/AudioEmotionRunner.cpp second pass hardened: initialize/shutdown now use compare_exchange/exchange guards and join any stale joinable worker before relaunch, aligning the primary audio-side inference runner with the rescued worker-thread pattern
36. Intentional raw-boundary sweep: kelly_ffi.h/.cpp re-read; malloc/free contract is internally consistent and intentionally ABI-facing, so no semantic change applied
37. core/memory.cpp third pass hardened: sentinel destruction and pop tail handoff now route through local std::unique_ptr guards, preserving the raw queue shape while making final-node release more explicit
38. src/common/RTMemoryPool.cpp + include/penta/common/RTMemoryPool.h hardened: deallocate() now rejects out-of-pool or misaligned pointers via contains()/isBlockAligned() before pushing back onto the free list, preventing accidental free-list corruption without changing RT allocation semantics
39. src/bridge/StateBridge.cpp hardened for lifecycle idempotence: initialize() now short-circuits if already live, resets shutdownRequested_ on re-entry, worker teardown uses explicit atomic store, and shutdown() flushes once on the active shutdown path only
40. src/ml/MLBridge.cpp and src/bridge/OrchestratorBridge.cpp narrowed async-thread sweep: replaced ineffective !joinable()-based pruning with completion-aware AsyncWorker tracking so finished threads get joined and erased without blocking on still-running work at each spawn
41. src/ml/InferenceThreadManager.h hardened: start() now always normalizes prior thread state through stop(), stop() uses exchange(false) with joinable guarding, and relaunch semantics now match the other rescued worker-thread modules
42. src/biometric/BiometricInput.h/.cpp/.mm cleanup: removed dead unused streamingThread_/shouldStream_/streamingLoop declarations and ctor init from both C++ and Objective-C++ twins, eliminating stale thread-owner state that was no longer implemented
43. intent_ir_ffi.cpp residual FFI-contract pass hardened session-id coherence: validate_and_store(), clamp_and_store(), and intent_ir_new_session_id() now keep the atomic/session snapshot aligned, and get_current_session_id() reads from the frame snapshot under the mutex so plugin/UI inspection cannot drift from the live Intent IR frame
44. OSCBridge.cpp/.h residual bridge-logic pass hardened pending-request cleanup: introduced takePendingRequest(), erased pending entries before external error callbacks, and cleaned up silent leak paths when send() fails for chord/process/suggest/ping requests
45. Latest validation: KellyCore and KellyFFI rebuilds remain clean after the Intent IR/OSC boundary coherence pass
46. Next batch: continue residual subsystem-logic audit on other intentional boundary code only if a similarly concrete contract-drift or cleanup bug is grounded by file inspection

Inspection heuristics for next batches:
- raw pointer ownership hidden behind typedefs or factory methods
- manual lock/unlock or resource acquire/release without guard objects
- owning pointers passed by reference without clear lifetime contract
- malloc/free/new/delete hidden in helper methods or third-party wrapper usage
- Python C API refcount edges lacking RAII/GIL guards
- JUCE object creation APIs that require raw returns: isolate allocation in unique_ptr and release only at boundary
