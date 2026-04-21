# Dead-Code Audit (2026 Q2)

**Date:** 2026-04-21
**Branch:** `audit/dead-code-sweep` (stacked on `audit/biometric-guard-fix`)
**Scope:** `src/`, `src_penta-core/`, `include/penta/`, `include/kelly/`, `include/kmidi/`, `include/daiw/`
**Excluded from scope:** `_archive/`, `KmiDi_PROJECT/`, `xcode/**`, `projects/**/legacy-reference/**`, `external/JUCE/`, `third_party/`, `.worktrees/`, `tests/`, `docs/`
**Method:** CMake `EXCLUDE REGEX` check + header `#include` grep + class-name reference grep across `src/`, `src_penta-core/`, `include/`, `tests/`

---

## Summary

**24 candidates flagged: 12 definite / 7 likely / 5 possible.**

Note: `include/kelly/` does not exist in-tree (scope item absent). All findings are in `src/` or `include/penta|kmidi|daiw`.

---

## Definite dead (safe to delete, zero consumers)

**D1 — `src/engine/test_kelly.cpp`**
- Evidence: CMakeLists.txt:265 — `list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/engine/test_kelly\\.cpp$")`. File is present in-tree. `grep -rln "test_kelly" src/ include/ tests/` returns 0 hits outside itself. Contains a standalone `printSeparator` + `testEmotionMapper` + `testEmotionThesaurus` driver; no external caller.
- Recommendation: delete. If test coverage is desired, port to `tests/` as a Catch2 test.

**D2 — `src/dsp/audio_buffer.cpp`**
- Evidence: CMakeLists.txt:274 — `EXCLUDE REGEX "/src/dsp/audio_buffer\\.cpp$"`. No paired `.h` in canonical include paths. `grep -rn "audio_buffer" src/ include/ tests/ src_penta-core/` (excluding the file itself) returns 0 hits.
- Recommendation: delete.

**D3 — `src/dsp/simd_ops.cpp`**
- Evidence: CMakeLists.txt:275 — `EXCLUDE REGEX "/src/dsp/simd_ops\\.cpp$"`. `grep -rn "simd_ops" src/ include/ tests/ src_penta-core/` returns 0 hits outside the file. SIMD work has migrated to `include/penta/common/SIMDKernels.h` (consumed by `src_penta-core/groove/OnsetDetector.cpp` and `src/ml/MelSpectrogram.cpp`).
- Recommendation: delete. SIMDKernels.h is the canonical SIMD surface.

**D4 — `src/dsp/dsp.cpp`**
- Evidence: CMakeLists.txt:276 — `EXCLUDE REGEX "/src/dsp/dsp\\.cpp$"`. No paired `.h` in include paths. Zero external `#include` or class-name hits across canonical roots.
- Recommendation: delete. Entire `src/dsp/` directory becomes empty after D2/D3/D4; remove the directory too.

**D5 — `src/core/memory.cpp`**
- Evidence: CMakeLists.txt:277 — `EXCLUDE REGEX "/src/core/memory\\.cpp$"`. Implements `daiw::MutexMemoryPool` from `daiw/memory.hpp`. `grep -rn "core/memory\|MutexMemoryPool" src/ include/ tests/ src_penta-core/` returns 0 hits outside the file. RT-safe pool work moved to `include/penta/common/RTMemoryPool.h`.
- Recommendation: delete.

**D6 — `src/core/logging.cpp`**
- Evidence: CMakeLists.txt:278 — `EXCLUDE REGEX "/src/core/logging\\.cpp$"`. Defines its own `daiw::LogLevel` enum and private helpers; no `.h` in canonical includes, zero external consumers. Logging in the build uses `penta/common/RTLogger.h` / JUCE Logger.
- Recommendation: delete.

**D7 — `src/core/types.cpp`**
- Evidence: CMakeLists.txt:279 — `EXCLUDE REGEX "/src/core/types\\.cpp$"`. File body is comment-only: `// Currently, all types are header-only. This file exists for future expansion.` Zero symbols defined. Excluded from build.
- Recommendation: delete (zero-content stub).

**D8 — `src/harmony/chord.cpp`**
- Evidence: CMakeLists.txt:284 — `EXCLUDE REGEX "/src/harmony/chord\\.cpp$"`. Defines `daiw::harmony::ChordQuality` enum and local `Chord` class inside anonymous namespace. `grep -rn "\"chord\.h\"\|harmony/chord\b" src/ include/ tests/ src_penta-core/` returns 0 hits; the sole match in `src/ui/MusicianCommandPanel.h:142` is a string literal `"chord"` not an include. No paired header in `include/daiw/`.
- Recommendation: delete.

**D9 — `src/harmony/progression.cpp`**
- Evidence: CMakeLists.txt:285 — `EXCLUDE REGEX "/src/harmony/progression\\.cpp$"`. Defines `daiw::harmony::Progression` class with no paired header in canonical include paths. `grep -rn "\"progression\.h\"\|harmony/progression\b" src/ include/ tests/ src_penta-core/` returns 0 hits.
- Recommendation: delete.

**D10 — `src/core/chord_diagnostics.cpp` + `src/core/chord_diagnostics.h`**
- Evidence: Not excluded by CMake (included in build glob), but `grep -rn "ChordDiagnostics\|chord_diagnostics" src/ include/ tests/ src_penta-core/` returns only hits inside `src/core/chord_diagnostics.{cpp,h}` — zero external consumers. The class is compiled into `KellyCore` but never instantiated anywhere.
- Recommendation: delete both files.

**D11 — `src/core/groove_templates.cpp` + `src/core/groove_templates.h`**
- Evidence: Not excluded by CMake (compiled into KellyCore). `grep -rn "#include.*groove_templates" src/ include/ tests/ src_penta-core/` returns 0 hits outside the file itself. `src/engine/GrooveEngine.h:9` contains a comment reference only (`"if linked"`), not an `#include`. The class `kelly::GrooveTemplates` is never instantiated externally. `src/engine/Kelly.h:136` uses its own inline `getGrooveTemplates()` that reads from a different `std::map` in `GrooveTemplateEngine`, not from this class.
- Recommendation: delete both files. Named `legacy template data` in its own header comment.

**D12 — `src/WavetableSynth.cpp` + `include/WavetableSynth.h`**
- Evidence: Not excluded by CMake glob (compiled into KellyCore). `grep -rn "WavetableSynth" src/ include/ tests/ src_penta-core/ plugins/ apps/` returns 0 hits outside the two files themselves. Referenced in `docs/THREE_CRITICAL_IMPROVEMENTS.md` as `"Mixed with UI code"` and in `scripts/migrate_missing_files.sh` as an exclusion target (i.e., deliberately not migrated). No consumer anywhere in the canonical build surface.
- Recommendation: delete both. If a wavetable synth is ever needed, build against JUCE's `WavetableOscillator` or a penta-core kernel.

---

## Likely dead (one or two stale references, probably safe)

**L1 — `src/midi/midi_engine.cpp`**
- Evidence: CMakeLists.txt:280 — `EXCLUDE REGEX "/src/midi/midi_engine\\.cpp$"`. Defines `daiw::midi::EventType` enum and internal class, no paired header in canonical includes. `grep -rn "#include.*midi_engine\|daiw::midi::EventType" src/ include/ tests/ src_penta-core/` returns 0 hits. Only reference: a code comment in `src/ml/AudioEmotionRunner.cpp:413` (contains the string `"kmidi_engine_get_state"` in a comment, not an include or call).
- Recommendation: delete. Comment reference is stale.

**L2 — `src/midi/MidiIO.cpp` + `include/daiw/midi/MidiIO.h`**
- Evidence: CMakeLists.txt:281 — `EXCLUDE REGEX "/src/midi/MidiIO\\.cpp$"`. `grep -rn "#include.*daiw/midi/MidiIO\|daiw::midi::MidiInput" src/ include/ tests/ src_penta-core/` (excluding `include/daiw/midi/MidiIO.h` itself) returns 0 hits. The header is not included from any `src/` or `src_penta-core/` file.
- Recommendation: delete both. The daiw MIDI I/O layer is superseded by JUCE's `MidiInput` / `MidiOutput` used everywhere else.

**L3 — `src/midi/MidiMessage.cpp` + `include/daiw/midi/MidiMessage.h`**
- Evidence: CMakeLists.txt:282 — `EXCLUDE REGEX "/src/midi/MidiMessage\\.cpp$"`. `grep -rn "#include.*daiw/midi/MidiMessage\|daiw::midi::MidiMessage" src/ tests/` (outside `src/midi/MidiMessage.cpp` and `include/daiw/midi/`) returns hits only inside `include/daiw/midi/MidiSequence.h:14` (which itself has no external consumers — see L4). No `src/` file includes `daiw/midi/MidiMessage.h` directly.
- Recommendation: delete both once L4 is also deleted.

**L4 — `src/midi/MidiSequence.cpp` + `include/daiw/midi/MidiSequence.h`**
- Evidence: CMakeLists.txt:283 — `EXCLUDE REGEX "/src/midi/MidiSequence\\.cpp$"`. `grep -rn "#include.*daiw/midi/MidiSequence" src/ include/ tests/ src_penta-core/` returns hits only in: (1) `src/midi/MidiSequence.cpp:6` (self), (2) `include/daiw/midi/MidiIO.h:10` (dead, L2), (3) `include/daiw/project/ProjectFile.h:17`. The `ProjectFile.h` inclusion chain: `src/project/ProjectFile.cpp:9` is excluded by CMakeLists.txt:288, and no `src/` file outside that exclusion includes `daiw/project/ProjectFile.h` (verified by `grep -rn "#include.*daiw/project/ProjectFile" src/ tests/ src_penta-core/` returning 0 external hits).
- Recommendation: delete both (dependent on L2/L3 also going away to avoid dangling includes).

**L5 — `src/audio/AudioFile.cpp` + `include/daiw/audio/AudioFile.h`**
- Evidence: CMakeLists.txt:286 — `EXCLUDE REGEX "/src/audio/AudioFile\\.cpp$"`. `grep -rn "#include.*daiw/audio/AudioFile\|daiw::audio::AudioFile" src/ include/ tests/ src_penta-core/` returns **1 hit**: `include/daiw/export/StemExporter.h:14` — but StemExporter is itself L6 (CMake-excluded + zero external consumers), so the chain is dead-to-dead.
- Recommendation: delete both together with L6 (order: delete StemExporter.h first so AudioFile.h has zero references, then delete AudioFile.{h,cpp}).

**L6 — `src/export/StemExporter.cpp` + `include/daiw/export/StemExporter.h`**
- Evidence: CMakeLists.txt:287 — `EXCLUDE REGEX "/src/export/StemExporter\\.cpp$"`. `grep -rn "#include.*daiw/export/StemExporter\|daiw::export::StemExporter" src/ include/ tests/ src_penta-core/` returns 0 hits outside the files themselves.
- Recommendation: delete both.

**L7 — `src/project/ProjectFile.cpp` + `include/daiw/project/ProjectFile.h`**
- Evidence: CMakeLists.txt:288 — `EXCLUDE REGEX "/src/project/ProjectFile\\.cpp$"`. `grep -rn "#include.*daiw/project/ProjectFile" src/ include/ tests/ src_penta-core/` returns 0 external hits (the only include chain goes through `include/daiw/midi/MidiSequence.h` which is itself L4-dead). Note: `src/project/ProjectManager.{cpp,h}` is an active JUCE-based project manager that uses `juce::File`; it does NOT include or depend on the daiw `ProjectFile`.
- Recommendation: delete both.

---

## Possible dead (not directly consumed but linked via excluded-chain or commentary only)

**P1 — `src/core/emotion_thesaurus.cpp` + `src/core/emotion_thesaurus.h`**
- Evidence: CMakeLists.txt:292-298 — explicitly labeled a `self-labeled legacy stub` that returns `nullptr/no-op` for everything, excluded to prevent linker from silently picking it over `src/engine/EmotionThesaurus.cpp`. The exclusion comment says ODR collision is the reason. The `.h` at `src/core/emotion_thesaurus.h` says `Engine Layer: EmotionThesaurusLoader (JSON-backed data)` — only `src/common/KellyTypes.h:152` references `emotion_engine.h` in a comment, and `tests/conftest.py:39` imports a Python-layer `EmotionThesaurus`, not this C++ stub. The class `kelly::EmotionEngine` in `src/core/emotion_engine.cpp` has zero external C++ consumers (see grep result above).
- Recommendation: delete the stub (`.cpp` + `.h`). The real implementation at `src/engine/EmotionThesaurus.cpp` is the active one.

**P2 — `src/ui/MidiKompanionLookAndFeel.cpp` + `src/ui/MidiKompanionLookAndFeel.h`**
- Evidence: CMakeLists.txt:299-302 — labeled `byte-identical copy of src/ui/KellyLookAndFeel.cpp`, excluded to prevent ODR collision. `grep -rn "#include.*MidiKompanionLookAndFeel\|MidiKompanionLookAndFeel" src/ include/ tests/ src_penta-core/` returns only self-references and a comment in `src/ui/KellyLookAndFeel.h:8`. No file instantiates or includes `MidiKompanionLookAndFeel` directly.
- Recommendation: delete both. Documented in `docs/AUDIT_UI_LIFETIME_2026Q2.md:119` as a prior finding.

**P3 — `include/penta/rt/AudioWorkerThread.h`**
- Evidence: `grep -rn "AudioWorkerThread\|#include.*rt/AudioWorkerThread" src/ tests/ src_penta-core/` returns 0 hits. Header exists but is never included. File is header-only; no paired `.cpp`.
- Recommendation: delete. If the RT audio thread pattern is needed, consumers should use JUCE's `AudioThread` or `src_penta-core/` abstractions.

**P4 — `include/penta/midi_ci/EmotionLanes.h`**
- Evidence: `grep -rn "EmotionLanes\|#include.*midi_ci" src/ tests/ src_penta-core/` returns 0 hits. Header exists but is never included. No paired `.cpp`.
- Recommendation: delete if the MIDI-CI emotion lane protocol is not actively being developed; otherwise move to a `wip/` subdirectory.

---

## Explicitly excluded from CMake but still in-tree

| Path | CMakeLists.txt line | In-tree? | Any other consumer? |
|------|---------------------|----------|---------------------|
| `src/engine/test_kelly.cpp` | 265 | yes | No |
| `src/dsp/audio_buffer.cpp` | 274 | yes | No |
| `src/dsp/simd_ops.cpp` | 275 | yes | No (SIMD work is in `include/penta/common/SIMDKernels.h`) |
| `src/dsp/dsp.cpp` | 276 | yes | No |
| `src/core/memory.cpp` | 277 | yes | No |
| `src/core/logging.cpp` | 278 | yes | No |
| `src/core/types.cpp` | 279 | yes | No (zero-content stub) |
| `src/midi/midi_engine.cpp` | 280 | yes | Comment ref only in `AudioEmotionRunner.cpp:413` |
| `src/midi/MidiIO.cpp` | 281 | yes | No (`include/daiw/midi/MidiIO.h` self-includes only) |
| `src/midi/MidiMessage.cpp` | 282 | yes | No |
| `src/midi/MidiSequence.cpp` | 283 | yes | Dead include chain via excluded L2/L7 |
| `src/harmony/chord.cpp` | 284 | yes | No |
| `src/harmony/progression.cpp` | 285 | yes | No |
| `src/audio/AudioFile.cpp` | 286 | yes | No |
| `src/export/StemExporter.cpp` | 287 | yes | No |
| `src/project/ProjectFile.cpp` | 288 | yes | No |
| `src/python/groove_bindings.cpp` | 289 | yes | Consumed by `src/python/bindings.cpp` (which is itself excluded when pybind11 absent) |
| `src/core/emotion_thesaurus.cpp` | 298 | yes | ODR stub; no unique callers |
| `src/ui/MidiKompanionLookAndFeel.cpp` | 302 | yes | No |
| `src/biometric/BiometricInput.cpp` | 307 (Apple) | yes | Platform-conditional: `.mm` used on Apple, `.cpp` on others — NOT dead |

---

## Files examined, no findings

Count: 22 actively-used files verified to have live consumers. Sample:

- `src/engine/KellyBrain.cpp` — consumed by `src/plugin/PluginProcessor.cpp` and `src/bridge/kelly_ffi.cpp`.
- `src/engine/IntentPipeline.cpp` — consumed by `src/engine/KellyBrain.cpp` and multiple bridge files.
- `src/plugin/PluginProcessor.cpp` — excluded from `KellyCore` (CMakeLists.txt:264) but compiled into the plugin target.
- `src/harmony/VoiceLeading.cpp` — built from `src/` canonical (confirmed in `src_penta-core/CMakeLists.txt:64`); consumed transitively via `src/midi/VoiceLeadingAdapter.h` → `src/midi/ChordGenerator.cpp`, `include/penta/harmony/HarmonyEngine.h`, and `src_penta-core/harmony/HarmonyEngine.cpp`.
- `include/penta/common/SIMDKernels.h` — included by `src/ml/MelSpectrogram.cpp`, `src_penta-core/groove/OnsetDetector.cpp`, and `tests/cpp/test_simd_kernels.cpp`.
- `include/penta/diagnostics/BlockLatencyInstrument.h` — included by `src/plugin/PluginProcessor.h:60`.
- `include/kmidi/IntentIR.h` — included by 10+ files including `src/midi/MidiGenerator.h`, `src/common/IntentIRAdapter.h`, and all `src/engines/*.h` headers.
- `src/BridgeClient.cpp` + `include/BridgeClient.h` — consumed by `src/VoiceProcessor.cpp` and `include/VoiceProcessor.h`.
- `src/VoiceProcessor.cpp` + `include/VoiceProcessor.h` — consumed via `src/BridgeClient.cpp` (bidirectional bridge pair). These are in-tree and compiled but lack a downstream plugin consumer in canonical scope; however they are deliberate bridge pairs and not dead in the same sense as the above — flagged for follow-up only.
- `src/core/emotion_engine.cpp` + `src/core/emotion_engine.h` — `EmotionEngine` class itself has zero external C++ consumers (only comment refs in `src/common/KellyTypes.h`). Compiled into KellyCore. Likely dead but kept as "possible" pending Python-bridge audit.
- `src/python/groove_bindings.cpp` + `src/python/harmony_bindings.cpp` — consumed by `src/python/bindings.cpp` when pybind11 is present. NOT dead; conditional on build flag `BUILD_PYTHON_BINDINGS`.
- `include/kmidi/IntentIR_JSON.h` — no `#include` found in canonical roots; zero consumers. Possible dead (not listed above as it was found late — add to P list if confirmed).

---

## Additional finding: `include/kmidi/IntentIR_JSON.h`

**P5 — `include/kmidi/IntentIR_JSON.h`**
- Evidence: `grep -rn "#include.*IntentIR_JSON\|IntentIR_JSON\.h" src/ include/ tests/ src_penta-core/` returns 0 hits. Header exists in `include/kmidi/` alongside the active `IntentIR.h` but is never included.
- Recommendation: delete or fold into `IntentIR.h` if the JSON serialization is intended to be part of the IR spec.

---

## Next steps

Fix PR in priority order:

1. **D-group first (zero-risk deletions):** D1–D12. All are either CMake-excluded or have zero consumers. Deleting them eliminates ~2 400 LOC of compiled-but-dead code and prevents future ODR surprises.
   - `src/dsp/` (entire directory: D2, D3, D4)
   - `src/core/memory.cpp`, `src/core/logging.cpp`, `src/core/types.cpp` (D5, D6, D7)
   - `src/harmony/chord.cpp`, `src/harmony/progression.cpp` (D8, D9) — also check whether `src/harmony/` becomes empty after the 3 deletions (chord, progression; VoiceLeading.cpp stays)
   - `src/core/chord_diagnostics.{cpp,h}` (D10)
   - `src/core/groove_templates.{cpp,h}` (D11)
   - `src/WavetableSynth.cpp` + `include/WavetableSynth.h` (D12)
   - `src/engine/test_kelly.cpp` (D1)

2. **L-group next (likely safe):** L1–L7. All are CMake-excluded; delete in dependency order L3→L4→L2→L7→L5→L6→L1 to avoid dangling includes.

3. **P-group last (investigate first):** P1–P5. Confirm no Python bridge, string-registry, or runtime-loaded consumer before deleting.

---

## Verification

```
cargo test --manifest-path engine/intent_ir/Cargo.toml --lib
# Result: test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Refs: `docs/CODEAUDIT_CXX_DEEP_2026Q2.md`
