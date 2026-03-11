# Audio Unit (AU) Plugin Architecture

Architecture for KmiDi/iDAW Audio Unit plugins on macOS and iOS (AUv3).

## Overview

| Platform | Format | Build Option | Notes |
|----------|--------|--------------|-------|
| macOS | AU (AUv2) | `DAIW_BUILD_AU` (KmiDi_FINAL) | Legacy DAIW engine |
| macOS | AU (JUCE) | `BUILD_PLUGINS` + `KMIDI_BUILD_JUCE_UI` | Root Kelly plugin |
| iOS | AUv3 | Generated via `music_brain.mobile.ios_audio_unit` | Swift/Obj-C scaffolding |

## Build Contexts

**Root project** (`/KmiDi`):
- `BUILD_PLUGINS`, `KMIDI_BUILD_JUCE_UI` — VST3/CLAP and optionally AU via JUCE.
- Produces `KellyPlugin` targets.

**Legacy DAIW** (`KmiDi_FINAL/engine/cpp_music_brain`):
- `DAIW_BUILD_VST3`, `DAIW_BUILD_AU` — separate plugin pipeline.
- Do not mix option names across these two build roots.

## iOS AUv3 (Generated Scaffolding)

`music_brain.mobile.ios_audio_unit` generates Swift/Obj-C scaffolding for AUv3:

- **Config:** `iOSAudioUnitConfig` — bundle ID, AU type (`aumu` instrument / `aufx` effect), subtype, manufacturer, sample rates, channel configs, MIDI, UI dimensions.
- **Outputs:** Info.plist, `*AudioUnit.swift`, `*DSPKernel.hpp`/`.mm`, `Parameters.swift`, `AudioUnitViewController.swift`.
- **Factory:** `*AudioUnitFactory` for host discovery.

### AUv3 Component Structure

```
AUv3 Extension
├── Info.plist          # NSExtension, AudioComponents
├── *AudioUnit.swift    # AUAudioUnit subclass, factory
├── *DSPKernel.hpp/mm   # C++ DSP (optional bridge to Kelly)
├── Parameters.swift    # Parameter definitions
└── AudioUnitViewController.swift  # UI (SwiftUI or UIKit)
```

### Requirements (from `get_ios_au_requirements`)

- Xcode, iOS SDK
- Swift 5+
- Audio Toolbox, AVFoundation
- Sandbox-safe design for App Store distribution

## macOS AU (JUCE)

When `KMIDI_BUILD_JUCE_UI=ON` and `BUILD_PLUGINS=ON`, the root CMake builds plugin formats including AU via JUCE's `juce_add_plugin`:

- **Processor:** Kelly emotion/DSP logic in C++.
- **Editor:** JUCE `AudioProcessorEditor` or custom UI.
- **Formats:** VST3, CLAP, AU (macOS).

## macOS AU (Legacy DAIW)

In `KmiDi_FINAL/engine/cpp_music_brain`:

- `DAIW_BUILD_AU=ON` builds the legacy DAIW AU plugin.
- Separate codebase from root Kelly plugin.

## Integration Points

| Layer | Responsibility |
|-------|----------------|
| **Host** | Loads AU, provides audio/MIDI I/O, project context |
| **AU wrapper** | Parameter persistence, preset handling, latency reporting |
| **DSP kernel** | Real-time audio processing (Kelly engine or custom) |
| **UI** | Parameter controls, visualization (SwiftUI/UIKit or JUCE) |

## Latency and Quality

- **Latency:** Report via `latencyInSeconds`; keep processing blocks small for low latency.
- **Quality:** DSP runs at host sample rate; avoid sample-rate conversion in the critical path.
- **Thread safety:** AU callbacks run on real-time threads; no allocations, locks, or system calls.

## References

- `music_brain/mobile/ios_audio_unit.py` — AUv3 scaffolding generator
- `music_brain/mobile/platforms.py` — Platform capabilities, AU requirements
- `docs/PROJECT_FEATURES_DEBATE.md` — Plugin suite (VST3, AU, CLAP)
- `docs/FULL_STACK_BUILD.md` — Root build and plugin targets
