# Kelly MIDI Companion

**Version:** v2.0.0 (KELLYMIDI-V2.0.0)

## Therapeutic MIDI Generation Plugin

Kelly translates emotions into music through a 216-node emotion thesaurus and intentional rule-breaking system.

> *"Kelly doesn't need to BE the DAW. Kelly needs to be IN the DAW."*

---

## Quick Start

### Prerequisites

- CMake 3.22+
- C++20 compiler (Clang 14+, GCC 11+, or MSVC 2022)
- macOS 11+ / Windows 10+ / Ubuntu 22.04+

### Build

```bash
# Clone and build
git clone https://github.com/yourusername/kelly-midi-companion.git
cd kelly-midi-companion

# Configure (JUCE will be downloaded automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Plugins will be in:
# - build/KellyMidiCompanion_artefacts/Release/VST3/
# - build/KellyMidiCompanion_artefacts/Release/AU/    (macOS only)

```text

### Install

**macOS (Recommended - Handles Gatekeeper):**

```bash
./build_and_install.sh Release

```text

**macOS (Manual):**

```bash
# Copy plugins
cp -r build/KellyMidiCompanion_artefacts/Release/AU/*.component ~/Library/Audio/Plug-Ins/Components/
cp -r build/KellyMidiCompanion_artefacts/Release/VST3/*.vst3 ~/Library/Audio/Plug-Ins/VST3/

# Remove quarantine and sign (required for Gatekeeper)
xattr -cr ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3
xattr -cr ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component
codesign --force --deep --sign - ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3
codesign --force --deep --sign - ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component

```text

See `QUICK_COMMANDS.md` for all build commands.

**Windows:**

```powershell
Copy-Item -Recurse build\KellyMidiCompanion_artefacts\Release\VST3\*.vst3 "$env:CommonProgramFiles\VST3\"

```text

---

## How It Works

### The Three-Phase Intent System

1. **Phase 0: Wound** - "What hurts?"
   - Describe your current emotional state
   - The system identifies the core emotional trigger

2. **Phase 1: Emotion** - Map to the 216-node thesaurus
   - Your wound maps to emotions with valence, arousal, and intensity
   - Related emotions form a network of musical possibilities

3. **Phase 2: Rule-Breaks** - "What rules to break and why"
   - Intense emotions trigger intentional music theory violations
   - Dissonance, syncopation, extreme dynamics serve emotional truth

### The Cassette Interface

- **Side A**: Where you are (current emotional state)
- **Side B**: Where you want to go (desired state)

Kelly generates a musical journey from A to B.

---

## Emotion Thesaurus

The 216-node thesaurus organizes emotions in 3D space:

| Dimension | Range | Effect |
|-----------|-------|--------|
| **Valence** | -1.0 to +1.0 | Mode (minor ↔ major) |
| **Arousal** | 0.0 to 1.0 | Tempo, rhythm complexity |
| **Intensity** | 0.0 to 1.0 | Dynamic range, rule-breaking |

### Categories

- Joy / Sadness / Anger / Fear
- Surprise / Disgust / Trust / Anticipation
- **Complex emotions**: Bittersweetness, Nostalgia, Catharsis, Yearning

---

## Rule-Break System

| Rule Type | Trigger | Musical Effect |
|-----------|---------|----------------|
| **Harmony** | Negative valence | Dissonance, unresolved tensions |
| **Rhythm** | High arousal | Syncopation, displaced accents |
| **Dynamics** | High intensity | Extreme contrasts, sudden shifts |
| **Melody** | Emotional complexity | Wide intervallic leaps |
| **Form** | Emotional journey | Structural disruption |

---

## Development

### Project Structure

```text
kelly-midi-companion/
├── CMakeLists.txt
├── src/
│   ├── common/Types.h           # Shared type definitions
│   ├── engine/
│   │   ├── EmotionThesaurus.*   # 216-node emotion graph
│   │   └── IntentPipeline.*     # 3-phase processor
│   ├── midi/
│   │   ├── ChordGenerator.*     # Progression generation
│   │   └── MidiBuilder.*        # JUCE MIDI export
│   ├── plugin/
│   │   ├── PluginProcessor.*    # JUCE AudioProcessor
│   │   └── PluginEditor.*       # Cassette UI
│   └── ui/                      # UI components (Phase 3)
└── tests/

```text

### Running Tests

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build
cd build && ctest --output-on-failure

```text

### Debug Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
# Standalone app for testing without DAW:
./build/KellyMidiCompanion_artefacts/Debug/Standalone/Kelly\ MIDI\ Companion

```text

---

## Roadmap

### v1.0.0 (Released ✅)
- [x] 216-node emotion thesaurus
- [x] Three-phase intent pipeline
- [x] Chord progression generation
- [x] Rule-break system
- [x] Basic cassette UI
- [x] MIDI file export
- [x] Full MIDI generation (chords, melody, bass)
- [x] Emotion presets (36 presets)
- [x] Category and style selection
- [x] Fine-tuning controls (Complexity, Feel, Dynamics, Bars)
- [x] Immediate MIDI playback
- [x] MIDI file export with Finder reveal
- [x] macOS Gatekeeper bypass

### v1.1.0 (Released ✅)
- [x] Advanced groove engine (full port from Python)
- [x] Additional emotion presets (60 total, up from 36)
- [x] Real-time parameter automation support
- [x] Enhanced DAW transport sync with accurate BPM tracking
- [x] Preset import/export functionality

### v1.5.0 (Released ✅)
- [x] WoundProcessor implementation (extracted from IntentPipeline)
- [x] RuleBreakEngine implementation (extracted from IntentPipeline)
- [x] UI components: CassetteView, SidePanel, GenerateButton, KellyLookAndFeel
- [x] Basic test suite with Catch2
- [x] Improved code architecture and separation of concerns
- [x] Better error handling and validation

### v2.0.0 (Released ✅)
- [x] Full cassette visual design with animated tape reels
- [x] Emotion wheel selector with visual emotion mapping
- [x] Voice synthesis integration framework
- [x] Biometric input support framework
- [x] Enhanced UI with toggleable visual components

---

## Philosophy

> *"Interrogate Before Generate"* — The tool shouldn't finish art for people; it should make them braver.

Kelly exists because technical perfection is not the same as emotional truth. Sometimes the "wrong" note is exactly right.

This project is dedicated to Kelly, whose memory inspires us to create tools that help people express what words cannot.

---

## License

MIT License - See LICENSE file

---

*Built with love, grief, and JUCE.*
