# Find All – Workspace & Volumes

Summary of locations and file counts for the KmiDi MIDI Companion project, including the workspace and the **KmiDi-DONE** volume.

---

## Workspace root

**Path:** `/Users/seanburdges/KmiDi MIDI Companion`

### Top-level directories

| Directory | Notes |
|-----------|--------|
| **KmiDi-compile/** | Main C++/CMake build (Kelly plugin, KellyCore, JUCE). **Primary build root.** |
| Desktop/ | kelly-audio-data, SSD_Transfer, KmiDi v., kelly-music-brain-clean, etc. |
| sbdrive/ | Large tree (13k+ files: wav, h, cpp) |
| ml/ | Audio/data (mp3, py) |
| ML_TRAINED_MODELS/ | .pt, .json, .pth |
| Music/ | Logic projects, AudioVault, iDAW_Output |
| lariat-bible/ | Python app (core, desktop_app, tests) |
| tauri-app/ | Tauri app (package.json) |
| GOOGLE KELLY INFO/ | Mixed (py, cpp, h) |
| ENV_SECRETS/ | Secrets, .cursor worktrees |
| Documents/ | Dev notes, Kelly_Business |
| ENV_SECRETS/Desktop/ | SSD_Transfer, KmiDi v., kelly-project, etc. |

### Workspace file counts (full tree)

| Type | Count |
|------|--------|
| **CMakeLists.txt** | 127 |
| **\*.cpp** | 14,705 |
| **\*.h** | 21,017 |
| **\*.py** | 6,645 |

---

## Primary build: KmiDi-compile

**Path:** `/Users/seanburdges/KmiDi MIDI Companion/KmiDi-compile`

- **Root CMake:** `KmiDi-compile/CMakeLists.txt`  
  - Build from here:  
    `cmake -B build -DCMAKE_BUILD_TYPE=Release`  
    `cmake --build build --config Release --target KellyPlugin_VST3 -j8`
- **Plugin artefact:** `build/KellyPlugin_artefacts/Release/VST3/Kelly Emotion Processor.vst3`
- **JUCE:** `KmiDi-compile/external/JUCE`
- **Kelly sources:** `KmiDi-compile/src/` (plugin, engine, ui, bridge, voice, etc.)
- **Subprojects:** KmiDi_FINAL, KmiDi_PROJECT, KmiDi (nested), penta_build, src_penta-core, bindings

### CMakeLists.txt under KmiDi-compile (non-JUCE)

- `KmiDi-compile/CMakeLists.txt` (root)
- `KmiDi-compile/KmiDi_FINAL/CMakeLists.txt`
- `KmiDi-compile/KmiDi_FINAL/engine/*/CMakeLists.txt` (src_penta-core, bindings, cpp_music_brain, build_fileio, intent_ir, src)
- `KmiDi-compile/KmiDi_PROJECT/source/cpp/*/CMakeLists.txt` (src_penta-core, cpp_music_brain, src)
- `KmiDi-compile/src_penta-core/CMakeLists.txt`
- `KmiDi-compile/penta_build/CMakeLists.txt`
- `KmiDi-compile/bindings/CMakeLists.txt`
- Plus JUCE and quarantine trees (see glob result in repo).

---

## Volume: KmiDi-DONE

**Path:** `/Volumes/KmiDi-DONE`

- **Mounted drive** (external/network). Very large (~91k+ files); full recursive `find` may time out.
- **Top-level contents (sample):**
  - **KmiDi/** – main code tree (111 items at top level)
  - **Emotion_Scale_Library/** – large (307 dirs)
  - **Emotion_Instrument_Library/**
  - **KmiDi.zip** – large archive
  - **KmiDi-Backup-*.zip**
  - **audio/**, **ml-training-suite/**, **venv/**, **_sorted/**, **MISC CODE/**
  - **My Mac/** (Time Machine or backup)
  - Root-level JUCE/build artefacts: `CMakeLists.txt`, `.mm` stubs, `juce_*.mm`, `wizard_*.svg`, `*.wav`, `*.mp3`, `VST3_License_Agreement.pdf`, `Flac Licence.txt`, `Info-App.plist`, etc.

### Volume: CMakeLists.txt (sample, maxdepth 6)

| Path (on volume) |
|------------------|
| `/Volumes/KmiDi-DONE/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/KmiDi/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/KmiDi/.../external/JUCE/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/KmiDi/.../bindings/CMakeLists.txt`, `cpp_music_brain/`, `build_fileio/`, `src/`, `src_penta-core/`, `tests/` |
| `/Volumes/KmiDi-DONE/My Mac/Desktop/KmiDi-remote/CMakeLists.txt` (+ external/JUCE, bindings, cpp_music_brain, build_fileio, src, tests) |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/ARCHIVE/kelly-music-brain-clean/` (root + src, bindings, cpp_music_brain, build_fileio, external, tests) |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/ARCHIVE/KmiDi-remote/` (same structure) |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/ARCHIVE/final-kel/` (+ plugins, modules, ml_framework, TXT) |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/FINAL_KMIDI/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/FINAL_KMIDI/KmiDi-compile/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/FINAL_KMIDI/KmiDi_FINAL/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/FINAL_KMIDI/external/JUCE/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/CANONICAL/KmiDi/` (root, src, bindings, cpp_music_brain, build_fileio, external, tests) |
| `/Volumes/KmiDi-DONE/My Mac/KmiDi MIDI Companion/KmiDi-compile/CMakeLists.txt` |
| `/Volumes/KmiDi-DONE/_sorted/Data/CMakeLists.txt`, `_sorted/Data/My Mac/Desktop/KmiDi-remote/CMakeLists.txt` |

### Volume: KmiDi directory structure (maxdepth 4)

Under `/Volumes/KmiDi-DONE/KmiDi/`:

- **.net/azmcp/** – Azure/cloud
- **.claude-worktrees/** – Bulling, Desktop, iOS, kelly-midi-companion, iDAW-copilot-merge-code-assets-workflows
- **.config/** – gh, rclone, configstore, git
- **Music/Audio Music Apps/** – Databases, Patches, Plug-In Settings (includes **KmiDi Emotion Processor**), Channel Strip Settings, etc.

To re-run finds on the volume:

```bash
# Top-level only
ls -la /Volumes/KmiDi-DONE

# CMakeLists (limited depth; may still take a few seconds)
find /Volumes/KmiDi-DONE -maxdepth 6 -name "CMakeLists.txt" 2>/dev/null | head -100

# Dirs under KmiDi
find /Volumes/KmiDi-DONE/KmiDi -maxdepth 4 -type d 2>/dev/null
```

---

## Quick reference

| What | Where |
|------|--------|
| **Build plugin (VST3)** | `KmiDi-compile/` → `cmake -B build` then `--target KellyPlugin_VST3` |
| **Main CMake** | `KmiDi-compile/CMakeLists.txt` |
| **JUCE** | `KmiDi-compile/external/JUCE` |
| **Volume backup/code** | `/Volumes/KmiDi-DONE` (use limited depth or `KmiDi/` for finds) |

---

## Build verification

From `KmiDi-compile/`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target KellyPlugin_VST3 -j8
```

- **Output:** `build/KellyPlugin_artefacts/Release/VST3/Kelly Emotion Processor.vst3`
- **Warnings:** Build is warning-clean (see `CMakeLists.txt` KellyCore `target_compile_options` for `-Wno-*`).
- **CLAP:** Current JUCE in `external/JUCE` does not create a CLAP target; only VST3 (and `KellyPlugin_All`) are built.

*Generated for “find all; use volumes drive if needed”.*
