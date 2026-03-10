# AU Plugin Build Reference

> Exact CMake changes, build steps, code-signing, validation, and install
> procedures for shipping a KmiDi Audio Unit on macOS.

## 1. CMake Configuration Change

### Current state (VST3 + CLAP only)

```cmake
# CMakeLists.txt line ~352
juce_add_plugin(KellyPlugin
    COMPANY_NAME "Kelly"
    PLUGIN_MANUFACTURER_CODE Klly
    PLUGIN_CODE Klp1
    FORMATS VST3 CLAP              # ← AU missing
    PRODUCT_NAME "Kelly Emotion Processor"
    VST3_CATEGORIES Fx Synth
    CLAP_ID com.kelly.emotion-processor
)
```

### Required change

```cmake
juce_add_plugin(KellyPlugin
    COMPANY_NAME "KmiDi"
    PLUGIN_MANUFACTURER_CODE Klly
    PLUGIN_CODE Klp1
    FORMATS AU VST3 CLAP Standalone   # ← AU + Standalone added

    PRODUCT_NAME "KmiDi Emotion Processor"

    # AU-specific settings
    AU_MAIN_TYPE com.apple.audio.units.type.midi-processor   # 'aumi'
    AU_EXPORT_PREFIX KmiDiAU
    AU_SANDBOX_SAFE TRUE

    # VST3 settings (unchanged)
    VST3_CATEGORIES Fx Synth
    CLAP_ID com.kmidi.emotion-processor

    # Bundle identifiers
    BUNDLE_ID com.kmidi.emotion-processor
    MICROPHONE_PERMISSION_ENABLED FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT TRUE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE

    ${KellyPlugin_ICON_ARGS}
)
```

### Key properties explained

| Property | Value | Why |
|---|---|---|
| `FORMATS AU VST3 CLAP Standalone` | All four formats | Full DAW coverage |
| `AU_MAIN_TYPE` | `com.apple.audio.units.type.midi-processor` | Maps to `'aumi'` — MIDI generation without audio I/O |
| `AU_EXPORT_PREFIX` | `KmiDiAU` | C function prefix for AU entry points (avoids symbol collisions) |
| `AU_SANDBOX_SAFE` | `TRUE` | Declares plugin safe for sandboxed hosts |
| `IS_MIDI_EFFECT` | `TRUE` | JUCE uses this to configure bus layouts and AU type |
| `BUNDLE_ID` | `com.kmidi.emotion-processor` | macOS bundle identifier for codesigning |

### Additional compile definitions

```cmake
target_compile_definitions(KellyPlugin PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
    JUCE_AU_WRAPPERS_SAVE_PROGRAM_STATES=1    # Persist AU presets
)

# AU-specific target (JUCE creates KellyPlugin_AU)
if(TARGET KellyPlugin_AU)
    target_compile_definitions(KellyPlugin_AU PRIVATE
        JUCE_AU_WRAPPERS_SAVE_PROGRAM_STATES=1
    )
endif()
```

## 2. Build Commands

### Configure

```bash
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PLUGINS=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

| Flag | Purpose |
|---|---|
| `CMAKE_OSX_DEPLOYMENT_TARGET=12.0` | Minimum macOS version (Logic Pro 10.8+ requires 12.0+) |
| `CMAKE_OSX_ARCHITECTURES="arm64;x86_64"` | Universal Binary for Intel + Apple Silicon |

### Build

```bash
cmake --build build --target KellyPlugin_AU --config Release
```

Or build all formats:
```bash
cmake --build build --target KellyPlugin_All --config Release
```

### Output locations

```
build/KellyPlugin_artefacts/Release/
├── AU/
│   └── KmiDi Emotion Processor.component    ← AU plugin bundle
├── VST3/
│   └── KmiDi Emotion Processor.vst3
├── CLAP/
│   └── KmiDi Emotion Processor.clap
└── Standalone/
    └── KmiDi Emotion Processor.app
```

## 3. Install

### Development (local testing)

```bash
# Copy to user plugin directory
cp -R "build/KellyPlugin_artefacts/Release/AU/KmiDi Emotion Processor.component" \
    ~/Library/Audio/Plug-Ins/Components/

# Force the system to re-scan
killall -9 AudioComponentRegistrar 2>/dev/null || true
```

### Production (installer)

```bash
# System-wide install (requires sudo)
sudo cp -R "KmiDi Emotion Processor.component" \
    /Library/Audio/Plug-Ins/Components/
```

## 4. Validation with auval

### List all installed AU plugins

```bash
auval -a
```

### Validate the KmiDi plugin

```bash
# Syntax: auval -v TYPE SUBTYPE MANUFACTURER
auval -v aumi Klp1 Klly
```

Expected output for a passing plugin:
```
  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         AU Validation Tool
         Version: 1.7.0
         Copyright 2003-2024, Apple Inc. All Rights Reserved.
  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  Validating AU: KmiDi - KmiDi Emotion Processor (aumi Klp1 Klly)
  ...
  AU VALIDATION SUCCEEDED.
```

### Common auval failures and fixes

| Failure | Cause | Fix |
|---|---|---|
| "can't find component" | Plugin not installed or registrar stale | `killall -9 AudioComponentRegistrar` and reinstall |
| "failed to open" | Code signing issue | Sign with valid identity (see §5) |
| "failed render test" | `processBlock` crashes or produces NaN | Debug with JUCE `AudioPluginHost` first |
| "incorrect tail time" | `getTailLengthSeconds()` wrong | Return accurate value |
| "properties not implemented" | Missing required AU properties | JUCE handles most; check custom property overrides |

## 5. Code Signing & Notarization

### Development signing (local only)

```bash
# Sign with ad-hoc identity for local testing
codesign --force --deep --sign - \
    "KmiDi Emotion Processor.component"
```

### Production signing (distribution)

```bash
# 1. Sign with Developer ID
codesign --force --deep --options runtime \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    "KmiDi Emotion Processor.component"

# 2. Create zip for notarization
ditto -c -k --keepParent \
    "KmiDi Emotion Processor.component" \
    "KmiDi_AU.zip"

# 3. Submit for notarization
xcrun notarytool submit "KmiDi_AU.zip" \
    --apple-id "you@example.com" \
    --team-id "TEAMID" \
    --password "app-specific-password" \
    --wait

# 4. Staple the notarization ticket
xcrun stapler staple "KmiDi Emotion Processor.component"
```

### Requirements

- Apple Developer Program membership ($99/year)
- Hardened Runtime enabled (`--options runtime`)
- No unsigned frameworks or dylibs inside the bundle
- Notarization passes Apple's malware scan

## 6. Testing in DAW Hosts

### Logic Pro

1. Install AU to `~/Library/Audio/Plug-Ins/Components/`
2. Open Logic Pro → Preferences → Plug-In Manager
3. Click "Reset & Rescan" to discover new plugins
4. Create a Software Instrument track
5. Insert KmiDi on a MIDI track (appears under MIDI FX if `'aumi'`)
6. Verify: parameter automation, state save/load, MIDI output routing

### GarageBand

1. Same install as Logic Pro
2. Create a track → Smart Controls → Plug-ins
3. Verify basic functionality

### Reaper

1. Reaper auto-scans AU paths on startup
2. Preferences → Plug-ins → AU → "Rescan" if needed
3. Insert on MIDI track

### JUCE AudioPluginHost (debugging)

```bash
# Build JUCE's plugin host for debugging
cd KmiDi/external/JUCE/extras/AudioPluginHost
cmake -B build -G Ninja
cmake --build build
./build/AudioPluginHost_artefacts/AudioPluginHost
```

This is the best tool for debugging AU issues before testing in a real DAW.

## 7. CI/CD Considerations

### GitHub Actions (macOS runner)

```yaml
- name: Build AU Plugin
  run: |
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PLUGINS=ON \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0
    cmake --build build --target KellyPlugin_AU

- name: Validate AU
  run: |
    cp -R "build/KellyPlugin_artefacts/Release/AU/KmiDi Emotion Processor.component" \
      ~/Library/Audio/Plug-Ins/Components/
    killall -9 AudioComponentRegistrar || true
    sleep 2
    auval -v aumi Klp1 Klly
```

## 8. Troubleshooting Checklist

- [ ] JUCE submodule at `KmiDi/external/JUCE` resolves to JUCE 8 tag
- [ ] `CMakeLists.txt` includes `AU` in FORMATS list
- [ ] `CMAKE_OSX_DEPLOYMENT_TARGET` set (12.0+ recommended)
- [ ] Universal binary: `CMAKE_OSX_ARCHITECTURES="arm64;x86_64"`
- [ ] Plugin installed to `~/Library/Audio/Plug-Ins/Components/`
- [ ] `AudioComponentRegistrar` restarted after install
- [ ] `auval -v aumi Klp1 Klly` passes
- [ ] Code signed (at minimum ad-hoc for local dev)
- [ ] No `NaN` or `inf` from `processBlock()` (causes auval render failure)
- [ ] `getStateInformation()` / `setStateInformation()` round-trip correctly
