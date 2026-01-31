# Kelly Quick Commands

## Build & Install (Everything)

```bash
./build_and_install.sh
```

Or for Release build:

```bash
./build_and_install.sh Release
```

## Manual Steps

### Build

```bash
cmake --build build --config Release -j8

```text

Or Debug:

```bash
cmake --build build --config Debug -j8

```text

### Copy

```bash
# VST3
cp -r build/KellyMidiCompanion_artefacts/Release/VST3/Kelly\ MIDI\ Companion.vst3 ~/Library/Audio/Plug-Ins/VST3/

# AU (macOS only)
cp -r build/KellyMidiCompanion_artefacts/Release/AU/Kelly\ MIDI\ Companion.component ~/Library/Audio/Plug-Ins/Components/

```text

### Sign (IMPORTANT!)

```bash
# VST3
codesign --force --deep --sign - ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3

# AU
codesign --force --deep --sign - ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component

```text

### Remove Quarantine

```bash
# VST3
xattr -cr ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3

# AU
xattr -cr ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component

```text

## Test in Logic

```bash
open -a "Logic Pro X"

```text

## All-in-One Manual Install

```bash
# Build
cmake --build build --config Release -j8

# Copy VST3
cp -r build/KellyMidiCompanion_artefacts/Release/VST3/Kelly\ MIDI\ Companion.vst3 ~/Library/Audio/Plug-Ins/VST3/

# Copy AU
cp -r build/KellyMidiCompanion_artefacts/Release/AU/Kelly\ MIDI\ Companion.component ~/Library/Audio/Plug-Ins/Components/

# Remove quarantine
xattr -cr ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3
xattr -cr ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component

# Sign
codesign --force --deep --sign - ~/Library/Audio/Plug-Ins/VST3/Kelly\ MIDI\ Companion.vst3
codesign --force --deep --sign - ~/Library/Audio/Plug-Ins/Components/Kelly\ MIDI\ Companion.component

```text

## Troubleshooting

If plugin doesn't load:
1. Check it's in the right directory
2. Remove quarantine: `xattr -cr <plugin_path>`
3. Sign: `codesign --force --deep --sign - <plugin_path>`
4. Rescan plugins in your DAW
5. Check Console.app for error messages

