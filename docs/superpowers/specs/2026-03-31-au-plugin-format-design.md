# AU Plugin Format — Design Spec

**Date:** 2026-03-31
**Phase:** 3 (Local AU Helper) of the 90-Day Demo Roadmap

## Goal

Enable Audio Unit format for the existing KellyPlugin so it loads in Logic Pro / GarageBand as a Music Effect, exposing emotion parameters to AU host automation.

## Changes

### CMakeLists.txt

Add `AU` to the `FORMATS` list in `juce_add_plugin(KellyPlugin ...)` (currently `VST3 CLAP`).

Set AU-specific properties:
- `AU_MAIN_TYPE` = `kAudioUnitType_MusicEffect` — processes audio AND generates MIDI

### No Code Changes

The existing PluginProcessor, APVTS parameters, ML inference pipeline, Master EQ, and MIDI generation all work unchanged under AU. JUCE handles the AU wrapper automatically.

## What This Achieves

- KellyPlugin builds as VST3 + CLAP + AU
- Appears in Logic Pro under Music Effects (Audio → MIDI)
- Exposes 21 emotion/generation parameters + 32 EQ parameters to AU host automation
- Audio passthrough with Master EQ processing
- MIDI generation from emotion parameters

## Testing

1. Build: `cmake -S . -B build -G Ninja -DBUILD_PLUGINS=ON -DKMIDI_BUILD_JUCE_UI=ON && cmake --build build --target KellyPlugin_AU -j8`
2. Verify VST3 still builds: `cmake --build build --target KellyPlugin_VST3 -j8`
3. Manual: Load AU in Logic Pro, verify parameter exposure and audio passthrough

## What This Does NOT Include

- Wiring AudioEmotionRunner into the plugin (separate task)
- New parameters or UI changes
- Core ML integration
- Any code changes to PluginProcessor or PluginEditor

## Acceptance Criteria

1. `AU` appears in `FORMATS` list in CMakeLists.txt
2. `cmake --build build --target KellyPlugin_AU` succeeds (when JUCE is available)
3. Existing VST3 target unaffected
