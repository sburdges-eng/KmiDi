# AU Plugin Architecture Reference

> Canonical reference for KmiDi's Audio Unit plugin format, hosting model,
> signal flow, and Apple-specific constraints. Read this before touching
> CMake, PluginProcessor, or anything AU-related.

## 1. Audio Unit Format Landscape

### AUv2 (Component Manager — the format DAWs actually use)

| Property | Value |
|---|---|
| Bundle extension | `.component` |
| Install paths | `/Library/Audio/Plug-Ins/Components/` (system) or `~/Library/Audio/Plug-Ins/Components/` (user) |
| Discovery | `AudioComponentRegistrar` scans install dirs on launch |
| API surface | C-based `AudioComponent*` / `AudioUnit*` functions |
| Hosting model | In-process dynamic library loaded by host |
| JUCE wrapper | `juce_audio_plugin_client_AU_1.mm` / `_AU_2.mm` |

AUv2 is the format Logic Pro, GarageBand, Reaper (macOS), Ableton Live (macOS),
and most macOS DAWs load natively. **This is our primary AU target.**

### AUv3 (App Extension — the Apple-blessed future)

| Property | Value |
|---|---|
| Bundle extension | `.appex` inside a host app bundle |
| API surface | `AUAudioUnit` Objective-C / Swift classes |
| Hosting model | Can run in-process or out-of-process (sandbox) |
| Sandbox | Runs in app extension sandbox by default |
| JUCE wrapper | `juce_audio_plugin_client_AUv3.mm` |

**Caveat:** Logic Pro's AUv3 support is incomplete as of macOS 15. Instruments
load but some effect configurations don't. GarageBand and third-party hosts
have better AUv3 support. For maximum Logic Pro compatibility, **target AUv2
first**, then add AUv3 as a secondary format.

### Recommendation for KmiDi

```
Primary:   AUv2  (.component)  — Logic Pro, GarageBand, all macOS DAWs
Secondary: AUv3  (.appex)      — future-proofing, iOS potential
Also ship: VST3, CLAP, Standalone
```

## 2. AU Type Codes

Apple defines four-character type codes that determine how a DAW categorizes
and routes audio/MIDI to your plugin.

| Type Code | Constant | Use Case | Audio In | Audio Out | MIDI In |
|---|---|---|---|---|---|
| `'aumf'` | `kAudioUnitType_MusicEffect` | Effect that responds to MIDI | Yes | Yes | Yes |
| `'aumu'` | `kAudioUnitType_MusicDevice` | Software instrument (synth/sampler) | No | Yes | Yes |
| `'aufx'` | `kAudioUnitType_Effect` | Audio effect (no MIDI) | Yes | Yes | No |
| `'aumi'` | `kAudioUnitType_MIDIProcessor` | MIDI-only processor (no audio) | No | No | Yes |

### KmiDi Plugin Type Decision

KmiDi's `PluginProcessor` is configured as:
```cpp
bool acceptsMidi()   const override { return true; }
bool producesMidi()  const override { return true; }
bool isMidiEffect()  const override { return true; }
```

This means JUCE will select `'aumi'` (MIDI Processor) by default. However,
KmiDi also has `MasterEQProcessor` which processes audio. The correct
configuration depends on which plugin variant we're building:

| Variant | Type | Rationale |
|---|---|---|
| **KmiDi Composer** (MIDI generation) | `'aumi'` | Generates MIDI from emotion/intent, no audio I/O |
| **KmiDi Full** (MIDI + audio EQ) | `'aumf'` | MIDI generation + master EQ on audio bus |
| **KmiDi Instrument** (future synth) | `'aumu'` | If we add wavetable/DDSP synthesis output |

For the initial AU build, **use `'aumi'` (MIDI Processor)** since the core
value proposition is emotion-driven MIDI generation. The Master EQ can be
a separate `'aufx'` plugin or combined into an `'aumf'` variant later.

## 3. Four-Character Identification Codes

Every AU requires three four-character codes registered with the system:

| Code | Current Value | Purpose |
|---|---|---|
| **Type** | `'aumi'` | Plugin category (see table above) |
| **Subtype** | `'Klp1'` | Unique plugin identifier within manufacturer |
| **Manufacturer** | `'Klly'` | Manufacturer identifier |

These codes are set in CMake via `juce_add_plugin()`:
```cmake
PLUGIN_MANUFACTURER_CODE Klly
PLUGIN_CODE Klp1
```

JUCE maps `AU_MAIN_TYPE` automatically from `isMidiEffect()` / `producesMidi()`.
To override, use:
```cmake
AU_MAIN_TYPE com.apple.audio.units.type.midi-processor
```

**Validation command:**
```bash
auval -v aumi Klp1 Klly
```

## 4. Signal Flow: Host → Plugin → Host

### MIDI Processor Flow (`'aumi'`)

```
┌─────────────────────────────────────────────────────┐
│ DAW HOST (Logic Pro, etc.)                          │
│                                                     │
│  MIDI Track ──► [KmiDi AU Plugin] ──► MIDI Output   │
│                      │                              │
│              ┌───────┴────────┐                     │
│              │ processBlock() │                     │
│              │                │                     │
│              │  1. Read host  │                     │
│              │     tempo/pos  │                     │
│              │  2. Read MIDI  │                     │
│              │     input      │                     │
│              │  3. Read params│                     │
│              │     (emotion,  │                     │
│              │      intent)   │                     │
│              │  4. Generate   │                     │
│              │     MIDI via   │                     │
│              │     KellyBrain │                     │
│              │  5. Write MIDI │                     │
│              │     output     │                     │
│              └────────────────┘                     │
│                                                     │
│  MIDI Output routes to instrument tracks            │
└─────────────────────────────────────────────────────┘
```

### Music Effect Flow (`'aumf'`) — Future variant

```
┌─────────────────────────────────────────────────────┐
│ DAW HOST                                            │
│                                                     │
│  Audio ──► [KmiDi AU] ──► Audio (EQ'd)              │
│  MIDI  ──►     │     ──► MIDI (generated)           │
│           ┌────┴─────┐                              │
│           │  process  │                              │
│           │  Block()  │                              │
│           │           │                              │
│           │ Audio in  │──► MasterEQ ──► Audio out    │
│           │ MIDI in   │──► KellyBrain ──► MIDI out  │
│           │ Params    │──► Both paths                │
│           └───────────┘                              │
└─────────────────────────────────────────────────────┘
```

## 5. Threading Model

AU hosts call `processBlock()` on the **real-time audio thread**. The existing
`PluginProcessor` already handles this correctly:

| Thread | What it does | Constraints |
|---|---|---|
| **Audio thread** | `processBlock()`, parameter reads via `getRawParameterValue()` | **No blocking**, no allocation, no locks (use `try_lock`) |
| **UI thread** | `parameterChanged()`, `generateMidi()`, editor painting | Can block, can allocate |
| **Inference thread** | ML model inference (`InferenceThreadManager`) | Background, feeds results atomically |

**Apple-specific:** On Apple Silicon, hosts may provide an `AudioWorkgroup` via
`audioWorkgroupContextChanged()`. The existing `PluginProcessor` already stores
this — workers should join the workgroup for optimal scheduling.

## 6. State Persistence

AU hosts save/restore plugin state via:
- `getStateInformation()` → serialize to `MemoryBlock`
- `setStateInformation()` → deserialize from buffer

The existing `PluginState` class handles this. Key AU-specific requirements:

1. **State must be deterministic** — same state bytes → same behavior
2. **Versioned** — use `PARAM_VERSION` for forward/backward compat
3. **Compact** — hosts may store state in project files, memory-mapped
4. **Fast** — `setStateInformation` is called on load, must not block audio

## 7. Bus Layouts

For `'aumi'` (MIDI only), no audio buses are needed. JUCE handles this when
`isMidiEffect()` returns `true`.

For `'aumf'` (music effect), declare:
```cpp
bool isBusesLayoutSupported(const BusesLayout& layouts) const override
{
    // Accept stereo or mono
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;
    // Input must match output
    return layouts.getMainInputChannelSet() == layouts.getMainOutputChannelSet();
}
```

## 8. Parameter Tree (AU-specific)

AU hosts expose parameters via `AUParameterTree`. JUCE maps
`AudioProcessorValueTreeState` parameters automatically. Current KmiDi
parameters that map to AU automation:

| Parameter ID | AU Display Name | Range | Default |
|---|---|---|---|
| `valence` | Valence | 0–1 | 0.5 |
| `arousal` | Arousal | 0–1 | 0.5 |
| `intensity` | Intensity | 0–100% | 50% |
| `complexity` | Complexity | 0–1 | 0.5 |
| `humanize` | Humanize | 0–100% | 50% |
| `feel` | Feel | 0–1 | 0.5 |
| `dynamics` | Dynamics | 0–1 | 0.5 |
| `bars` | Bars | 1–64 | 8 |
| `bypass` | Bypass | 0/1 | 0 |
| `eq_bypass` | EQ Bypass | 0/1 | 0 |
| `ai_eq_intensity` | AI EQ Amount | 0–100% | 50% |

All parameters support host automation. Smoothing is handled in `processBlock()`
per the spec in `docs/specs/07_PLUGIN_SPECIFIC.md`.

## 9. Latency Reporting

If the plugin introduces latency (e.g., ML lookahead buffer), report it via:
```cpp
int getLatencySamples() const override
{
    return lookaheadSamples_; // Currently ML_LOOKAHEAD_MS = 20ms
}
```

AU hosts use this to compensate plugin delay. The existing
`PluginLatencyManager` already tracks this.

## 10. Key Apple Documentation

| Resource | URL |
|---|---|
| Audio Unit Types | https://developer.apple.com/documentation/audiotoolbox/audio_unit_v2_c_api/1584142-audio_unit_types |
| Creating an AU Extension | https://developer.apple.com/documentation/avfaudio/creating-an-audio-unit-extension |
| Audio Components | https://developer.apple.com/documentation/audiotoolbox/audio-components |
| auval reference | `man auvaltool` on macOS |
