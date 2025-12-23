# Music Brain VST/AU Plugin - Implementation Plan

## Overview

Transform Music Brain into a DAW plugin (VST3/AU) for real-time MIDI groove transformation within Logic Pro, Ableton, FL Studio, etc.

---

## Architecture Options

### Option 1: JUCE Framework (RECOMMENDED)
**Pros:**
- Industry standard for audio plugins
- Cross-platform (VST3, AU, AAX)
- Built-in MIDI processing
- Professional UI components
- Extensive documentation

**Cons:**
- C++ learning curve
- Python backend needs bridging

**Tech Stack:**
- C++ with JUCE framework
- Python backend via embedded interpreter
- IPC bridge for communication

### Option 2: Python-based (PJUCE/PlugData)
**Pros:**
- Keep existing Python codebase
- Faster prototyping
- Direct access to music_brain

**Cons:**
- Limited plugin framework support
- Performance concerns
- Distribution complexity (Python runtime)

### Option 3: Hybrid (Python + C++ Wrapper)
**Pros:**
- Best of both worlds
- Reuse music_brain logic
- Professional plugin interface

**Cons:**
- More complex architecture
- Requires both C++ and Python expertise

---

## Recommended Approach: JUCE + Python Backend

### Phase 1: JUCE Plugin Shell
**Goal:** Create empty VST3/AU that loads in DAW

**Tasks:**
1. Setup JUCE Projucer project
2. Configure plugin formats (VST3, AU, Standalone)
3. Implement basic MIDI pass-through
4. Test loading in Logic Pro

**Deliverable:** Empty plugin that receives/sends MIDI

### Phase 2: Python Integration
**Goal:** Bridge JUCE C++ with music_brain Python

**Options:**
```cpp
// Option A: Embedded Python
#include <Python.h>

class MusicBrainProcessor {
    PyObject* pModule;
    PyObject* pGrooveEngine;

    void initPython() {
        Py_Initialize();
        pModule = PyImport_ImportModule("music_brain.groove");
        pGrooveEngine = PyObject_CallMethod(pModule, "Applicator", NULL);
    }

    void processMIDI(MidiBuffer& buffer) {
        // Convert JUCE MIDI to Python
        PyObject* pMidi = convertToP ython(buffer);
        PyObject* pResult = PyObject_CallMethod(pGrooveEngine, "apply", "O", pMidi);
        // Convert result back to JUCE MIDI
        convertFromPython(pResult, buffer);
    }
};
```

```cpp
// Option B: IPC via Socket
class MusicBrainProcessor {
    zmq::socket_t socket;

    void processMIDI(MidiBuffer& buffer) {
        // Serialize MIDI to JSON
        auto json = serializeMIDI(buffer);
        socket.send(json);

        // Receive processed MIDI
        auto response = socket.recv();
        deserializeMIDI(response, buffer);
    }
};
```

**Recommended:** Option A (Embedded Python) for lower latency

### Phase 3: Core Features
**Goal:** Implement groove transformation

**Features:**
1. **Genre Selector**
   - Dropdown: Rock, Jazz, Hip-Hop, EDM, etc.
   - Load corresponding template

2. **Intensity Control**
   - Slider: 0-100%
   - Blend dry/wet signal

3. **Timing Controls**
   - Swing amount
   - Pocket offset
   - Timing tightness

4. **Velocity Shaping**
   - Curve selection
   - Dynamic range
   - Ghost note threshold

5. **Real-time Preview**
   - Visualize MIDI before/after
   - Piano roll display

### Phase 4: UI Design
**Goal:** Professional, usable interface

**Layout:**
```
┌─────────────────────────────────────────────┐
│  Music Brain Groove Engine          v1.0.0  │
├─────────────────────────────────────────────┤
│                                               │
│  Genre: [Rock ▼]          Intensity: ▓▓▓░░   │
│                                      [75%]    │
│                                               │
│  ┌───── TIMING ─────┐  ┌───── VELOCITY ────┐ │
│  │ Swing:    [0.62] │  │ Curve:    [▲]     │ │
│  │ Pocket:   [+5ms] │  │ Range:    [40dB]  │ │
│  │ Tight:    [0.80] │  │ Ghosts:   [30]    │ │
│  └──────────────────┘  └────────────────────┘ │
│                                               │
│  ┌────────── PIANO ROLL ──────────────────┐  │
│  │  Input:   ▂▄▅▇▁▃▄▆  (straight)        │  │
│  │  Output:  ▃▆▃▇▂▁▄▅  (grooved)         │  │
│  └──────────────────────────────────────┘  │
│                                               │
│  [Bypass]  [A/B Compare]      [Save Preset]  │
└─────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Setup Development Environment
```bash
# Install JUCE
brew install juce

# Clone template
git clone https://github.com/juce-framework/JUCE
cd JUCE/extras/Projucer/Builds/MacOSX
xcodebuild

# Create new plugin project
./Projucer.app
```

### Step 2: Create Basic Plugin
```cpp
// PluginProcessor.h
class MusicBrainAudioProcessor : public juce::AudioProcessor {
public:
    void processBlock(juce::AudioBuffer<float>& buffer,
                     juce::MidiBuffer& midiMessages) override {
        // Process MIDI here
        for (const auto midi : midiMessages) {
            // Transform using music_brain
        }
    }

    juce::AudioProcessorEditor* createEditor() override {
        return new MusicBrainAudioProcessorEditor(*this);
    }
};
```

### Step 3: Python Bridge
```cpp
// MusicBrainEngine.h
class MusicBrainEngine {
    PyObject* groove_applicator;

public:
    MusicBrainEngine() {
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('/Users/.../DAiW')");

        PyObject* pModule = PyImport_ImportModule("music_brain.groove.applicator");
        groove_applicator = PyObject_GetAttrString(pModule, "Applicator");
    }

    void applyGroove(std::vector<MidiMessage>& messages,
                     const std::string& genre,
                     float intensity) {
        // Convert MIDI to Python dict
        PyObject* pMidi = convertMIDI(messages);
        PyObject* pGenre = PyUnicode_FromString(genre.c_str());
        PyObject* pIntensity = PyFloat_FromDouble(intensity);

        // Call Python function
        PyObject* pResult = PyObject_CallMethod(
            groove_applicator, "apply",
            "OOO", pMidi, pGenre, pIntensity
        );

        // Convert back
        convertFromPython(pResult, messages);
    }
};
```

### Step 4: UI Components
```cpp
// MusicBrainEditor.cpp
MusicBrainEditor::MusicBrainEditor(MusicBrainProcessor& p)
    : AudioProcessorEditor(&p), processor(p) {

    // Genre selector
    genreCombo.addItem("Rock", 1);
    genreCombo.addItem("Jazz", 2);
    genreCombo.addItem("Hip-Hop", 3);
    genreCombo.onChange = [this] { updateGenre(); };

    // Intensity slider
    intensitySlider.setRange(0.0, 1.0);
    intensitySlider.onValueChange = [this] { updateIntensity(); };

    // Add to UI
    addAndMakeVisible(genreCombo);
    addAndMakeVisible(intensitySlider);
}
```

---

## Packaging & Distribution

### macOS
```bash
# Build AU/VST3
xcodebuild -project MusicBrain.xcodeproj -scheme "MusicBrain - AU" -configuration Release

# Package with Python runtime
mkdir -p MusicBrain.component/Contents/Resources/python
cp -r ~/Desktop/DAiW MusicBrain.component/Contents/Resources/
# Embed Python framework

# Install
cp -r MusicBrain.component ~/Library/Audio/Plug-Ins/Components/
cp -r MusicBrain.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

### Windows
```powershell
# Build VST3
MSBuild MusicBrain.sln /p:Configuration=Release

# Package with embedded Python
xcopy DAiW\ MusicBrain.vst3\Contents\Resources\ /E

# Install
copy MusicBrain.vst3 "C:\Program Files\Common Files\VST3\"
```

---

## Testing Plan

### Unit Tests
- MIDI conversion (Python ↔ C++)
- Groove application accuracy
- Parameter ranges

### Integration Tests
- Load in Logic Pro
- Load in Ableton Live
- Load in FL Studio
- Process MIDI region
- Save/recall presets

### Performance Tests
- Latency measurement (target: <5ms)
- CPU usage (target: <10%)
- Memory footprint (target: <100MB)

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. JUCE Shell | 1 week | Empty plugin that loads |
| 2. Python Bridge | 2 weeks | Python functions callable from C++ |
| 3. Core Features | 3 weeks | Groove transformation working |
| 4. UI Design | 2 weeks | Professional interface |
| 5. Testing & Polish | 2 weeks | Stable, optimized, documented |
| **Total** | **10 weeks** | **Production-ready VST/AU** |

---

## Alternative: Quick Prototype with REAPER ReaScript

For faster prototyping, create REAPER JS plugin first:

```javascript
// music_brain_groove.jsfx
desc:Music Brain Groove Engine

slider1:0<0,10,1{Rock,Jazz,HipHop,EDM}>Genre
slider2:0.75<0,1,0.01>Intensity

@slider
  genre = slider1;
  intensity = slider2;

@block
  while (midirecv(offset, msg1, msg2, msg3)) {
    // Apply groove transformation
    // (call Python via file-based IPC)
    midisend(offset, msg1, msg2, msg3);
  }
```

---

*Next Step: Begin Phase 1 - JUCE Plugin Shell*
