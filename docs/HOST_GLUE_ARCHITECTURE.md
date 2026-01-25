# Host Glue Architecture

**Date:** January 18, 2026  
**Purpose:** Define the host glue layer that translates between frameworks and DSP  
**Reference:** Architectural Boundary Compliance Report, Plugin Patterns

## Core Principle

**"Host glue is plumbing. Important. Unsexy. Replaceable."**

Host glue translates formats, manages lifecycle, handles threading. It does NOT contain DSP logic or UI logic.

## What Host Glue Does

### ✅ Host Glue Responsibilities

1. **Format Translation**
   - Host format → DSP format
   - DSP format → Host format
   - Parameter mapping between systems

2. **Lifecycle Management**
   - Plugin instantiation
   - State save/load
   - Resource allocation/deallocation

3. **Threading Boundaries**
   - Audio thread ↔ UI thread communication
   - Lock-free queues for parameter updates
   - Thread-safe state management

4. **Memory Policies**
   - Buffer allocation strategies
   - Memory pool management
   - Resource cleanup

5. **Host Quirks Handling**
   - DAW-specific behaviors
   - Platform-specific requirements
   - Framework limitations

## What Host Glue Does NOT Do

### ❌ Forbidden Host Glue Operations

1. **DSP Logic**
   ```cpp
   // WRONG: Host glue implementing DSP
   void PluginProcessor::processBlock(...) {
       // DSP logic here
       for (int i = 0; i < numSamples; ++i) {
           output[i] = input[i] * gain;  // This is DSP, not glue
       }
   }
   ```

2. **UI Logic**
   ```cpp
   // WRONG: Host glue implementing UI
   void PluginEditor::paint(...) {
       // Complex UI rendering
       drawCustomWaveform(...);  // This is UI, not glue
   }
   ```

3. **Creative Decisions**
   ```cpp
   // WRONG: Host glue making creative choices
   void PluginProcessor::processBlock(...) {
       if (shouldApplyEffect()) {  // Creative decision
           // ...
       }
   }
   ```

## Host Glue Structure

### Directory Organization

```
host/
├── juce/
│   ├── PluginProcessor.cpp    # JUCE plugin processor
│   ├── PluginEditor.cpp        # JUCE plugin UI
│   ├── ParameterAdapter.cpp    # Parameter translation
│   └── StateManager.cpp        # State save/load
├── standalone/
│   ├── AudioEngine.cpp         # Core Audio integration
│   ├── DeviceManager.cpp       # Audio device management
│   └── TransportControl.cpp   # Playback control
└── common/
    ├── ParameterBridge.cpp     # Parameter translation
    ├── StateSerializer.cpp     # State serialization
    └── ThreadSafeQueue.h       # Thread communication
```

## JUCE Plugin Host Glue

### Plugin Processor

```cpp
// host/juce/PluginProcessor.cpp
#include "dsp/Engine.h"  // Host glue includes DSP

class PluginProcessor : public juce::AudioProcessor {
    dsp::Engine dspEngine_;  // DSP core (pure)
    Parameters currentParams_;  // Parameter state
    
public:
    void prepareToPlay(double sampleRate, int blockSize) override {
        // Host glue: Prepare DSP with host-provided settings
        dspEngine_.prepare(sampleRate, blockSize);
    }
    
    void processBlock(juce::AudioBuffer<float>& buffer, 
                     juce::MidiBuffer& midi) override {
        // Host glue: Convert JUCE buffers to plain arrays
        float* channels[buffer.getNumChannels()];
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            channels[ch] = buffer.getWritePointer(ch);
        }
        
        // Host glue: Update parameters from host automation
        updateParametersFromHost();
        
        // Host glue: Call pure DSP
        dspEngine_.processBlock(
            channels,
            buffer.getNumChannels(),
            buffer.getNumSamples(),
            currentParams_,
            dspState_
        );
    }
    
    void getStateInformation(juce::MemoryBlock& destData) override {
        // Host glue: Serialize state for host
        auto state = dspEngine_.getState();
        serializeState(state, destData);
    }
    
    void setStateInformation(const void* data, int sizeInBytes) override {
        // Host glue: Deserialize state from host
        auto state = deserializeState(data, sizeInBytes);
        dspEngine_.setState(state);
    }
    
private:
    void updateParametersFromHost() {
        // Host glue: Read host automation
        currentParams_.cutoff = *cutoffParameter;
        currentParams_.resonance = *resonanceParameter;
    }
};
```

### Plugin Editor

```cpp
// host/juce/PluginEditor.cpp
class PluginEditor : public juce::AudioProcessorEditor {
    PluginProcessor& processor_;
    
public:
    PluginEditor(PluginProcessor& p) 
        : AudioProcessorEditor(&p), processor_(p) {
        // Host glue: Create UI components
        addAndMakeVisible(cutoffSlider);
        addAndMakeVisible(resonanceSlider);
        
        // Host glue: Connect UI to parameters
        cutoffAttachment = std::make_unique<SliderAttachment>(
            processor_.parameters, "cutoff", cutoffSlider
        );
    }
    
    void paint(juce::Graphics& g) override {
        // Host glue: Read parameter snapshot (not live audio)
        auto params = processor_.getParameterSnapshot();
        
        // UI displays parameters (UI responsibility, not glue)
        g.drawText("Cutoff: " + juce::String(params.cutoff), ...);
    }
    
    void sliderValueChanged(juce::Slider* slider) override {
        // Host glue: Write user intent to parameters
        if (slider == &cutoffSlider) {
            processor_.setParameter("cutoff", slider->getValue());
        }
    }
};
```

## Standalone App Host Glue

### Audio Engine

```cpp
// host/standalone/AudioEngine.cpp
#include "dsp/Engine.h"  // Host glue includes DSP
#include <CoreAudio/CoreAudio.h>

class AudioEngine {
    dsp::Engine dspEngine_;
    AudioUnit audioUnit_;
    
public:
    void initialize() {
        // Host glue: Set up Core Audio
        setupAudioUnit();
        
        // Host glue: Prepare DSP
        dspEngine_.prepare(44100.0, 512);
    }
    
    OSStatus renderCallback(AudioUnitRenderActionFlags* flags,
                           const AudioTimeStamp* timeStamp,
                           UInt32 busNumber,
                           UInt32 numFrames,
                           AudioBufferList* ioData) {
        // Host glue: Convert Core Audio buffers to plain arrays
        float* channels[ioData->mNumberBuffers];
        for (UInt32 i = 0; i < ioData->mNumberBuffers; ++i) {
            channels[i] = (float*)ioData->mBuffers[i].mData;
        }
        
        // Host glue: Call pure DSP
        dspEngine_.processBlock(
            channels,
            ioData->mNumberBuffers,
            numFrames,
            currentParams_,
            dspState_
        );
        
        return noErr;
    }
    
private:
    void setupAudioUnit() {
        // Host glue: Core Audio setup (OS-specific)
        // ...
    }
};
```

### Device Manager

```cpp
// host/standalone/DeviceManager.cpp
class DeviceManager {
public:
    std::vector<AudioDevice> enumerateDevices() {
        // Host glue: Enumerate audio devices (OS-specific)
        // ...
    }
    
    bool setDevice(const AudioDevice& device) {
        // Host glue: Set audio device (OS-specific)
        // ...
    }
};
```

## Parameter Translation

### Host → DSP Parameter Mapping

```cpp
// host/common/ParameterBridge.cpp
class ParameterBridge {
public:
    // Convert host parameter format to DSP format
    dsp::Parameters hostToDSP(const HostParameters& host) {
        dsp::Parameters dsp;
        
        // Host glue: Map host parameters to DSP parameters
        dsp.cutoff = normalizeCutoff(host.cutoffValue);
        dsp.resonance = normalizeResonance(host.resonanceValue);
        
        return dsp;
    }
    
    // Convert DSP parameter format to host format
    HostParameters dspToHost(const dsp::Parameters& dsp) {
        HostParameters host;
        
        // Host glue: Map DSP parameters to host parameters
        host.cutoffValue = denormalizeCutoff(dsp.cutoff);
        host.resonanceValue = denormalizeResonance(dsp.resonance);
        
        return host;
    }
    
private:
    float normalizeCutoff(float value) {
        // Host glue: Parameter normalization (host-specific)
        return value / 20000.0f;  // Normalize to 0-1
    }
};
```

## State Management

### State Serialization

```cpp
// host/common/StateSerializer.cpp
class StateSerializer {
public:
    // Serialize DSP state for host storage
    juce::MemoryBlock serialize(const dsp::State& state) {
        juce::MemoryBlock block;
        
        // Host glue: Serialize state (format-specific)
        juce::MemoryOutputStream stream(block, false);
        stream.writeFloat(state.phase);
        stream.writeFloat(state.lastSample);
        // ...
        
        return block;
    }
    
    // Deserialize state from host storage
    dsp::State deserialize(const juce::MemoryBlock& block) {
        dsp::State state;
        
        // Host glue: Deserialize state (format-specific)
        juce::MemoryInputStream stream(block, false);
        state.phase = stream.readFloat();
        state.lastSample = stream.readFloat();
        // ...
        
        return state;
    }
};
```

## Thread Communication

### Lock-Free Parameter Queue

```cpp
// host/common/ThreadSafeQueue.h
template<typename T>
class LockFreeQueue {
    std::atomic<size_t> writeIndex_{0};
    std::atomic<size_t> readIndex_{0};
    T buffer_[QUEUE_SIZE];
    
public:
    // UI thread: Push parameter update
    bool push(const T& item) {
        size_t next = (writeIndex_.load() + 1) % QUEUE_SIZE;
        if (next == readIndex_.load()) {
            return false;  // Queue full
        }
        buffer_[writeIndex_.load()] = item;
        writeIndex_.store(next);
        return true;
    }
    
    // Audio thread: Try to pop parameter update
    bool tryPop(T& item) {
        if (readIndex_.load() == writeIndex_.load()) {
            return false;  // Queue empty
        }
        item = buffer_[readIndex_.load()];
        readIndex_.store((readIndex_.load() + 1) % QUEUE_SIZE);
        return true;
    }
};
```

## Host-Specific Adaptations

### JUCE Plugin Format

```cpp
// host/juce/PluginProcessor.cpp
// JUCE-specific host glue
class PluginProcessor : public juce::AudioProcessor {
    // JUCE parameter system
    juce::AudioProcessorValueTreeState parameters_;
    
    // Host glue: Adapt JUCE parameters to DSP
    void updateDSPParameters() {
        dspParams_.cutoff = *parameters_.getRawParameterValue("cutoff");
        dspParams_.resonance = *parameters_.getRawParameterValue("resonance");
    }
};
```

### Core Audio Format

```cpp
// host/standalone/AudioEngine.cpp
// Core Audio-specific host glue
class AudioEngine {
    // Core Audio unit
    AudioUnit audioUnit_;
    
    // Host glue: Adapt Core Audio to DSP
    OSStatus renderCallback(...) {
        // Core Audio buffer format → DSP format
        // ...
    }
};
```

## Testing Host Glue

### Test 1: Host Glue Compiles Without DSP

```cpp
// Test: Host glue structure is independent
// (DSP is linked, but structure is separate)
void test_host_glue_structure() {
    // Host glue should have clear interface to DSP
    // Not embedded DSP logic
}
```

### Test 2: Port to New Host Format

```cpp
// Test: Can create new host glue without changing DSP
void test_portability() {
    // Create new host glue for different format
    // DSP core remains unchanged
    class NewHostProcessor {
        dsp::Engine dspEngine_;  // Same DSP core
        // New host-specific glue only
    };
}
```

### Test 3: State Round-Trip

```cpp
// Test: State save/load works correctly
void test_state_roundtrip() {
    dsp::State originalState = createTestState();
    
    // Host glue: Serialize
    auto serialized = serializer.serialize(originalState);
    
    // Host glue: Deserialize
    auto loadedState = serializer.deserialize(serialized);
    
    assert(loadedState == originalState);
}
```

## Migration Checklist

If existing code violates host glue boundaries:

- [ ] Extract DSP logic from host glue
- [ ] Extract UI logic from host glue
- [ ] Create clear parameter translation layer
- [ ] Implement proper state serialization
- [ ] Add thread-safe communication mechanisms
- [ ] Document host-specific adaptations
- [ ] Test: Port to new host format

## References

- `ARCHITECTURAL_BOUNDARY_COMPLIANCE_REPORT.md` - Full boundary analysis
- `cpp/PLUGIN_PATTERNS.md` - JUCE plugin patterns
- `cpp_audio_architecture.md` - Brain/Body architecture