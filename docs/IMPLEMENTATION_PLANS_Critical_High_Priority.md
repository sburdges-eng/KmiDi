# Detailed Implementation Plans: Critical & High Priority Features

## Table of Contents
1. [Critical Priority](#critical-priority)
   - [KellyBrain MIDI Generation](#1-kellybrain-midi-generation)
   - [MidiIO Implementation](#2-midiiio-implementation)
   - [OSC System (Client/Server/Queue)](#3-osc-system-implementation)
   - [Groove Engine Core](#4-groove-engine-core-implementation)
2. [High Priority](#high-priority)
   - [UI Panel Implementations](#5-ui-panel-implementations)
   - [LearningPanel Features](#6-learningpanel-features)
   - [ML Inference Processing](#7-ml-inference-processing)

---

## Critical Priority

### 1. KellyBrain MIDI Generation

**Status**: Placeholder implementation  
**File**: `src/engine/KellyBrain.cpp:228-253`  
**Priority**: CRITICAL - Blocks core functionality

#### Overview
The `generateMidi()` method currently returns empty MIDI notes. Need to integrate with `MidiGenerator` to produce actual MIDI sequences from intent results.

#### Dependencies
- `src/midi/MidiGenerator.h` - MIDI generation engine (fully implemented)
- `src/music_theory/MusicTheoryBrain.h` - Music theory processing
- `src/common/KellyTypes.h` - Type definitions
- `src/engines/*.h` - All MIDI generation engines (MelodyEngine, BassEngine, etc.)

#### Current MidiGenerator Interface
```cpp
// From src/midi/MidiGenerator.h
class MidiGenerator {
public:
    MidiGenerator();
    
    GeneratedMidi generate(
        const IntentResult& intent,
        int bars = 8,
        float complexity = 0.5f,
        float humanize = 0.4f,
        float feel = 0.0f,
        float dynamics = 0.75f
    );
    
private:
    ChordGenerator chordGen_;
    GrooveEngine grooveEngine_;
    MelodyEngine melodyEngine_;
    BassEngine bassEngine_;
    PadEngine padEngine_;
    // ... and more engines
};
```

#### Implementation Steps

**Step 1: Review IntentResult to GeneratedMidi Conversion**
The `MidiGenerator::generate()` method already exists and takes `IntentResult`. We need to convert `KellyTypesIntentResult` to `IntentResult` (from `Types.h`).

**Step 2: Add MidiGenerator Member to KellyBrain**
```cpp
// In KellyBrain.h, add:
#include "midi/MidiGenerator.h"

class KellyBrain {
private:
    std::unique_ptr<IntentPipeline> pipeline_;
    std::unique_ptr<MidiGenerator> midiGenerator_;  // ADD THIS
    bool initialized_ = false;
};
```

**Step 3: Initialize MidiGenerator in Constructor or initialize()**
```cpp
// In KellyBrain.cpp
KellyBrain::KellyBrain() 
    : pipeline_(std::make_unique<IntentPipeline>())
    , midiGenerator_(std::make_unique<MidiGenerator>())  // ADD THIS
{
}

bool KellyBrain::initialize(const std::string& dataPath) {
    // Existing initialization...
    initialized_ = true;
    
    // MidiGenerator is already constructed, no additional setup needed
    // It initializes all engines internally
    
    return true;
}
```

**Step 4: Implement generateMidi() with Type Conversion**
```cpp
// Note: KellyTypesIntentResult is actually IntentResult (Types.h includes KellyTypes.h)
// But we need to ensure emotion field is synced with sourceWound.primaryEmotion

GeneratedMidi KellyBrain::generateMidi(const KellyTypesIntentResult& intent, int bars) {
    if (!midiGenerator_) {
        // Fallback: Return empty structure
        GeneratedMidi result;
        result.tempoBpm = intent.tempoBpm;
        result.bars = bars;
        result.key = intent.key;
        result.mode = intent.mode;
        return result;
    }
    
    // Convert KellyTypesIntentResult to IntentResult for MidiGenerator
    // Since Types.h includes KellyTypes.h, they're compatible, but sync emotion field
    IntentResult intentForGenerator = intent;
    intentForGenerator.emotion = intent.sourceWound.primaryEmotion;
    intentForGenerator.tempo = static_cast<float>(intent.tempoBpm) / 120.0f;
    
    // Extract parameters from intent
    float complexity = 0.5f;  // Default, could be derived from intent
    float humanize = intent.humanization;
    float feel = 0.0f;  // Could map from intent.syncopationLevel
    float dynamics = intent.dynamicRange;
    
    // Generate MIDI using MidiGenerator
    GeneratedMidi result = midiGenerator_->generate(
        intentForGenerator,
        bars,
        complexity,
        humanize,
        feel,
        dynamics
    );
    
    // Ensure result has correct metadata
    result.tempoBpm = intent.tempoBpm;
    result.bars = bars;
    result.key = intent.key;
    result.mode = intent.mode;
    result.lengthInBeats = bars * 4.0;  // Assuming 4/4 time
    result.bpm = static_cast<float>(intent.tempoBpm);
    
    return result;
}
```

**Step 5: Remove Fallback Helper (No Longer Needed)**
The MidiGenerator already handles all MIDI generation, so the `generateBasicMidiNotes()` helper is not needed.

**Step 5: Helper Function for Basic MIDI Generation**
```cpp
void KellyBrain::generateBasicMidiNotes(GeneratedMidi& midi, 
                                       const KellyTypesIntentResult& intent,
                                       int bars) {
    const float beatsPerBar = 4.0f;
    const int ticksPerBeat = 480;
    const int ticksPerBar = static_cast<int>(beatsPerBar * ticksPerBeat);
    
    int currentTick = 0;
    
    for (int bar = 0; bar < bars; ++bar) {
        for (const auto& chord : midi.chords) {
            // Generate chord notes
            std::vector<int> chordNotes = getChordNotes(chord.symbol, intent.key);
            
            // Add note-on events
            for (int note : chordNotes) {
                MidiNote midiNote;
                midiNote.pitch = note;
                midiNote.velocity = 80; // Default velocity
                midiNote.startTick = currentTick;
                midiNote.durationTicks = ticksPerBeat; // Quarter note
                midiNote.channel = 0;
                
                midi.notes.push_back(midiNote);
            }
            
            currentTick += ticksPerBeat;
        }
    }
}
```

#### Testing Approach
1. Unit test: Generate MIDI from simple intent
2. Integration test: Full wound → intent → MIDI pipeline
3. Verify: MIDI notes match chord progression
4. Verify: Tempo and time signature are correct

#### Estimated Effort
- Implementation: 4-6 hours
- Testing: 2-3 hours
- Total: 6-9 hours

---

### 2. MidiIO Implementation

**Status**: Stub implementation  
**File**: `src/midi/MidiIO.cpp`  
**Priority**: CRITICAL - Required for MIDI device communication

#### Overview
Replace stub implementation with actual MIDI I/O using JUCE MIDI classes (since JUCE is already a dependency).

#### Dependencies
- JUCE Framework (`juce_audio_devices`, `juce_audio_basics`)
- Platform-specific MIDI APIs (handled by JUCE)
- Already linked in CMakeLists.txt: `juce::juce_audio_devices`

#### Implementation Steps

**Step 1: Update Header to Use JUCE Types**
```cpp
// In src/midi/MidiIO.h (or daiw/midi/MidiIO.h), add JUCE includes
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <functional>
#include <vector>
#include <memory>

namespace daiw {
namespace midi {

struct MidiDeviceInfo {
    int id;
    std::string name;
    std::string identifier;
};

class MidiInput {
public:
    MidiInput();
    ~MidiInput();
    
    static std::vector<MidiDeviceInfo> getAvailableDevices();
    bool open(int deviceId);
    void close();
    void start();
    void stop();
    
    void setCallback(std::function<void(const juce::MidiMessage&)> callback);
    
private:
    std::unique_ptr<juce::MidiInput> juceMidiInput_;
    std::function<void(const juce::MidiMessage&)> callback_;
    int deviceId_ = -1;
    bool isOpen_ = false;
    bool isRunning_ = false;
    
    // JUCE callback wrapper
    class CallbackWrapper : public juce::MidiInputCallback {
    public:
        CallbackWrapper(MidiInput* owner) : owner_(owner) {}
        void handleIncomingMidiMessage(juce::MidiInput* source, 
                                      const juce::MidiMessage& message) override {
            if (owner_ && owner_->callback_) {
                owner_->callback_(message);
            }
        }
    private:
        MidiInput* owner_;
    };
    std::unique_ptr<CallbackWrapper> callbackWrapper_;
};
```

**Step 2: Implement Constructor and getAvailableDevices()**
```cpp
// In MidiIO.cpp
MidiInput::MidiInput()
    : deviceId_(-1)
    , isOpen_(false)
    , isRunning_(false)
    , callbackWrapper_(std::make_unique<CallbackWrapper>(this))
{
}

MidiInput::~MidiInput() {
    close();
}

std::vector<MidiDeviceInfo> MidiInput::getAvailableDevices() {
    std::vector<MidiDeviceInfo> devices;
    
    auto juceDevices = juce::MidiInput::getAvailableDevices();
    for (int i = 0; i < juceDevices.size(); ++i) {
        MidiDeviceInfo info;
        info.id = i;
        info.name = juceDevices[i].name.toStdString();
        info.identifier = juceDevices[i].identifier.toStdString();
        devices.push_back(info);
    }
    
    return devices;
}

void MidiInput::setCallback(std::function<void(const juce::MidiMessage&)> callback) {
    callback_ = callback;
}
```

**Step 3: Implement open() with Error Handling**
```cpp
bool MidiInput::open(int deviceId) {
    // Close existing device if open
    if (isOpen_) {
        close();
    }
    
    auto devices = juce::MidiInput::getAvailableDevices();
    
    if (deviceId < 0 || deviceId >= devices.size()) {
        juce::Logger::writeToLog("MidiInput::open: Invalid device ID: " + 
                                 juce::String(deviceId));
        return false;
    }
    
    auto device = devices[deviceId];
    
    // Open device with callback wrapper
    juceMidiInput_ = juce::MidiInput::openDevice(
        device.identifier, 
        callbackWrapper_.get()
    );
    
    if (juceMidiInput_) {
        deviceId_ = deviceId;
        isOpen_ = true;
        juce::Logger::writeToLog("MidiInput::open: Successfully opened device: " + 
                                 device.name);
        return true;
    }
    
    juce::Logger::writeToLog("MidiInput::open: Failed to open device: " + 
                             device.name);
    return false;
}
```

**Step 4: Implement start() and stop()**
```cpp
void MidiInput::start() {
    if (!isOpen_ || !juceMidiInput_) {
        return;
    }
    
    juceMidiInput_->start();
    isRunning_ = true;
}

void MidiInput::stop() {
    if (juceMidiInput_ && isRunning_) {
        juceMidiInput_->stop();
        isRunning_ = false;
    }
}
```

**Step 5: Implement MidiOutput**
```cpp
bool MidiOutput::open(int deviceId) {
    auto devices = juce::MidiOutput::getAvailableDevices();
    
    if (deviceId < 0 || deviceId >= devices.size()) {
        return false;
    }
    
    auto device = devices[deviceId];
    juceMidiOutput_ = juce::MidiOutput::openDevice(device.identifier);
    
    if (juceMidiOutput_) {
        deviceId_ = deviceId;
        isOpen_ = true;
        return true;
    }
    
    return false;
}

bool MidiOutput::sendMessage(const MidiMessage& message) {
    if (!isOpen_ || !juceMidiOutput_) {
        return false;
    }
    
    juce::MidiMessage juceMsg(message.data.data(), 
                              static_cast<int>(message.data.size()));
    juceMidiOutput_->sendMessageNow(juceMsg);
    return true;
}
```

#### Testing Approach
1. List available MIDI devices
2. Open MIDI input device
3. Receive MIDI messages
4. Open MIDI output device
5. Send MIDI messages
6. Test error handling (invalid device IDs)

#### Estimated Effort
- Implementation: 6-8 hours
- Testing: 3-4 hours
- Total: 9-12 hours

---

### 3. OSC System Implementation

**Status**: Stub implementations  
**Files**: 
- `src/osc/RTMessageQueue.cpp`
- `src/osc/OSCClient.cpp`
- `src/osc/OSCServer.cpp`  
**Priority**: CRITICAL - Required for DAW communication

#### Overview
Implement lock-free OSC message queue and RT-safe OSC client/server using oscpack library (already in external/).

#### Dependencies
- **Option 1**: `oscpack` library (external/oscpack-stub exists, need real implementation)
- **Option 2**: JUCE OSC (`juce_osc` module - already available in external/JUCE/)
- `readerwriterqueue` - Lock-free queue (header-only, needs real implementation)

**Decision**: Use JUCE OSC since JUCE is already a dependency and `juce_osc` is available. This avoids adding another external dependency.

#### Implementation Steps

**Step 1: Add readerwriterqueue Dependency**

First, add the real readerwriterqueue library via CMake FetchContent:

```cmake
# In src_penta-core/CMakeLists.txt or root CMakeLists.txt
include(FetchContent)

FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
)

FetchContent_MakeAvailable(readerwriterqueue)

# Then link to penta_core
target_link_libraries(penta_core PUBLIC readerwriterqueue)
```

**Step 2: Implement RTMessageQueue**
```cpp
// In src/osc/RTMessageQueue.cpp
#include "penta/osc/RTMessageQueue.h"
#include "readerwriterqueue.h"  // Real library, not stub

namespace penta::osc {

RTMessageQueue::RTMessageQueue(size_t capacity)
    : capacity_(capacity)
    , queue_(std::make_unique<moodycamel::ReaderWriterQueue<OSCMessage>>(capacity))
    , writeIndex_(0)
    , readIndex_(0)
{
    // Pre-allocate buffer for atomic operations
    buffer_.reserve(capacity);
}

RTMessageQueue::~RTMessageQueue() = default;

bool RTMessageQueue::push(const OSCMessage& message) noexcept {
    if (!queue_) {
        return false;
    }
    
    // RT-safe: try_enqueue never blocks
    bool success = queue_->try_enqueue(message);
    
    if (success) {
        writeIndex_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return success;
}

bool RTMessageQueue::pop(OSCMessage& outMessage) noexcept {
    if (!queue_) {
        return false;
    }
    
    // RT-safe: try_dequeue never blocks
    bool success = queue_->try_dequeue(outMessage);
    
    if (success) {
        readIndex_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return success;
}

bool RTMessageQueue::isEmpty() const noexcept {
    if (!queue_) {
        return true;
    }
    
    // RT-safe: size_approx() is lock-free
    return queue_->size_approx() == 0;
}

size_t RTMessageQueue::size() const noexcept {
    if (!queue_) {
        return 0;
    }
    
    // RT-safe: approximate size (lock-free)
    return queue_->size_approx();
}
```

**Note**: Update header to match implementation:
```cpp
// In include/penta/osc/RTMessageQueue.h
private:
    std::unique_ptr<moodycamel::ReaderWriterQueue<OSCMessage>> queue_;
    size_t capacity_;
    std::atomic<size_t> writeIndex_;
    std::atomic<size_t> readIndex_;
    std::vector<OSCMessage> buffer_;  // Pre-allocated buffer
```

**Step 3: Implement OSCClient Using JUCE OSC**

```cpp
// In src/osc/OSCClient.cpp
#include "penta/osc/OSCClient.h"
#include "penta/osc/OSCMessage.h"
#include <juce_osc/juce_osc.h>

namespace penta::osc {

struct OSCClient::SocketImpl {
    std::unique_ptr<juce::OSCSender> sender_;
    juce::String address_;
    int port_;
};

OSCClient::OSCClient(const std::string& address, uint16_t port)
    : address_(address)
    , port_(port)
    , socket_(std::make_unique<SocketImpl>())
{
    socket_->address_ = juce::String(address);
    socket_->port_ = static_cast<int>(port);
    socket_->sender_ = std::make_unique<juce::OSCSender>();
    
    if (!socket_->sender_->connect(socket_->address_, socket_->port_)) {
        juce::Logger::writeToLog("OSCClient: Failed to connect to " + 
                                 socket_->address_ + ":" + 
                                 juce::String(socket_->port_));
    }
}

OSCClient::~OSCClient() {
    if (socket_ && socket_->sender_) {
        socket_->sender_->disconnect();
    }
}

bool OSCClient::send(const OSCMessage& message) noexcept {
    if (!socket_ || !socket_->sender_ || !socket_->sender_->isConnected()) {
        return false;
    }
    
    try {
        juce::OSCMessage juceMsg(juce::String(message.getAddress()));
        
        // Add arguments from OSCMessage
        for (size_t i = 0; i < message.getArgumentCount(); ++i) {
            const auto& arg = message.getArgument(i);
            
            if (std::holds_alternative<int32_t>(arg)) {
                juceMsg.addInt32(std::get<int32_t>(arg));
            } else if (std::holds_alternative<float>(arg)) {
                juceMsg.addFloat32(std::get<float>(arg));
            } else if (std::holds_alternative<std::string>(arg)) {
                juceMsg.addString(juce::String(std::get<std::string>(arg)));
            }
            // Handle blob if needed
        }
        
        // RT-safe: send() is non-blocking
        return socket_->sender_->send(juceMsg);
    } catch (...) {
        return false;
    }
}

bool OSCClient::sendFloat(const char* address, float value) noexcept {
    if (!socket_ || !socket_->sender_ || !socket_->sender_->isConnected()) {
        return false;
    }
    
    juce::OSCMessage msg(juce::String(address));
    msg.addFloat32(value);
    return socket_->sender_->send(msg);
}

bool OSCClient::sendInt(const char* address, int32_t value) noexcept {
    if (!socket_ || !socket_->sender_ || !socket_->sender_->isConnected()) {
        return false;
    }
    
    juce::OSCMessage msg(juce::String(address));
    msg.addInt32(value);
    return socket_->sender_->send(msg);
}

bool OSCClient::sendString(const char* address, const char* value) noexcept {
    if (!socket_ || !socket_->sender_ || !socket_->sender_->isConnected()) {
        return false;
    }
    
    juce::OSCMessage msg(juce::String(address));
    msg.addString(juce::String(value));
    return socket_->sender_->send(msg);
}

void OSCClient::setDestination(const std::string& address, uint16_t port) {
    address_ = address;
    port_ = port;
    
    if (socket_ && socket_->sender_) {
        socket_->sender_->disconnect();
        socket_->address_ = juce::String(address);
        socket_->port_ = static_cast<int>(port);
        socket_->sender_->connect(socket_->address_, socket_->port_);
    }
}
```

**Step 4: Implement OSCServer Using JUCE OSC**

```cpp
// In src/osc/OSCServer.cpp
#include "penta/osc/OSCServer.h"
#include "penta/osc/OSCMessage.h"
#include <juce_osc/juce_osc.h>

namespace penta::osc {

// OSC message listener that pushes to RT-safe queue
class OSCServer::OSCListener : public juce::OSCReceiver::Listener<OSCReceiver::RealtimeCallback> {
public:
    OSCListener(OSCServer* server) : server_(server) {}
    
    void oscMessageReceived(const juce::OSCMessage& message) override {
        if (!server_ || !server_->messageQueue_) {
            return;
        }
        
        // Convert JUCE OSC message to penta::osc::OSCMessage
        OSCMessage pentaMsg;
        pentaMsg.setAddress(message.getAddressPattern().toString().toStdString());
        pentaMsg.setTimestamp(juce::Time::getMillisecondCounterHiRes());
        
        // Add arguments
        for (const auto& arg : message) {
            if (arg.isFloat32()) {
                pentaMsg.addFloat(arg.getFloat32());
            } else if (arg.isInt32()) {
                pentaMsg.addInt(arg.getInt32());
            } else if (arg.isString()) {
                pentaMsg.addString(arg.getString().toStdString());
            }
        }
        
        // Push to RT-safe queue (non-blocking)
        server_->messageQueue_->push(pentaMsg);
    }
    
private:
    OSCServer* server_;
};

OSCServer::OSCServer(const std::string& address, uint16_t port)
    : address_(address)
    , port_(port)
    , running_(false)
    , messageQueue_(std::make_unique<RTMessageQueue>(4096))
    , listener_(std::make_unique<OSCListener>(this))
    , receiver_()
{
    // Register listener
    receiver_.addListener(listener_.get());
}

OSCServer::~OSCServer() {
    stop();
    receiver_.removeListener(listener_.get());
}

bool OSCServer::start() {
    if (running_.load()) {
        return true; // Already running
    }
    
    // Connect receiver to port
    if (!receiver_.connect(static_cast<int>(port_))) {
        juce::Logger::writeToLog("OSCServer: Failed to bind to port " + 
                                 juce::String(port_));
        return false;
    }
    
    running_.store(true);
    juce::Logger::writeToLog("OSCServer: Started on port " + juce::String(port_));
    return true;
}

void OSCServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    receiver_.disconnect();
    running_.store(false);
    juce::Logger::writeToLog("OSCServer: Stopped");
}

RTMessageQueue& OSCServer::getMessageQueue() {
    return *messageQueue_;
}
```

**Note**: Update header to match implementation:
```cpp
// In include/penta/osc/OSCServer.h, add:
#include <juce_osc/juce_osc.h>

private:
    std::unique_ptr<OSCListener> listener_;
    juce::OSCReceiver receiver_;
```

#### CMake Integration

Add JUCE OSC module to CMakeLists.txt:

```cmake
# In src_penta-core/CMakeLists.txt or root CMakeLists.txt
target_link_libraries(penta_core PUBLIC
    juce::juce_osc  # ADD THIS
    readerwriterqueue
)
```

#### Testing Approach
1. **RTMessageQueue**: 
   - Test push/pop from audio thread (use JUCE AudioProcessorTest)
   - Verify lock-free behavior (no blocking)
   - Stress test with 10,000+ messages
   
2. **OSCClient**: 
   - Send messages to test server (use `oscdump` or similar)
   - Verify message encoding/decoding
   - Test reconnection on failure
   
3. **OSCServer**: 
   - Receive messages and verify queue
   - Test multiple simultaneous clients
   - Verify RT-safe queue operations
   
4. **Integration**: 
   - Test full round-trip (Client → Server → Queue → Consumer)
   - Stress test: High-frequency message sending (1000+ msg/sec)
   - Latency test: Measure RT-safe performance (<1ms target)

#### Error Handling
- Network failures: Log and return false, don't throw
- Queue full: Return false from push(), log warning
- Invalid messages: Skip and log, don't crash
- Disconnection: Auto-reconnect on next send (OSCClient)

#### Estimated Effort
- **CMake setup**: 1 hour
- **RTMessageQueue**: 2-3 hours
- **OSCClient**: 4-5 hours  
- **OSCServer**: 5-6 hours
- **Testing**: 4-5 hours
- **Total**: 16-20 hours

---

### 4. Groove Engine Core Implementation

**Status**: Stub implementations  
**File**: `src/groove/GrooveEngine.cpp`  
**Priority**: CRITICAL - Core rhythm analysis

#### Overview
Implement tempo estimation, time signature detection, and swing analysis.

#### Dependencies
- FFT library: JUCE FFT (`juce_dsp`) or FFTW3
- Autocorrelation algorithm (can implement directly)
- Sample rate information (from GrooveEngine config)

#### Current GrooveEngine Structure
```cpp
// From src/groove/GrooveEngine.cpp
class GrooveEngine {
    Config config_;  // Contains sampleRate
    GrooveAnalysis analysis_;
    std::unique_ptr<OnsetDetector> onsetDetector_;
    std::unique_ptr<TempoEstimator> tempoEstimator_;
    std::unique_ptr<RhythmQuantizer> quantizer_;
    uint64_t samplePosition_;
    std::vector<uint64_t> onsetHistory_;
};
```

#### Implementation Steps

**Step 1: Implement updateTempoEstimate()**
```cpp
void GrooveEngine::updateTempoEstimate() noexcept {
    if (onsetHistory_.size() < 2) {
        return;
    }
    
    // Calculate inter-onset intervals (IOIs)
    std::vector<float> iois;
    for (size_t i = 1; i < onsetHistory_.size(); ++i) {
        float ioi = (onsetHistory_[i] - onsetHistory_[i-1]) / sampleRate_;
        iois.push_back(ioi);
    }
    
    // Autocorrelation to find tempo
    float bestTempo = 120.0f;
    float bestCorrelation = 0.0f;
    
    // Test tempos from 60 to 200 BPM
    for (float testTempo = 60.0f; testTempo <= 200.0f; testTempo += 0.5f) {
        float testInterval = 60.0f / testTempo; // Seconds per beat
        
        float correlation = autocorrelateIOIs(iois, testInterval);
        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestTempo = testTempo;
        }
    }
    
    analysis_.currentTempo = bestTempo;
    analysis_.tempoConfidence = bestCorrelation;
}
```

**Step 2: Implement detectTimeSignature()**
```cpp
void GrooveEngine::detectTimeSignature() noexcept {
    if (onsetHistory_.size() < 8) {
        return; // Need more data
    }
    
    // Analyze beat patterns
    std::vector<float> beatStrengths = calculateBeatStrengths();
    
    // Find strongest beats (likely downbeats)
    int strongestBeat = 0;
    float maxStrength = 0.0f;
    for (size_t i = 0; i < beatStrengths.size(); ++i) {
        if (beatStrengths[i] > maxStrength) {
            maxStrength = beatStrengths[i];
            strongestBeat = static_cast<int>(i);
        }
    }
    
    // Count beats between strong beats
    int beatsPerMeasure = countBeatsBetweenStrongBeats(strongestBeat);
    
    // Common time signatures: 4/4, 3/4, 2/4
    if (beatsPerMeasure == 4) {
        analysis_.timeSignatureNum = 4;
        analysis_.timeSignatureDen = 4;
    } else if (beatsPerMeasure == 3) {
        analysis_.timeSignatureNum = 3;
        analysis_.timeSignatureDen = 4;
    } else {
        analysis_.timeSignatureNum = 4;
        analysis_.timeSignatureDen = 4; // Default
    }
}
```

**Step 3: Implement analyzeSwing()**
```cpp
void GrooveEngine::analyzeSwing() noexcept {
    if (onsetHistory_.size() < 4) {
        return;
    }
    
    // Analyze timing of off-beat notes (eighth notes)
    std::vector<float> onBeatTimings;
    std::vector<float> offBeatTimings;
    
    float beatInterval = 60.0f / analysis_.currentTempo;
    
    for (size_t i = 0; i < onsetHistory_.size(); ++i) {
        float position = onsetHistory_[i] / sampleRate_;
        float beatPosition = fmod(position, beatInterval) / beatInterval;
        
        if (beatPosition < 0.1f || beatPosition > 0.9f) {
            onBeatTimings.push_back(beatPosition);
        } else if (beatPosition > 0.4f && beatPosition < 0.6f) {
            offBeatTimings.push_back(beatPosition);
        }
    }
    
    if (offBeatTimings.empty()) {
        analysis_.swing = 0.0f; // No swing detected
        return;
    }
    
    // Calculate average off-beat timing
    float avgOffBeat = 0.0f;
    for (float timing : offBeatTimings) {
        avgOffBeat += timing;
    }
    avgOffBeat /= offBeatTimings.size();
    
    // Swing ratio: 0.5 = straight, 0.6+ = swung
    // Convert to -1.0 (straight) to 1.0 (heavy swing)
    analysis_.swing = (avgOffBeat - 0.5f) * 2.0f;
    analysis_.swing = std::clamp(analysis_.swing, -1.0f, 1.0f);
}
```

**Step 4: Implement quantizeToGrid()**
```cpp
uint64_t GrooveEngine::quantizeToGrid(uint64_t timestamp) const noexcept {
    float beatInterval = (60.0f / analysis_.currentTempo) * sampleRate_;
    float beatPosition = fmod(static_cast<float>(timestamp), beatInterval);
    
    // Quantize to nearest grid point
    float quantizedBeat = std::round(beatPosition / (beatInterval / 4.0f)) 
                         * (beatInterval / 4.0f);
    
    uint64_t quantizedTimestamp = timestamp - static_cast<uint64_t>(beatPosition) 
                                  + static_cast<uint64_t>(quantizedBeat);
    
    return quantizedTimestamp;
}
```

**Step 5: Implement applySwing()**
```cpp
uint64_t GrooveEngine::applySwing(uint64_t position) const noexcept {
    float beatInterval = (60.0f / analysis_.currentTempo) * sampleRate_;
    float beatPosition = fmod(static_cast<float>(position), beatInterval);
    float beatFraction = beatPosition / beatInterval;
    
    // Apply swing to off-beats (0.5 position)
    if (beatFraction > 0.4f && beatFraction < 0.6f) {
        float swingOffset = (analysis_.swing * 0.1f) * beatInterval;
        position = static_cast<uint64_t>(position + swingOffset);
    }
    
    return position;
}
```

#### Testing Approach
1. Test tempo estimation with known tempo audio
2. Test time signature detection (4/4, 3/4)
3. Test swing detection (straight vs swung)
4. Test quantization accuracy
5. Test with various musical styles

#### Estimated Effort
- Tempo estimation: 6-8 hours
- Time signature detection: 4-5 hours
- Swing analysis: 4-5 hours
- Quantization: 2-3 hours
- Testing: 4-5 hours
- Total: 20-26 hours

---

## High Priority

### 5. UI Panel Implementations

**Status**: Header-only files  
**Files**: 
- `src/ui/ScoreEntryPanel.h` (485 lines)
- `src/ui/MixerConsolePanel.h` (552 lines)  
**Priority**: HIGH - Major UI features

#### Overview
Implement full .cpp files for sheet music entry and mixing console panels.

#### ScoreEntryPanel Implementation

**Step 1: Create ScoreEntryPanel.cpp Structure**
```cpp
#include "ScoreEntryPanel.h"
#include <juce_graphics/juce_graphics.h>

namespace midikompanion {

ScoreEntryPanel::ScoreEntryPanel()
    : currentClef_(Clef::Treble)
    , currentKey_(Key::C)
    , currentTimeSig_(4, 4)
    , entryMode_(EntryMode::Standard)
{
    setupComponents();
}

void ScoreEntryPanel::setupComponents() {
    // Clef selector
    clefSelector_.addItem("Treble", 1);
    clefSelector_.addItem("Bass", 2);
    clefSelector_.addItem("Alto", 3);
    clefSelector_.addItem("Tenor", 4);
    clefSelector_.onChange = [this] {
        int selected = clefSelector_.getSelectedId();
        if (selected == 1) currentClef_ = Clef::Treble;
        else if (selected == 2) currentClef_ = Clef::Bass;
        // ... etc
        repaint();
    };
    addAndMakeVisible(clefSelector_);
    
    // Key signature selector
    keySelector_.addItem("C", 1);
    keySelector_.addItem("G", 2);
    // ... all keys
    addAndMakeVisible(keySelector_);
    
    // Note entry area (custom component)
    noteEntryArea_ = std::make_unique<NoteEntryArea>(*this);
    addAndMakeVisible(noteEntryArea_.get());
}

void ScoreEntryPanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff2a2a2a));
    
    // Draw staff
    drawStaff(g, getLocalBounds());
    
    // Draw key signature
    drawKeySignature(g);
    
    // Draw time signature
    drawTimeSignature(g);
    
    // Draw entered notes
    drawNotes(g);
}

void ScoreEntryPanel::drawStaff(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    g.setColour(juce::Colours::white);
    
    int staffTop = bounds.getY() + 50;
    int staffHeight = 80;
    int lineSpacing = staffHeight / 4;
    
    // Draw 5 lines
    for (int i = 0; i < 5; ++i) {
        int y = staffTop + (i * lineSpacing);
        g.drawLine(bounds.getX(), y, bounds.getRight(), y, 2.0f);
    }
}

void ScoreEntryPanel::addNoteAtPosition(int x, int y) {
    // Convert screen position to musical note
    int pitch = screenYToPitch(y);
    double beat = screenXToBeat(x);
    
    NotationNote note;
    note.pitch = pitch;
    note.startBeat = beat;
    note.value = NoteValue::Quarter;
    note.dynamic = Dynamic::MezzoForte;
    
    score_.notes.push_back(note);
    repaint();
}
```

#### MixerConsolePanel Implementation

**Step 1: Create MixerConsolePanel.cpp Structure**
```cpp
#include "MixerConsolePanel.h"

namespace midikompanion {

MixerConsolePanel::MixerConsolePanel()
    : viewMode_(ViewMode::Channels)
{
    setupChannels();
}

void MixerConsolePanel::setupChannels() {
    // Create default channels
    for (int i = 0; i < 8; ++i) {
        auto channel = std::make_unique<ChannelStrip>(
            "Channel " + std::to_string(i + 1));
        channels_.push_back(std::move(channel));
        addAndMakeVisible(channels_.back().get());
    }
}

void MixerConsolePanel::resized() {
    auto bounds = getLocalBounds();
    
    int channelWidth = 80;
    int x = 0;
    
    for (auto& channel : channels_) {
        channel->setBounds(x, 0, channelWidth, bounds.getHeight());
        x += channelWidth + 5; // 5px spacing
    }
}

void MixerConsolePanel::addChannel(const std::string& name) {
    auto channel = std::make_unique<ChannelStrip>(name);
    channels_.push_back(std::move(channel));
    addAndMakeVisible(channels_.back().get());
    resized();
}
```

**Step 2: Implement ChannelStrip**
```cpp
ChannelStrip::ChannelStrip(const std::string& channelName)
    : name_(channelName)
    , volume_(0.0f)
    , pan_(0.0f)
    , muted_(false)
    , soloed_(false)
{
    setupControls();
}

void ChannelStrip::setupControls() {
    // Volume fader
    volumeFader_.setSliderStyle(juce::Slider::LinearVertical);
    volumeFader_.setRange(0.0, 1.0, 0.01);
    volumeFader_.setValue(0.7);
    volumeFader_.onValueChange = [this] {
        volume_ = static_cast<float>(volumeFader_.getValue());
    };
    addAndMakeVisible(volumeFader_);
    
    // Pan knob
    panKnob_.setSliderStyle(juce::Slider::Rotary);
    panKnob_.setRange(-1.0, 1.0, 0.01);
    panKnob_.setValue(0.0);
    addAndMakeVisible(panKnob_);
    
    // Mute/Solo buttons
    muteButton_.setButtonText("M");
    muteButton_.onClick = [this] { muted_ = !muted_; };
    addAndMakeVisible(muteButton_);
    
    soloButton_.setButtonText("S");
    soloButton_.onClick = [this] { soloed_ = !soloed_; };
    addAndMakeVisible(soloButton_);
}

void ChannelStrip::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff1a1a1a));
    
    // Draw channel name
    g.setColour(juce::Colours::white);
    g.drawText(name_, 0, 0, getWidth(), 20, 
               juce::Justification::centred);
    
    // Draw VU meter
    drawVUMeter(g);
}
```

#### Estimated Effort
- ScoreEntryPanel: 20-25 hours
- MixerConsolePanel: 25-30 hours
- Testing: 8-10 hours
- Total: 53-65 hours

---

### 6. LearningPanel Features

**Status**: TODO comments  
**File**: `src/ui/theory/LearningPanel.cpp:69, 74`  
**Priority**: HIGH - User experience

#### Implementation Steps

**Step 1: Implement MIDI Playback**
```cpp
void LearningPanel::playExampleButton_.onClick = [this] {
    if (currentConcept_.empty()) {
        return;
    }
    
    // Get MIDI example for concept
    auto midiExample = brain_->getConceptExample(currentConcept_);
    
    if (midiExample) {
        // Play MIDI using JUCE MidiPlayer or similar
        midiPlayer_.loadMidiFile(*midiExample);
        midiPlayer_.play();
        
        playExampleButton_.setButtonText("Stop");
        playExampleButton_.onClick = [this] {
            midiPlayer_.stop();
            playExampleButton_.setButtonText("Play Example");
        };
    }
};
```

**Step 2: Implement Exercise Loading**
```cpp
void LearningPanel::nextExerciseButton_.onClick = [this] {
    if (currentConcept_.empty()) {
        return;
    }
    
    // Get next exercise for concept
    auto exercise = brain_->getNextExercise(currentConcept_);
    
    if (exercise) {
        currentExercise_ = *exercise;
        displayExercise(*exercise);
    } else {
        // No more exercises
        explanationDisplay_.setText("No more exercises available for this concept.");
    }
};

void LearningPanel::displayExercise(const Exercise& exercise) {
    juce::String text;
    text += "Exercise: " + juce::String(exercise.title) + "\n\n";
    text += exercise.description + "\n\n";
    text += "Instructions: " + exercise.instructions;
    
    explanationDisplay_.setText(text);
    
    // Load exercise MIDI if available
    if (!exercise.midiFile.empty()) {
        exerciseMidi_ = juce::File(exercise.midiFile);
    }
}
```

#### Estimated Effort
- MIDI playback: 3-4 hours
- Exercise loading: 2-3 hours
- Testing: 2 hours
- Total: 7-9 hours

---

### 7. ML Inference Processing

**Status**: Placeholder implementations  
**Files**: 
- `src/ml/RTNeuralProcessor.cpp`
- `src/ml/MLBridge.cpp`  
**Priority**: HIGH - AI features

#### Implementation Steps

**Step 1: Implement RTNeural JSON Parsing**
```cpp
bool RTNeuralProcessor::loadModel(const juce::File& jsonFile) {
#ifdef ENABLE_RTNEURAL
    std::ifstream file(jsonFile.getFullPathName().toStdString());
    if (!file.is_open()) {
        return false;
    }
    
    nlohmann::json modelJson;
    file >> modelJson;
    
    // Parse RTNeural model structure
    // Input layer
    if (modelJson.contains("in_size")) {
        inputSize_ = modelJson["in_size"];
    }
    
    // Layers
    if (modelJson.contains("layers")) {
        for (const auto& layer : modelJson["layers"]) {
            // Parse layer type and weights
            parseLayer(layer);
        }
    }
    
    isLoaded_ = true;
    return true;
#else
    return false;
#endif
}
```

**Step 2: Implement process() Method**
```cpp
void RTNeuralProcessor::process(const float* input, float* output, int numSamples) {
#ifdef ENABLE_RTNEURAL
    if (!isModelLoaded()) {
        std::memcpy(output, input, numSamples * sizeof(float));
        return;
    }
    
    // Process through model
    model_.reset();
    for (int i = 0; i < numSamples; ++i) {
        float in = input[i];
        float out = model_.forward(&in);
        output[i] = out;
    }
#else
    std::memcpy(output, input, numSamples * sizeof(float));
#endif
}
```

**Step 3: Implement Async ML Inference**
```cpp
bool MLBridge::processAsync(const std::string& inputJson, 
                           std::function<void(const std::string&)> callback) {
    if (!isInitialized()) {
        return false;
    }
    
    // Submit to thread pool
    threadPool_.addJob([this, inputJson, callback]() {
        std::string result = processSync(inputJson);
        callback(result);
    });
    
    return true;
}
```

#### Estimated Effort
- RTNeural JSON parsing: 6-8 hours
- Process method: 4-5 hours
- Async inference: 4-5 hours
- Testing: 4-5 hours
- Total: 18-23 hours

---

## Summary

### Critical Priority Total: 50-66 hours
- KellyBrain MIDI: 6-9 hours
- MidiIO: 9-12 hours
- OSC System: 15-19 hours
- Groove Engine: 20-26 hours

### High Priority Total: 78-97 hours
- UI Panels: 53-65 hours
- LearningPanel: 7-9 hours
- ML Inference: 18-23 hours

### Grand Total: 128-163 hours

---

## Implementation Order Recommendation

1. **Week 1**: KellyBrain MIDI + MidiIO (15-21 hours)
2. **Week 2**: OSC System (15-19 hours)
3. **Week 3**: Groove Engine Core (20-26 hours)
4. **Week 4**: LearningPanel Features (7-9 hours)
5. **Week 5-7**: UI Panels (53-65 hours)
6. **Week 8**: ML Inference (18-23 hours)

Total: 8 weeks for critical + high priority features

---

## Quick Reference: Key Files and Dependencies

### Files to Modify

| Component | Header | Implementation | Priority |
|-----------|--------|---------------|----------|
| KellyBrain MIDI | `src/engine/KellyBrain.h` | `src/engine/KellyBrain.cpp:228-253` | CRITICAL |
| MidiIO | `src/midi/MidiIO.h` | `src/midi/MidiIO.cpp` | CRITICAL |
| RTMessageQueue | `include/penta/osc/RTMessageQueue.h` | `src/osc/RTMessageQueue.cpp` | CRITICAL |
| OSCClient | `include/penta/osc/OSCClient.h` | `src/osc/OSCClient.cpp` | CRITICAL |
| OSCServer | `include/penta/osc/OSCServer.h` | `src/osc/OSCServer.cpp` | CRITICAL |
| GrooveEngine | `include/penta/groove/GrooveEngine.h` | `src/groove/GrooveEngine.cpp` | CRITICAL |
| ScoreEntryPanel | `src/ui/ScoreEntryPanel.h` | `src/ui/ScoreEntryPanel.cpp` (NEW) | HIGH |
| MixerConsolePanel | `src/ui/MixerConsolePanel.h` | `src/ui/MixerConsolePanel.cpp` (NEW) | HIGH |
| LearningPanel | `src/ui/theory/LearningPanel.h` | `src/ui/theory/LearningPanel.cpp:69,74` | HIGH |
| RTNeuralProcessor | `src/ml/RTNeuralProcessor.h` | `src/ml/RTNeuralProcessor.cpp:35` | HIGH |

### Dependencies to Add

| Dependency | Purpose | CMake Target | Status |
|------------|---------|--------------|--------|
| `readerwriterqueue` | Lock-free queue | `readerwriterqueue` | Need FetchContent |
| JUCE OSC | OSC communication | `juce::juce_osc` | Already available |
| JUCE MIDI | MIDI I/O | `juce::juce_audio_devices` | Already linked |
| JUCE FFT | FFT for onset detection | `juce::juce_dsp` | May need to add |
| RTNeural | Neural network inference | External library | Optional |

### CMake Changes Required

```cmake
# Add to src_penta-core/CMakeLists.txt or root CMakeLists.txt

# 1. Fetch readerwriterqueue
include(FetchContent)
FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
)
FetchContent_MakeAvailable(readerwriterqueue)

# 2. Link JUCE OSC module
target_link_libraries(penta_core PUBLIC
    juce::juce_osc  # ADD THIS
    readerwriterqueue
)

# 3. Link JUCE DSP for FFT (if needed)
target_link_libraries(penta_core PUBLIC
    juce::juce_dsp  # For FFT in GrooveEngine
)
```

### Testing Checklist

- [ ] KellyBrain generates MIDI notes (not empty)
- [ ] MidiIO can enumerate and open devices
- [ ] RTMessageQueue is lock-free (no blocking in audio thread)
- [ ] OSCClient sends messages successfully
- [ ] OSCServer receives messages and queues them
- [ ] GrooveEngine detects tempo accurately (±2 BPM)
- [ ] ScoreEntryPanel renders staff notation
- [ ] MixerConsolePanel displays channel strips
- [ ] LearningPanel plays MIDI examples
- [ ] RTNeuralProcessor loads and processes models

---

## Additional Resources

### Related Documentation
- `docs/DEPENDENCIES.md` - Full dependency list
- `docs/PHASE3_DESIGN.md` - Architecture overview
- `docs/cpp_audio_architecture.md` - C++ audio architecture
- `CLAUDE.md` - Project guide

### External Libraries
- [readerwriterqueue](https://github.com/cameron314/readerwriterqueue) - Lock-free queue
- [JUCE Documentation](https://docs.juce.com/) - JUCE framework docs
- [OSC Specification](https://opensoundcontrol.stanford.edu/) - OSC protocol

### Code Examples
- `examples/integration_example.py` - Python integration examples
- `tests/penta_core/osc_test.cpp` - OSC test code
- `tests/penta_core/groove_test.cpp` - Groove engine tests

