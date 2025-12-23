# 4-Cycle Agent Prompt: Incremental TODO Fix Implementation

**Token Limit**: 270,000 tokens per cycle  
**Total Cycles**: 4  
**Goal**: Incrementally implement all critical and high-priority TODOs

---

## Context

You are working on the iDAW (Intelligent Digital Audio Workstation) project. There are **40+ incomplete features** identified across:
- UI Components (13 items)
- OSC System (4 items)
- Groove Engine (4 items)
- MIDI/Audio (3 items)
- Harmony Engine (2 items)
- ML/AI (5 items)
- Bridges (3 items)
- Biometric (2 items)

**Reference Documents**:
- `docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md` - Detailed implementation plans
- `docs/DEPENDENCY_SETUP_GUIDE.md` - Dependency setup instructions
- `CLAUDE.md` - Project architecture guide

---

## Cycle Breakdown

### **Cycle 1: Critical Foundation (Weeks 1-2)**
**Focus**: Core MIDI generation and I/O  
**Target**: 6-9 hours of work  
**Priority**: CRITICAL - Blocks core functionality

**Tasks**:
1. ✅ **KellyBrain MIDI Generation** (`src/engine/KellyBrain.cpp:228-253`)
   - Integrate `MidiGenerator` into `KellyBrain`
   - Implement `generateMidi()` method
   - Convert `KellyTypesIntentResult` to `IntentResult`
   - Test MIDI note generation

2. ✅ **MidiIO Implementation** (`src/midi/MidiIO.cpp`)
   - Implement JUCE MIDI device enumeration
   - Implement `MidiInput::open()`, `start()`, `stop()`
   - Implement `MidiOutput::open()`, `send()`
   - Add error handling and logging

**Success Criteria**:
- KellyBrain generates non-empty MIDI notes
- MidiIO can enumerate and open devices
- All tests pass

**Files to Modify**:
- `src/engine/KellyBrain.h` / `.cpp`
- `src/midi/MidiIO.h` / `.cpp`
- `CMakeLists.txt` (if needed for JUCE MIDI)

---

### **Cycle 2: OSC Communication System (Week 3)**
**Focus**: Real-time OSC messaging  
**Target**: 16-20 hours of work  
**Priority**: CRITICAL - Required for DAW communication

**Tasks**:
1. ✅ **Setup Dependencies**
   - Add `readerwriterqueue` via CMake FetchContent
   - Link `juce::juce_osc` module
   - Update `src_penta-core/CMakeLists.txt`

2. ✅ **RTMessageQueue Implementation** (`src/osc/RTMessageQueue.cpp`)
   - Implement lock-free queue using `readerwriterqueue`
   - Add `push()`, `pop()`, `isEmpty()`, `size()` methods
   - Ensure RT-safe (no blocking)

3. ✅ **OSCClient Implementation** (`src/osc/OSCClient.cpp`)
   - Implement JUCE OSC sender
   - Add `send()`, `sendFloat()`, `sendInt()`, `sendString()`
   - Handle connection/disconnection

4. ✅ **OSCServer Implementation** (`src/osc/OSCServer.cpp`)
   - Implement JUCE OSC receiver
   - Add message listener that pushes to RT-safe queue
   - Implement `start()`, `stop()`, `getMessageQueue()`

**Success Criteria**:
- RTMessageQueue is lock-free (no blocking in audio thread)
- OSCClient sends messages successfully
- OSCServer receives messages and queues them
- Stress test: 1000+ messages/sec without blocking

**Files to Modify**:
- `src_penta-core/CMakeLists.txt` (dependencies)
- `src/osc/RTMessageQueue.cpp`
- `src/osc/OSCClient.cpp`
- `src/osc/OSCServer.cpp`
- `include/penta/osc/RTMessageQueue.h` (if needed)
- `include/penta/osc/OSCClient.h` (if needed)
- `include/penta/osc/OSCServer.h` (if needed)

---

### **Cycle 3: Groove Engine Core (Week 4)**
**Focus**: Rhythm analysis and tempo detection  
**Target**: 20-26 hours of work  
**Priority**: CRITICAL - Core rhythm analysis

**Tasks**:
1. ✅ **Setup FFT Dependency**
   - Link `juce::juce_dsp` for FFT (if not already linked)
   - Update CMakeLists.txt

2. ✅ **Tempo Estimation** (`src/groove/GrooveEngine.cpp`)
   - Implement `updateTempoEstimate()` using autocorrelation
   - Calculate inter-onset intervals (IOIs)
   - Find best tempo match (60-200 BPM range)

3. ✅ **Time Signature Detection** (`src/groove/GrooveEngine.cpp`)
   - Implement `detectTimeSignature()`
   - Analyze beat patterns and downbeats
   - Detect 4/4, 3/4, 2/4 time signatures

4. ✅ **Swing Analysis** (`src/groove/GrooveEngine.cpp`)
   - Implement `analyzeSwing()`
   - Measure timing of off-beat notes
   - Calculate swing ratio (0.0 = straight, 1.0 = max swing)

5. ✅ **Quantization** (`src/groove/RhythmQuantizer.cpp`)
   - Implement rhythm quantization with swing
   - Apply quantization strength parameter

**Success Criteria**:
- GrooveEngine detects tempo accurately (±2 BPM)
- Time signature detection works for 4/4, 3/4
- Swing analysis produces reasonable values
- Quantization applies correctly

**Files to Modify**:
- `src_penta-core/CMakeLists.txt` (juce_dsp)
- `src/groove/GrooveEngine.cpp`
- `src/groove/TempoEstimator.cpp`
- `src/groove/RhythmQuantizer.cpp`
- `src/groove/OnsetDetector.cpp` (if needed)

---

### **Cycle 4: High-Priority Features (Weeks 5-8)**
**Focus**: UI panels and ML inference  
**Target**: 78-97 hours of work  
**Priority**: HIGH - Enhances user experience

**Tasks**:

**Part A: UI Components (53-65 hours)**
1. ✅ **ScoreEntryPanel** (`src/ui/ScoreEntryPanel.cpp` - NEW FILE)
   - Implement staff rendering
   - Implement note entry (mouse/keyboard)
   - Implement key/time signature display
   - Add clef rendering

2. ✅ **MixerConsolePanel** (`src/ui/MixerConsolePanel.cpp` - NEW FILE)
   - Implement channel strips
   - Implement faders and pan controls
   - Implement VU meters
   - Add routing matrix

3. ✅ **LearningPanel Features** (`src/ui/theory/LearningPanel.cpp:69,74`)
   - Implement MIDI example playback
   - Implement exercise loading and display

**Part B: ML Inference (18-23 hours)**
4. ✅ **RTNeural JSON Parsing** (`src/ml/RTNeuralProcessor.cpp:35`)
   - Implement proper JSON model loading
   - Parse RTNeural model format

5. ✅ **RTNeural Process Method** (`src/ml/RTNeuralProcessor.h:54`)
   - Implement `process()` method
   - Process audio through model

6. ✅ **MLBridge Async Inference** (`src/ml/MLBridge.cpp:362,366`)
   - Implement async ML inference with thread pool
   - Implement intent result processing

**Success Criteria**:
- ScoreEntryPanel renders staff notation
- MixerConsolePanel displays channel strips
- LearningPanel plays MIDI examples
- RTNeuralProcessor loads and processes models
- MLBridge handles async inference

**Files to Modify**:
- `src/ui/ScoreEntryPanel.cpp` (NEW)
- `src/ui/MixerConsolePanel.cpp` (NEW)
- `src/ui/theory/LearningPanel.cpp`
- `src/ml/RTNeuralProcessor.cpp`
- `src/ml/MLBridge.cpp`

---

## Implementation Guidelines

### For Each Cycle:

1. **Read First**:
   - Read the relevant section in `docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md`
   - Read `docs/DEPENDENCY_SETUP_GUIDE.md` if adding dependencies
   - Review existing code structure

2. **Plan**:
   - Identify all files to modify/create
   - Check dependencies and CMake configuration
   - Review existing interfaces and type definitions

3. **Implement**:
   - Follow code examples in implementation plans
   - Use JUCE APIs where specified (not external libraries)
   - Ensure RT-safety for audio thread code
   - Add error handling and logging

4. **Test**:
   - Write unit tests where possible
   - Test RT-safety (no blocking in audio callbacks)
   - Verify integration with existing code
   - Check for memory leaks

5. **Document**:
   - Update inline comments
   - Document any deviations from plan
   - Note any issues or limitations

### Code Style:
- **C++**: C++20 standard, RT-safe where needed
- **Naming**: Follow existing conventions (PascalCase classes, camelCase methods)
- **Error Handling**: Log errors, return false/empty on failure (don't throw in RT code)
- **Memory**: Use smart pointers, no raw `new`/`delete`

### RT-Safety Rules:
- Mark audio callbacks `noexcept`
- No memory allocation in audio thread
- Use lock-free data structures for thread communication
- Pre-allocated memory pools where needed

---

## Cycle-Specific Instructions

### **CYCLE 1 PROMPT**:
```
You are implementing Cycle 1: Critical Foundation (KellyBrain MIDI Generation + MidiIO).

Your task:
1. Review src/engine/KellyBrain.cpp:228-253 and integrate MidiGenerator
2. Review src/midi/MidiIO.cpp and implement JUCE MIDI I/O
3. Follow the detailed plan in docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md sections 1-2
4. Ensure all code compiles and tests pass

Token limit: 270,000 tokens
Focus: Get MIDI generation working end-to-end
```

### **CYCLE 2 PROMPT**:
```
You are implementing Cycle 2: OSC Communication System.

Your task:
1. Setup dependencies (readerwriterqueue, juce_osc) per docs/DEPENDENCY_SETUP_GUIDE.md
2. Implement RTMessageQueue, OSCClient, OSCServer per docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md section 3
3. Use JUCE OSC (not oscpack) - it's already available
4. Ensure RT-safety (lock-free queue, no blocking)

Token limit: 270,000 tokens
Focus: Get OSC messaging working with RT-safe queue
```

### **CYCLE 3 PROMPT**:
```
You are implementing Cycle 3: Groove Engine Core.

Your task:
1. Link juce_dsp for FFT if needed
2. Implement tempo estimation, time signature detection, swing analysis per docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md section 4
3. Implement RhythmQuantizer with swing support
4. Test with various audio inputs

Token limit: 270,000 tokens
Focus: Get rhythm analysis working accurately
```

### **CYCLE 4 PROMPT**:
```
You are implementing Cycle 4: High-Priority Features (UI + ML).

Your task:
1. Implement ScoreEntryPanel and MixerConsolePanel (NEW files) per docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md section 5
2. Complete LearningPanel features per section 6
3. Implement RTNeural and MLBridge features per section 7
4. Ensure UI components integrate with existing code

Token limit: 270,000 tokens
Focus: Complete UI panels and ML inference
```

---

## Success Metrics

After all 4 cycles:

- ✅ **40+ TODOs completed**
- ✅ **All critical features working**
- ✅ **All high-priority features working**
- ✅ **RT-safety verified** (no blocking in audio thread)
- ✅ **Tests passing**
- ✅ **No memory leaks**
- ✅ **Code compiles cleanly**

---

## Notes

- **Don't skip steps**: Each cycle builds on previous work
- **Test incrementally**: Don't wait until the end to test
- **Ask for clarification**: If implementation plan is unclear
- **Document deviations**: Note any changes from the plan
- **Preserve existing code**: Only modify what's necessary

---

## Quick Reference

**Key Files**:
- Implementation Plans: `docs/IMPLEMENTATION_PLANS_Critical_High_Priority.md`
- Dependency Guide: `docs/DEPENDENCY_SETUP_GUIDE.md`
- Project Guide: `CLAUDE.md`

**Key Dependencies**:
- JUCE Framework: `external/JUCE/` (already available)
- readerwriterqueue: Via CMake FetchContent (Cycle 2)
- JUCE Modules: `juce_osc`, `juce_dsp`, `juce_audio_devices`

**CMake Files**:
- Root: `CMakeLists.txt`
- Penta-Core: `src_penta-core/CMakeLists.txt`

Good luck! 🚀

