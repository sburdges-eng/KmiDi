# UI Boundary Rules

**Date:** January 18, 2026  
**Purpose:** Define what UI can and cannot access from DSP/audio systems  
**Reference:** Architectural Boundary Compliance Report, Layout & Navigation Specs

## Core Principle

**"UI reads from DSP state. UI writes intent into parameters. Nothing else."**

Audio buffers never touch UI. UI threads never touch audio buffers.

## Allowed UI Operations

### ✅ What UI Can Read

1. **Parameter Snapshots**
   ```typescript
   // UI can read current parameter values
   const currentCutoff = getParameterSnapshot('cutoff');
   const currentResonance = getParameterSnapshot('resonance');
   ```

2. **State Snapshots**
   ```typescript
   // UI can read non-real-time state
   const isPlaying = getTransportState();
   const currentTime = getPlaybackPosition();
   ```

3. **Meter Values**
   ```typescript
   // UI can read level meters (pre-computed)
   const peakLevel = getMeterValue('peak');
   const rmsLevel = getMeterValue('rms');
   ```

4. **Analysis Data (Read-Only)**
   ```typescript
   // UI can read analysis results (computed offline)
   const frequencySpectrum = getSpectrumAnalysis();
   const chordDetection = getChordAnalysis();
   ```

### ✅ What UI Can Write

1. **User Intent → Parameters**
   ```typescript
   // UI writes user intent to parameters
   setParameter('cutoff', userSelectedValue);
   setParameter('resonance', sliderValue);
   ```

2. **User Actions → Commands**
   ```typescript
   // UI triggers actions (non-real-time)
   triggerGenerateMusic();
   loadPreset('preset_name');
   ```

## Forbidden UI Operations

### ❌ What UI Cannot Access

1. **Audio Buffers**
   ```typescript
   // WRONG: UI accessing audio buffers
   const audioData = getAudioBuffer();  // FORBIDDEN
   ```

2. **Real-Time Data Structures**
   ```typescript
   // WRONG: UI accessing real-time structures
   const voiceState = getVoiceState();  // FORBIDDEN
   ```

3. **Audio Thread State**
   ```typescript
   // WRONG: UI reading audio thread state
   const processingState = getAudioThreadState();  // FORBIDDEN
   ```

4. **Direct DSP Calls**
   ```typescript
   // WRONG: UI calling DSP directly
   dspEngine.processBlock(...);  // FORBIDDEN
   ```

## UI Layer Separation

### Standalone App UI (macOS)

**Technology:** Swift + SwiftUI (AppKit where SwiftUI falls apart)

**Responsibilities:**
- Native macOS menus
- File dialogs
- Project management
- Advanced configuration
- Visualizations (non-real-time)
- Long-running tasks

**Example:**
```swift
// apps/macOS/MainWindow.swift
import SwiftUI
import AppKit

struct MainWindow: View {
    @StateObject var audioEngine: AudioEngine
    
    var body: some View {
        VStack {
            // UI reads parameter snapshots
            ParameterDisplay(params: audioEngine.parameterSnapshot)
            
            // UI writes user intent
            ParameterControls(onChange: { param, value in
                audioEngine.setParameter(param, value)
            })
        }
    }
}
```

### Plugin UI (JUCE)

**Technology:** JUCE AudioProcessorEditor

**Responsibilities:**
- Minimal, focused controls
- Parameter automation
- Host-safe UI
- Deterministic layout
- No file dialogs
- No background tasks

**Example:**
```cpp
// plugins/PluginEditor.cpp
class PluginEditor : public juce::AudioProcessorEditor {
    void paint(juce::Graphics& g) override {
        // UI reads parameter snapshot (not live audio)
        auto params = processor.getParameterSnapshot();
        
        // Display parameter values
        g.drawText("Cutoff: " + juce::String(params.cutoff), ...);
    }
    
    void sliderValueChanged(juce::Slider* slider) override {
        // UI writes user intent to parameters
        if (slider == &cutoffSlider) {
            processor.setParameter("cutoff", slider->getValue());
        }
    }
};
```

### Web/React UI (Current Implementation)

**Technology:** React + Tauri

**Responsibilities:**
- Therapeutic interface (Side B)
- Emotion selection
- Intent input
- Documentation viewing
- Non-real-time operations

**Current Status:**
- ✅ Correctly uses Tauri bridge for API calls
- ✅ Does not access audio buffers
- ⚠️ Should be replaced with native macOS UI for app shell

## Data Flow Patterns

### Correct Pattern: UI → Parameters → DSP

```
[ UI Layer ]
   ↓ user intent
[ Parameter Layer ]
   ↓ atomic update
[ DSP Core ]
   ↓ audio processing
[ Audio Output ]
```

### Correct Pattern: DSP → Metrics → UI

```
[ DSP Core ]
   ↓ compute metrics
[ Meter/Analysis Layer ]
   ↓ snapshot (non-real-time)
[ UI Layer ]
   ↓ display
```

### ❌ Wrong Pattern: UI → Audio Buffers

```
[ UI Layer ]
   ↓ direct access
[ Audio Buffers ]  // FORBIDDEN
```

## Implementation Rules

### Rule 1: Snapshot Pattern

UI always reads snapshots, never live data:

```cpp
// CORRECT: Snapshot mechanism
class ParameterSnapshot {
    Parameters current;
    std::atomic<bool> updated{false};
    
    void update(const Parameters& newParams) {
        current = newParams;  // Copy (small struct)
        updated.store(true);
    }
    
    Parameters getSnapshot() const {
        return current;  // Read copy
    }
};

// UI reads snapshot
auto params = snapshot.getSnapshot();  // Safe, non-blocking
```

### Rule 2: Throttled Updates

UI updates must be throttled:

```typescript
// CORRECT: Throttled parameter display
class ParameterDisplay {
    private updateTimer: number = 0;
    
    update(value: number) {
        // Throttle to 10 updates/second max
        if (Date.now() - this.updateTimer > 100) {
            this.displayValue(value);
            this.updateTimer = Date.now();
        }
    }
}
```

### Rule 3: No Blocking Operations

UI never blocks audio thread:

```typescript
// CORRECT: Async parameter update
async function updateParameter(param: string, value: number) {
    // Non-blocking async call
    await invoke('set_parameter', { param, value });
}

// WRONG: Synchronous blocking call
function updateParameter(param: string, value: number) {
    // Blocks UI thread, may affect audio
    setParameterSync(param, value);  // FORBIDDEN
}
```

## UI Component Guidelines

### ✅ Allowed UI Components

- Parameter sliders/knobs (write intent)
- Meters/displays (read snapshots)
- Buttons (trigger actions)
- Text inputs (user intent)
- Visualizations (read analysis data)

### ❌ Forbidden UI Components

- Direct audio waveform display (would require buffer access)
- Real-time spectrum analyzer (would require buffer access)
- Audio thread monitors (would require thread access)

### Alternative: Pre-computed Visualizations

```typescript
// CORRECT: UI displays pre-computed analysis
function SpectrumDisplay() {
    // Analysis computed offline, UI displays result
    const spectrum = useSpectrumAnalysis();  // Pre-computed
    
    return <SpectrumChart data={spectrum} />;
}

// WRONG: UI computes from live audio
function SpectrumDisplay() {
    const audioBuffer = getLiveAudioBuffer();  // FORBIDDEN
    const spectrum = computeFFT(audioBuffer);  // FORBIDDEN
    return <SpectrumChart data={spectrum} />;
}
```

## Testing UI Boundaries

### Test 1: No Audio Buffer Access

```typescript
// Test: UI should not have access to audio buffers
describe('UI Boundary Tests', () => {
    it('should not expose audio buffers to UI', () => {
        // This should not exist
        expect(() => getAudioBuffer()).toThrow();
    });
});
```

### Test 2: Parameter Snapshot Only

```typescript
// Test: UI can only read parameter snapshots
it('should only read parameter snapshots', () => {
    const params = getParameterSnapshot();
    expect(params).toBeDefined();
    expect(typeof params.cutoff).toBe('number');
    
    // Should not have direct DSP access
    expect(() => dspEngine.getState()).toThrow();
});
```

### Test 3: Async Updates

```typescript
// Test: UI updates are async and non-blocking
it('should update parameters asynchronously', async () => {
    const start = Date.now();
    await setParameter('cutoff', 1000);
    const duration = Date.now() - start;
    
    // Should be fast, non-blocking
    expect(duration).toBeLessThan(10);  // ms
});
```

## Migration Checklist

If existing UI violates boundaries:

- [ ] Remove all audio buffer access from UI
- [ ] Replace direct DSP calls with parameter updates
- [ ] Implement snapshot mechanism for state reading
- [ ] Add throttling to all real-time displays
- [ ] Verify UI updates are async and non-blocking
- [ ] Test: UI should work even if audio thread is blocked

## References

- `ARCHITECTURAL_BOUNDARY_COMPLIANCE_REPORT.md` - Full boundary analysis
- `02_LAYOUT_NAVIGATION.md` - Layout specifications
- `04_CORE_MUSICAL_UI.md` - Musical UI patterns