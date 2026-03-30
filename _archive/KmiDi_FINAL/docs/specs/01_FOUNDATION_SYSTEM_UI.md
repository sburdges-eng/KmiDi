# 01. Foundation / System UI Specs

## Overview

These specifications ensure macOS and DAWs don't hate you. System-level compliance prevents crashes, hangs, and user frustration.

## Windowing & Host Compliance Spec

### Standalone App Windowing (AppKit Rules)

**✓ ALLOWED:**
- One main window, native macOS chrome
- Fullscreen + split view support
- Tabbed windows (macOS 10.12+)
- Native title bar with custom controls

**❌ FORBIDDEN:**
- Floating inspectors by default (may be user-optioned)
- Multiple independent windows
- Custom window chrome
- Window management outside AppKit

**Implementation:**
```objc
// AppDelegate.m - Correct pattern
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    self.mainWindow = [[NSWindow alloc] initWithContentRect:rect
                                                  styleMask:NSWindowStyleMaskTitled |
                                                           NSWindowStyleMaskClosable |
                                                           NSWindowStyleMaskMiniaturizable |
                                                           NSWindowStyleMaskResizable
                                                    backing:NSBackingStoreBuffered
                                                      defer:NO];
    // Enable fullscreen
    self.mainWindow.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;
    // Enable split view
    self.mainWindow.collectionBehavior |= NSWindowCollectionBehaviorManaged;
}
```

### Plug-in Windowing (JUCE AudioProcessorEditor)

**✓ ALLOWED:**
- Fixed layout zones, no detachable panels
- Resizing limits enforced
- Host-safe assumptions
- Editor bounds respected

**❌ FORBIDDEN:**
- Dynamic window creation
- Sub-windows or popovers
- Floating panels
- Resize beyond host constraints

**Implementation:**
```cpp
// PluginEditor.cpp - Correct pattern
class DAiWPluginEditor : public juce::AudioProcessorEditor
{
public:
    DAiWPluginEditor (DAiWPluginProcessor& p)
        : AudioProcessorEditor (&p)
    {
        // Fixed size, no resizing
        setSize (800, 600);
        setResizable (false, false);

        // No dynamic window creation
        // All components owned by this editor
    }

    // No sub-windows, no floating panels
    // All interaction contained within editor bounds
};
```

### Focus, DPI, Scaling Behavior

**✓ ALLOWED:**
- Native macOS focus rings
- High-DPI support via AppKit
- JUCE component scaling
- Keyboard navigation

**❌ FORBIDDEN:**
- Custom focus rings
- Manual DPI calculations
- Fixed pixel sizes

**Implementation:**
```objc
// AppKit handles DPI automatically
- (void)viewDidChangeBackingProperties {
    [super viewDidChangeBackingProperties];
    // AppKit provides backingScaleFactor
    // JUCE components scale automatically
}
```

## Input & Interaction Spec

### Mouse / Trackpad (Precise Hit Targets)

**✓ ALLOWED:**
- Standard macOS hit target sizes (44pt minimum)
- JUCE component standard sizes
- Precise selection within musical elements

**❌ FORBIDDEN:**
- Hit targets < 44pt on macOS
- Invisible click areas
- Misleading cursor changes

**Implementation:**
```cpp
// JUCE Component - Correct sizing
class ControlKnob : public juce::Slider
{
public:
    ControlKnob() {
        // Minimum 44pt hit target
        setSize (44, 44);
        // Clear visual feedback
        setMouseCursor (juce::MouseCursor::PointingHandCursor);
    }
};
```

### Scroll Behavior (Zoom Where Expected)

**✓ ALLOWED:**
- Timeline zoom with scroll + modifier
- List scrolling
- Standard macOS scroll direction

**❌ FORBIDDEN:**
- Scroll hijacking
- Non-standard scroll behavior
- Scroll without visual feedback

**Implementation:**
```cpp
// TimelineComponent.cpp
void mouseWheelMove (const juce::MouseEvent& event,
                    const juce::MouseWheelDetails& wheel) override
{
    if (event.mods.isCommandDown()) {
        // Cmd+scroll = zoom (macOS standard)
        zoomTimeline (wheel.deltaY > 0 ? 1.1f : 0.9f);
    } else {
        // Standard scroll = pan
        panTimeline (wheel.deltaX);
    }
}
```

### Keyboard Shortcuts (DAW Muscle Memory)

**Standalone App:**
- Standard macOS shortcuts (Cmd+N, Cmd+W, etc.)
- DAW-style shortcuts (Space = play/pause)

**Plug-in:**
- No shortcuts that hijack host
- Only respond to host-forwarded shortcuts

**❌ FORBIDDEN (Plug-in):**
- Global shortcuts
- Shortcuts that interfere with host
- Custom shortcut schemes

**Implementation:**
```cpp
// Standalone app - OK
bool StandaloneApp::keyPressed (const juce::KeyPress& key)
{
    if (key == juce::KeyPress::spaceKey) {
        togglePlayback(); // Space = play/pause (DAW standard)
        return true;
    }
    return false; // Let OS handle other shortcuts
}

// Plugin - FORBIDDEN
bool PluginEditor::keyPressed (const juce::KeyPress& key)
{
    // NEVER intercept keys in plugin
    // Host manages all shortcuts
    return false;
}
```

### Accessibility (Keyboard Navigable)

**✓ REQUIRED:**
- Tab navigation between controls
- Clear labels on all controls
- Screen reader support
- No color-only meaning

**❌ FORBIDDEN:**
- Controls without labels
- Color-only status indicators
- Non-standard navigation

**Implementation:**
```cpp
// JUCE Component - Correct accessibility
class ParameterControl : public juce::Component
{
public:
    void setParameterName (const juce::String& name) {
        setTitle (name); // Screen reader accessible
        setDescription (name + " control"); // Additional context
    }

    // Tab navigation works automatically in JUCE
    // if component is focusable
};
```

### What Never Intercepts Input (ML Overlays, Inspectors)

**ML Overlays:**
- Never intercept mouse
- Never prevent timeline interaction
- Low opacity, background only

**Inspectors:**
- Read-only displays
- No interactive controls
- Never modal dialogs

**❌ FORBIDDEN:**
```cpp
// WRONG - Overlays intercepting input
class MLOverlay : public juce::Component
{
    bool hitTest (int x, int y) override {
        return true; // WRONG: Intercepts all mouse events
    }
};
```

**✓ CORRECT:**
```cpp
// RIGHT - Overlays never intercept
class MLOverlay : public juce::Component
{
    bool hitTest (int x, int y) override {
        return false; // CORRECT: Transparent to mouse
    }
};
```

## Performance & Frame Budget Spec (NON-NEGOTIABLE)

### No Allocations in Paint

**❌ FORBIDDEN:**
```cpp
// WRONG - Allocation in paint
void paint (juce::Graphics& g) override
{
    juce::String text = "Value: " + juce::String(getValue()); // ALLOCATION!
    g.drawText (text, getLocalBounds(), juce::Justification::centred);
}
```

**✓ CORRECT:**
```cpp
// RIGHT - Pre-allocate and reuse
class MyComponent : public juce::Component
{
    juce::String cachedText;

    void paint (juce::Graphics& g) override
    {
        // Update cache only when needed
        if (valueChanged)
            cachedText = "Value: " + juce::String(getValue());

        g.drawText (cachedText, getLocalBounds(), juce::Justification::centred);
    }
};
```

### No Allocations in Audio Thread

**❌ FORBIDDEN:**
```cpp
// WRONG - Audio thread allocation
void processBlock (juce::AudioBuffer<float>& buffer,
                  juce::MidiBuffer& midiMessages) override
{
    std::vector<float> temp (buffer.getNumSamples()); // ALLOCATION!
    // ... process audio ...
}
```

**✓ CORRECT:**
```cpp
// RIGHT - Pre-allocate buffers
class Processor : public juce::AudioProcessor
{
    juce::AudioBuffer<float> tempBuffer;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        tempBuffer.setSize (getTotalNumOutputChannels(), samplesPerBlock);
    }

    void processBlock (juce::AudioBuffer<float>& buffer,
                      juce::MidiBuffer& midiMessages) override
    {
        // Use pre-allocated tempBuffer
        // No allocations here
    }
};
```

### UI Never Blocks Audio

**✓ REQUIRED:**
- Async UI operations
- Non-blocking parameter updates
- Throttled UI updates

**❌ FORBIDDEN:**
- Synchronous file I/O in UI
- Long operations on UI thread
- UI thread blocking audio thread

**Implementation:**
```cpp
// CORRECT - Async parameter updates
void parameterChanged (const juce::String& parameterID, float newValue)
{
    // Update UI asynchronously
    juce::MessageManager::callAsync ([this, parameterID, newValue]() {
        updateParameterUI (parameterID, newValue);
    });
}
```

### Editor Opens < 100ms

**✓ REQUIRED:**
- Fast editor instantiation
- Lazy component loading
- Cached resources

**❌ FORBIDDEN:**
- Heavy initialization in constructor
- Synchronous file loading
- Complex computations at startup

**Implementation:**
```cpp
// CORRECT - Fast editor creation
class PluginEditor : public juce::AudioProcessorEditor
{
public:
    PluginEditor (AudioProcessor& p) : AudioProcessorEditor (p)
    {
        // Fast: Just create components
        addAndMakeVisible (mainComponent);

        // Lazy: Load complex resources later
        juce::MessageManager::callAsync ([this]() {
            loadComplexResources();
        });

        setSize (800, 600); // Immediate size
    }

private:
    void loadComplexResources() {
        // Load images, complex UI elements here
    }
};
```

### Throttled Redraws

**✓ REQUIRED:**
- Rate-limited updates
- Dirty flag pattern
- Timer-based updates

**❌ FORBIDDEN:**
- Continuous repainting
- Unthrottled animations
- Real-time value updates

**Implementation:**
```cpp
// CORRECT - Throttled updates
class ParameterDisplay : public juce::Component, private juce::Timer
{
    float currentValue = 0.0f;
    float displayValue = 0.0f;
    bool needsUpdate = false;

    void parameterChanged (float newValue) {
        currentValue = newValue;
        needsUpdate = true;
        startTimer (16); // ~60fps max
    }

    void timerCallback() override {
        if (needsUpdate) {
            displayValue = currentValue; // Lerp for smooth animation
            needsUpdate = false;
            repaint();
        }
        stopTimer();
    }
};
```

## Audit Checklist

### Windowing Compliance
- [ ] Standalone: Single main window, AppKit chrome
- [ ] Standalone: Fullscreen + split view support
- [ ] Plug-in: Fixed editor bounds, no dynamic windows
- [ ] Plug-in: Resizing limits enforced
- [ ] Focus: Native focus rings, keyboard navigation
- [ ] DPI: AppKit/JUCE handles scaling automatically

### Input Compliance
- [ ] Hit targets ≥ 44pt on macOS
- [ ] Scroll: Zoom with Cmd+scroll, pan with scroll
- [ ] Keyboard: Standard shortcuts, no host hijacking
- [ ] Accessibility: Tab navigation, screen readers, labels
- [ ] Overlays: Never intercept mouse events

### Performance Compliance
- [ ] No allocations in paint()
- [ ] No allocations in audio thread
- [ ] UI never blocks audio (async operations)
- [ ] Editor opens < 100ms
- [ ] Throttled redraws (≤ 60fps)
- [ ] Lazy loading for complex resources

## Code Examples

### ✅ CORRECT: Fast Plugin Editor
```cpp
class FastPluginEditor : public juce::AudioProcessorEditor
{
public:
    FastPluginEditor (Processor& p) : AudioProcessorEditor (p)
    {
        // Immediate setup
        addAndMakeVisible (knob);
        addAndMakeVisible (label);
        setSize (400, 300); // Immediate size

        // Async complex setup
        juce::MessageManager::callAsync ([this]() {
            setupComplexUI();
        });
    }

private:
    juce::Slider knob;
    juce::Label label;

    void setupComplexUI() {
        // Load presets, complex graphics here
        // After editor is already visible
    }
};
```

### ❌ WRONG: Slow Plugin Editor
```cpp
class SlowPluginEditor : public juce::AudioProcessorEditor
{
public:
    SlowPluginEditor (Processor& p) : AudioProcessorEditor (p)
    {
        // WRONG: Synchronous file loading
        loadAllPresets(); // Blocks UI thread!

        // WRONG: Heavy computation at startup
        generateWaveforms(); // CPU intensive!

        setSize (400, 300);
    }
};
```

## Non-Compliance Fixes

### If Allocation in Paint Found:
1. Move string creation to cached member variables
2. Update cache only when value changes
3. Use pre-allocated buffers

### If Audio Thread Allocation Found:
1. Pre-allocate all buffers in prepareToPlay()
2. Use stack-based or pre-allocated temporary storage
3. Never use new/delete in processBlock()

### If UI Blocking Audio Found:
1. Move file I/O to background threads
2. Use async parameter updates
3. Implement progress indicators for long operations

### If Slow Editor Startup Found:
1. Move heavy initialization to callAsync blocks
2. Use lazy loading for complex components
3. Pre-calculate expensive operations

### If Unthrottled Redraws Found:
1. Implement dirty flag pattern
2. Use Timer-based updates (16ms = ~60fps)
3. Batch multiple updates into single repaint()