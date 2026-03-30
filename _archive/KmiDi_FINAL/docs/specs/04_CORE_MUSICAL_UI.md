# 04. Core Musical UI Specs

## Overview

This is where credibility lives or dies. Musical interfaces must feel professional and trustworthy.

## Timeline UI Spec (Standalone Only)

### Grid Rules

**✓ REQUIRED:**
- Quantized to musical divisions
- Visual grid lines (subtle)
- Snap-to-grid behavior
- Time signature awareness

**❌ FORBIDDEN:**
- Free-form placement without grid
- Invisible grid
- Grid that fights user intent

**Implementation:**
```cpp
// TimelineGrid.h - Musical grid system
class TimelineGrid
{
public:
    enum GridDivision { Whole, Half, Quarter, Eighth, Sixteenth, ThirtySecond };

    void setTimeSignature (int numerator, int denominator) {
        beatsPerBar = numerator;
        beatUnit = denominator;
        updateGridLines();
    }

    double snapToGrid (double timePosition) {
        // Snap to nearest grid division
        double gridSize = getGridSizeInSeconds();
        return std::round (timePosition / gridSize) * gridSize;
    }

private:
    int beatsPerBar = 4;
    int beatUnit = 4; // Quarter notes
    double tempoBPM = 120.0;

    double getGridSizeInSeconds() {
        // Sixteenth notes at 120 BPM = 0.125 seconds
        return (60.0 / tempoBPM) / 4.0; // Quarter note duration / 4
    }
};
```

### Zoom Behavior

**✓ REQUIRED:**
- Smooth zoom with mouse wheel
- Horizontal and vertical zoom
- Zoom center follows mouse
- Min/max zoom limits

**❌ FORBIDDEN:**
- Jerky zoom
- Zoom center not following cursor
- No zoom limits

**Implementation:**
```cpp
// TimelineZoom.h - Smooth zoom behavior
class TimelineZoom
{
public:
    void zoomAtPoint (double zoomFactor, juce::Point<double> centerPoint) {
        // Calculate new zoom level
        double newZoom = currentZoom * zoomFactor;
        newZoom = juce::jlimit (minZoom, maxZoom, newZoom);

        // Adjust view so centerPoint stays fixed
        double zoomRatio = newZoom / currentZoom;
        viewOffset = centerPoint - (centerPoint - viewOffset) * zoomRatio;

        currentZoom = newZoom;
        updateViewport();
    }

private:
    double currentZoom = 1.0;
    double minZoom = 0.1;  // 10% - very zoomed out
    double maxZoom = 10.0; // 1000% - very zoomed in
    juce::Point<double> viewOffset;
};
```

### Track Scaling

**✓ REQUIRED:**
- Individual track heights
- Auto-fit content
- Minimum/maximum heights
- Persistent per track

**❌ FORBIDDEN:**
- Fixed track heights
- No content-based scaling
- Heights that don't persist

**Implementation:**
```cpp
// TrackScaling.h - Intelligent track heights
class TrackScaling
{
public:
    void setTrackHeight (int trackIndex, int height) {
        heights[trackIndex] = juce::jlimit (minHeight, maxHeight, height);
        saveTrackHeights();
    }

    void autoFitTrack (int trackIndex) {
        // Calculate height based on content
        int contentHeight = calculateContentHeight (trackIndex);
        int padding = Spacing::large() * 2; // 32pt total padding
        setTrackHeight (trackIndex, contentHeight + padding);
    }

private:
    std::map<int, int> heights;
    int minHeight = 60;  // Minimum usable height
    int maxHeight = 200; // Maximum reasonable height

    int calculateContentHeight (int trackIndex) {
        // Calculate based on clips, automation, etc.
        return 40; // Placeholder
    }
};
```

### Playhead Behavior

**✓ REQUIRED:**
- Smooth animation during playback
- Precise positioning
- Visual prominence
- No animation when stopped

**❌ FORBIDDEN:**
- Jerky playhead movement
- Invisible playhead
- Animated playhead when not playing

**Implementation:**
```cpp
// PlayheadComponent.h - Smooth playhead
class PlayheadComponent : public juce::Component
{
public:
    void setPosition (double timeInSeconds, bool isPlaying) {
        targetPosition = timeInSeconds;
        this->isPlaying = isPlaying;

        if (!isPlaying) {
            // Snap to position when stopped
            currentPosition = targetPosition;
            repaint();
        }
        // When playing, animation handled in timerCallback
    }

    void paint (juce::Graphics& g) override {
        // Draw playhead line
        float x = timeToX (currentPosition);
        g.setColour (juce::Colours::red);
        g.drawLine (x, 0, x, getHeight(), 2.0f);

        // Draw playhead triangle at top
        juce::Path triangle;
        triangle.addTriangle (x - 6, 0, x + 6, 0, x, 12);
        g.fillPath (triangle);
    }

private:
    double currentPosition = 0.0;
    double targetPosition = 0.0;
    bool isPlaying = false;

    void timerCallback() override {
        if (isPlaying) {
            // Smooth interpolation to target
            double delta = targetPosition - currentPosition;
            currentPosition += delta * 0.1; // 10% interpolation
            repaint();
        }
    }
};
```

### What Never Animates

**❌ FORBIDDEN ANIMATIONS:**
- Playhead when transport stopped
- Grid lines
- Track boundaries
- Static UI elements

**✓ ALLOWED ANIMATIONS:**
- Playhead during playback (smooth)
- Hover states (fast)
- Loading indicators (subtle)
- ML overlays (fade in/out only)

**Implementation:**
```cpp
// AnimationRules.h - What can animate
class AnimationRules
{
public:
    static bool canAnimate (UIElement element, AnimationType type) {
        switch (element) {
            case Playhead:
                return type == SmoothMovement && isPlaying;
            case HoverState:
                return type == FastFade && duration < 0.1;
            case LoadingSpinner:
                return type == Rotate && subtle;
            case GridLines:
                return false; // NEVER animate grid
            case TrackBoundaries:
                return false; // NEVER animate boundaries
            default:
                return false; // Default: no animation
        }
    }

private:
    enum UIElement { Playhead, HoverState, LoadingSpinner, GridLines, TrackBoundaries };
    enum AnimationType { SmoothMovement, FastFade, Rotate };
};
```

## JUCE Embedding Spec

### AppKit ↔ JUCE Boundary

**✓ REQUIRED:**
- Clear ownership boundaries
- Proper coordinate system conversion
- Event forwarding rules
- Memory management across boundaries

**❌ FORBIDDEN:**
- JUCE creating AppKit views
- AppKit managing JUCE components
- Coordinate system confusion

**Implementation:**
```objc
// JUCEToAppKitBridge.h - Clean boundary
@interface JUCEToAppKitBridge : NSView

@property (nonatomic, strong) JUCE_NSView *juceView;

- (instancetype)initWithJUCEComponent:(juce::Component*)component;

// Convert coordinates properly
- (NSPoint)convertPointFromJUCE:(juce::Point<float>)jucePoint;
- (juce::Point<float>)convertPointToJUCE:(NSPoint)nsPoint;

// Forward events correctly
- (void)mouseDown:(NSEvent *)event;
- (void)keyDown:(NSEvent *)event;

@end
```

### Resize Handling

**✓ REQUIRED:**
- Synchronous resize propagation
- No layout during resize
- Deferred expensive operations
- Smooth resize experience

**❌ FORBIDDEN:**
- Expensive operations during resize
- Layout calculations in resize callback
- Resize-caused audio glitches

**Implementation:**
```cpp
// ResizeManager.h - Smooth resize handling
class ResizeManager : public juce::Component::SafePointer<juce::Component>
{
public:
    void componentBeingResized (juce::Component& component) override {
        // Fast: Just update bounds, no expensive operations
        updateBoundsOnly();

        // Defer expensive operations
        juce::MessageManager::callAsync ([this, &component]() {
            performExpensiveLayout (component);
        });
    }

private:
    void updateBoundsOnly() {
        // Just update positions, no drawing or complex calculations
    }

    void performExpensiveLayout (juce::Component& component) {
        // Redraw waveforms, recalculate layouts, etc.
        component.repaint();
    }
};
```

### Lifetime & Ownership Rules

**✓ REQUIRED:**
- Clear ownership hierarchy
- Proper destruction order
- No circular references
- RAII compliance

**❌ FORBIDDEN:**
- JUCE components owning AppKit objects
- AppKit objects owning JUCE components
- Improper cleanup

**Implementation:**
```cpp
// LifetimeManager.h - Proper ownership
class LifetimeManager
{
public:
    LifetimeManager() {
        // JUCE components owned by JUCE
        juceComponent = std::make_unique<MyJUCEComponent>();

        // AppKit views owned by AppKit
        appKitView = [[MyAppKitView alloc] init];

        // Bridge owns nothing, just facilitates communication
        bridge = std::make_unique<JUCEToAppKitBridge>();
    }

    ~LifetimeManager() {
        // Clean destruction order
        bridge.reset();      // Break connections first
        juceComponent.reset(); // Then JUCE objects
        // AppKit objects cleaned up by ARC
    }

private:
    std::unique_ptr<MyJUCEComponent> juceComponent;
    __strong MyAppKitView* appKitView; // ARC managed
    std::unique_ptr<JUCEToAppKitBridge> bridge;
};
```

### What JUCE Is Forbidden to Create

**❌ FORBIDDEN JUCE CREATIONS:**
- NSWindow objects
- NSView hierarchies
- NSToolbar items
- NSMenu items
- Any AppKit UI elements

**✓ ALLOWED JUCE CREATIONS:**
- juce::Component hierarchies
- juce::AudioProcessorEditor
- JUCE drawing operations
- JUCE event handling

**Implementation:**
```cpp
// Forbidden patterns - DO NOT DO THIS
class BadJUCEComponent : public juce::Component
{
    void createAppKitStuff() {
        // WRONG: JUCE should never create AppKit objects
        NSWindow* window = [[NSWindow alloc] init]; // FORBIDDEN!

        // WRONG: JUCE should never manage AppKit hierarchy
        [window setContentView:someNSView]; // FORBIDDEN!
    }
};

// Correct pattern
class GoodJUCEComponent : public juce::Component
{
    // JUCE only manages JUCE components
    // AppKit is handled by AppKit code
    // Communication through clean bridge
};
```

## Plugin Editor Layout Spec

### Fixed Regions Architecture

**✓ REQUIRED:**
- Header region (top, fixed height)
- Parameter region (middle, scrollable)
- Footer region (bottom, fixed height)
- No overlapping regions
- Clear visual separation

**❌ FORBIDDEN:**
- Dynamic region creation
- Overlapping panels
- No clear boundaries

**Implementation:**
```cpp
// PluginEditorLayout.h - Fixed regions
class PluginEditorLayout : public juce::Component
{
public:
    enum Region { Header, Parameters, Footer };

    juce::Rectangle<int> getRegionBounds (Region region) const {
        switch (region) {
            case Header:     return {0, 0, getWidth(), 60};
            case Parameters: return {0, 60, getWidth(), getHeight() - 120};
            case Footer:     return {0, getHeight() - 60, getWidth(), 60};
        }
        return {};
    }

    void layoutRegions() {
        headerComponent->setBounds (getRegionBounds (Header));
        parameterComponent->setBounds (getRegionBounds (Parameters));
        footerComponent->setBounds (getRegionBounds (Footer));
    }

private:
    std::unique_ptr<HeaderComponent> headerComponent;
    std::unique_ptr<ParameterComponent> parameterComponent;
    std::unique_ptr<FooterComponent> footerComponent;
};
```

### Resizing Limits

**✓ REQUIRED:**
- Minimum size: 600x400 (readable)
- Maximum size: Host-determined
- Aspect ratio constraints
- Smooth resize behavior

**❌ FORBIDDEN:**
- No minimum size
- Excessive maximum sizes
- Jerky resize behavior

**Implementation:**
```cpp
// PluginEditor.h - Proper resizing
class DAiWPluginEditor : public juce::AudioProcessorEditor
{
public:
    DAiWPluginEditor (AudioProcessor& p) : AudioProcessorEditor (p)
    {
        // Set initial size
        setSize (800, 600);

        // Enable resizing with constraints
        setResizable (true, true);
        setResizeLimits (600, 400, 2000, 1500); // Reasonable limits
    }

    void resized() override {
        // Smooth resize: just update bounds
        layoutManager.layoutRegions();
    }

private:
    PluginEditorLayout layoutManager;
};
```

### Host-Safe Assumptions

**✓ REQUIRED:**
- No assumptions about host window size
- No modal dialogs
- No file dialogs
- No external process spawning

**❌ FORBIDDEN:**
- Assuming minimum window size
- Modal operations
- File system access
- External command execution

**Implementation:**
```cpp
// HostSafeOperations.h - Plugin-safe operations only
class HostSafeOperations
{
public:
    // SAFE: Update parameters
    void updateParameter (int parameterIndex, float value) {
        processor.setParameter (parameterIndex, value);
    }

    // SAFE: Request UI update
    void triggerUIUpdate() {
        repaint();
    }

    // FORBIDDEN: Show file dialog
    // void showFileDialog() { /* FORBIDDEN */ }

    // FORBIDDEN: Spawn external process
    // void runExternalCommand() { /* FORBIDDEN */ }

    // FORBIDDEN: Show modal dialog
    // void showModal() { /* FORBIDDEN */ }
};
```

### What the Plug-in Is Not Allowed to Control

**❌ NOT ALLOWED TO CONTROL:**
- Host transport (play/stop/rewind)
- Host tempo
- Host time signature
- Host window management
- Host audio settings
- Other plug-ins

**✓ ALLOWED TO CONTROL:**
- Own parameters
- Own UI appearance
- Own audio processing
- Own state persistence

**Implementation:**
```cpp
// PluginProcessor.h - What plugin CAN control
class DAiWPluginProcessor : public juce::AudioProcessor
{
public:
    // CAN control: Own parameters
    void setParameter (int parameterIndex, float newValue) override {
        parameters.setParameter (parameterIndex, newValue);
    }

    // CAN control: Own processing
    void processBlock (AudioBuffer<float>& buffer, MidiBuffer& midi) override {
        // Process own audio
    }

    // CAN control: Own state
    void getStateInformation (MemoryBlock& destData) override {
        // Save own state
    }

    // CANNOT control: Host transport
    // void play() { /* FORBIDDEN */ }

    // CANNOT control: Host tempo
    // void setHostTempo (float bpm) { /* FORBIDDEN */ }

private:
    // Own state only
    AudioProcessorValueTreeState parameters;
};
```

## Audit Checklist

### Timeline UI Compliance (Standalone)
- [ ] Musical grid system (quantized divisions)
- [ ] Smooth zoom behavior (mouse-centered)
- [ ] Intelligent track scaling (content-based)
- [ ] Smooth playhead animation (only during playback)
- [ ] No animation of static elements

### JUCE Embedding Compliance
- [ ] Clear AppKit ↔ JUCE boundaries
- [ ] Proper coordinate conversion
- [ ] Synchronous resize handling
- [ ] Correct ownership hierarchy
- [ ] No JUCE creating AppKit objects

### Plugin Editor Layout Compliance
- [ ] Fixed regions (Header, Parameters, Footer)
- [ ] Proper resizing limits (600x400 min)
- [ ] Host-safe assumptions
- [ ] No control of host functions
- [ ] No modal dialogs or file operations

## Code Examples

### ✅ CORRECT: Timeline Grid System
```cpp
// TimelineGrid.cpp - Musical grid
void TimelineGrid::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    double pixelsPerSecond = getPixelsPerSecond();

    // Draw bar lines (thicker)
    g.setColour (gridColor.darker (0.3f));
    for (int bar = 0; bar < getNumBars(); ++bar) {
        float x = bar * beatsPerBar * pixelsPerSecond * (60.0 / tempoBPM);
        g.drawLine (x, 0, x, bounds.getBottom(), 2.0f);
    }

    // Draw beat lines (medium)
    g.setColour (gridColor.darker (0.1f));
    for (int beat = 0; beat < getNumBeats(); ++beat) {
        float x = beat * pixelsPerSecond * (60.0 / tempoBPM);
        g.drawLine (x, 0, x, bounds.getBottom(), 1.0f);
    }

    // Draw subdivision lines (subtle)
    g.setColour (gridColor.withAlpha (0.3f));
    for (int sub = 0; sub < getNumSubdivisions(); ++sub) {
        float x = sub * pixelsPerSecond * (60.0 / tempoBPM) / 4.0; // Sixteenth notes
        g.drawLine (x, 0, x, bounds.getBottom(), 0.5f);
    }
}
```

### ✅ CORRECT: Plugin Editor Layout
```cpp
// PluginEditor.cpp - Fixed region layout
class DAiWPluginEditor : public juce::AudioProcessorEditor
{
public:
    DAiWPluginEditor (DAiWPluginProcessor& p) : AudioProcessorEditor (&p)
    {
        setSize (800, 600);
        setResizable (true, true);
        setResizeLimits (600, 400, 2000, 1500);

        // Header - fixed top region
        addAndMakeVisible (header);
        header.setBounds (0, 0, 800, 60);

        // Parameters - scrollable middle
        addAndMakeVisible (parameters);
        parameters.setBounds (0, 60, 800, 480);

        // Footer - fixed bottom
        addAndMakeVisible (footer);
        footer.setBounds (0, 540, 800, 60);
    }

    void resized() override {
        // Maintain fixed regions during resize
        auto bounds = getLocalBounds();
        header.setBounds (bounds.removeFromTop (60));
        footer.setBounds (bounds.removeFromBottom (60));
        parameters.setBounds (bounds);
    }

private:
    HeaderComponent header;
    ParameterComponent parameters;
    FooterComponent footer;
};
```

### ❌ WRONG: Host-Unsafe Plugin
```cpp
// WRONG - Plugin trying to control host
class BadPluginProcessor : public juce::AudioProcessor
{
    void processBlock (AudioBuffer<float>& buffer, MidiBuffer& midi) override
    {
        // WRONG: Trying to control host transport
        if (someCondition) {
            // FORBIDDEN: Don't control host playback
            // hostTransport.play(); // ILLEGAL!
        }

        // WRONG: Trying to change host tempo
        // hostTransport.setTempo (120.0f); // ILLEGAL!
    }
};
```

## Non-Compliance Fixes

### If Timeline Issues Found:
1. Implement proper musical grid system
2. Add smooth zoom with mouse centering
3. Implement content-based track scaling
4. Make playhead animation smooth and playback-only
5. Remove animations from static elements

### If JUCE Embedding Issues Found:
1. Create clear ownership boundaries
2. Implement proper coordinate conversion
3. Move expensive operations out of resize
4. Ensure proper destruction order
5. Remove any JUCE creation of AppKit objects

### If Plugin Layout Issues Found:
1. Implement fixed region layout
2. Add proper resize limits
3. Remove host control attempts
4. Convert modals to inline UI
5. Remove file operations and external commands