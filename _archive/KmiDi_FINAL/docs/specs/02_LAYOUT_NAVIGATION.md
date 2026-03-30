# 02. Layout & Navigation Specs

## Overview

These stop your UI from turning into soup. Clear layout rules prevent confusion and ensure users can find what they need.

## Standalone App Layout (AppKit + JUCE + SwiftUI)

### Window Structure

**✓ REQUIRED:**
- One main window, native macOS chrome
- Fullscreen + split view support
- No floating inspectors by default

**❌ FORBIDDEN:**
- Multiple independent windows
- Custom window chrome
- Mandatory floating panels

### Timeline (Center, JUCE)

**✓ PRIMARY:**
- Always visible, never hidden
- Multi-track display
- Zoom, scroll, drag operations
- ML overlays allowed (ghost notes, harmony regions, groove indicators)

**Rules:**
- Never blocks input from overlays
- Never auto-applies ML suggestions
- Always remains primary focus area

**Implementation:**
```cpp
// TimelineComponent.h - Primary workspace
class TimelineComponent : public juce::Component
{
public:
    TimelineComponent() {
        // Always visible
        setInterceptsMouseClicks (true, true);

        // ML overlays are background
        addChildComponent (mlOverlay);
        mlOverlay.setInterceptsMouseClicks (false, false); // Never blocks input
    }

private:
    juce::Component mlOverlay; // Background only, never intercepts
};
```

### Inspector (Left, SwiftUI Island)

**✓ READ-ONLY:**
- Emotion Inspector (valence, arousal, dominance, complexity)
- Intent Schema Inspector (core wound, emotional intent, technical constraints)
- ML state visibility (confidence bars, current processing status)

**❌ FORBIDDEN:**
- Interactive controls that change sound
- Knobs or sliders in inspector
- Modal dialogs or popovers

**Implementation:**
```swift
// InspectorView.swift - Read-only island
struct InspectorView: View {
    @ObservedObject var emotionState: EmotionState
    @ObservedObject var intentState: IntentState
    @ObservedObject var mlState: MLState

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Emotion bars - read-only
            EmotionBarsView(emotionState: emotionState)
                .disabled(true) // No interaction

            // Intent display - read-only
            IntentSchemaView(intentState: intentState)
                .disabled(true) // No interaction

            // ML status - read-only
            MLStatusView(mlState: mlState)
                .disabled(true) // No interaction
        }
        .frame(width: 250) // Fixed width
    }
}
```

### Browser (Right, AppKit)

**✓ UTILITY:**
- Presets (save/load states)
- Patterns (reusable musical elements)
- Analysis results (MIDI analysis, harmony detection)
- No real-time DSP

**❌ FORBIDDEN:**
- Audio processing
- Real-time controls
- Modal operations

**Implementation:**
```objc
// BrowserPanel.h - Utility panel
@interface BrowserPanel : NSViewController

@property (strong) PresetBrowser *presetBrowser;
@property (strong) PatternBrowser *patternBrowser;
@property (strong) AnalysisBrowser *analysisBrowser;

// No audio processing here
// Just file management and display

@end
```

### Bottom Panel (Optional, Docked)

**✓ OPTIONAL:**
- Mixer OR editor, user choice
- Docked, not floating
- Hide/show via menu or shortcut
- Persistent state (remembers show/hide)

**❌ FORBIDDEN:**
- Multiple bottom panels
- Undocked operation
- Mandatory display

**Implementation:**
```cpp
// BottomPanelManager.h
class BottomPanelManager
{
public:
    enum PanelType { None, Mixer, Editor };

    void setPanelType (PanelType type) {
        currentPanel = type;
        savePreference (type); // Persist user choice
    }

    void toggleVisibility() {
        visible = !visible;
        updateLayout();
        saveVisibilityPreference (visible);
    }

private:
    PanelType currentPanel = None;
    bool visible = false;
};
```

## Plug-in Layout (JUCE Only)

### Architecture Requirements

**✓ JUCE ONLY:**
- No AppKit, No SwiftUI
- AudioProcessorEditor only
- Fixed regions, no detachable panels

**❌ FORBIDDEN:**
- Dynamic window creation
- Sub-windows or popovers
- Floating panels
- Resizing beyond host constraints

### Header (Top, Fixed)

**✓ REQUIRED:**
- Plugin name (non-editable)
- Bypass button (global enable/disable)
- Preset selector (dropdown)
- No transport controls

**Implementation:**
```cpp
// PluginHeader.h
class PluginHeader : public juce::Component
{
public:
    PluginHeader (AudioProcessor& processor)
    {
        // Plugin name - display only
        addAndMakeVisible (nameLabel);
        nameLabel.setText (processor.getName(), juce::dontSendNotification);
        nameLabel.setEditable (false);

        // Bypass toggle
        addAndMakeVisible (bypassButton);
        bypassButton.setButtonText ("Bypass");

        // Preset selector
        addAndMakeVisible (presetCombo);
        presetCombo.addItemList (getPresetNames(), 1);
    }

private:
    juce::Label nameLabel;
    juce::ToggleButton bypassButton;
    juce::ComboBox presetCombo;
};
```

### Emotion Panel (Top-Middle, Read-Only)

**✓ DISPLAY ONLY:**
- Valence (-1 → 1) - horizontal bar, neutral midpoint
- Arousal (0 → 1) - horizontal bar, low to high
- Dominance (0 → 1) - horizontal bar, low to high
- Complexity (0 → 1) - horizontal bar, simple to complex

**❌ FORBIDDEN:**
- Sliders or knobs
- "Set mood" buttons
- Interactive controls

**Implementation:**
```cpp
// EmotionPanel.h - Read-only bars
class EmotionPanel : public juce::Component
{
public:
    void setEmotion (float valence, float arousal,
                    float dominance, float complexity)
    {
        valenceBar.setValue (valence);   // -1 to 1, center = 0
        arousalBar.setValue (arousal);   // 0 to 1
        dominanceBar.setValue (dominance); // 0 to 1
        complexityBar.setValue (complexity); // 0 to 1
    }

private:
    juce::ProgressBar valenceBar;   // Range: -1 to 1, bipolar
    juce::ProgressBar arousalBar;   // Range: 0 to 1
    juce::ProgressBar dominanceBar; // Range: 0 to 1
    juce::ProgressBar complexityBar; // Range: 0 to 1
};
```

### Parameter Panel (Middle, Interactive)

**✓ CONTROLS:**
- ML Intensity (0-100%) - influences all ML processing
- Melody Influence (0-100%) - how much AI affects melody
- Harmony Influence (0-100%) - how much AI affects harmony
- Groove Influence (0-100%) - how much AI affects rhythm
- Dynamics Influence (0-100%) - how much AI affects dynamics

**Rules:**
- All parameters automatable
- All parameters smoothed (no zipper noise)
- Host automation friendly

**Implementation:**
```cpp
// ParameterPanel.h
class ParameterPanel : public juce::Component
{
public:
    ParameterPanel (AudioProcessor& processor)
    {
        // All parameters automatable
        intensityAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>
            (processor.getParameters(), "ml_intensity", intensitySlider);

        melodyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>
            (processor.getParameters(), "melody_influence", melodySlider);

        // ... other parameters ...
    }

private:
    juce::Slider intensitySlider, melodySlider, harmonySlider,
                 grooveSlider, dynamicsSlider;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
        intensityAttachment, melodyAttachment, harmonyAttachment,
        grooveAttachment, dynamicsAttachment;
};
```

### Master EQ Panel (Bottom, Special)

**✓ USER CONTROLLED:**
- User EQ curve (solid line)
- AI EQ curve (ghost/dashed line)
- Apply AI button (explicit action)
- A/B compare functionality

**❌ FORBIDDEN:**
- Auto-mastering
- AI curve applied without user consent
- No "smart" EQ decisions

**Implementation:**
```cpp
// MasterEQPanel.h
class MasterEQPanel : public juce::Component
{
public:
    void setAISuggestion (const juce::Path& aiCurve) {
        aiCurvePath = aiCurve;
        repaint();
    }

    void applyAISuggestion() {
        userCurvePath = aiCurvePath; // Explicit user action
        updateEQParameters();
        repaint();
    }

private:
    juce::Path userCurvePath; // Solid - user controlled
    juce::Path aiCurvePath;   // Dashed - AI suggestion

    juce::TextButton applyButton {"Apply AI"};

    void paint (juce::Graphics& g) override {
        // Draw user curve (solid)
        g.strokePath (userCurvePath, juce::PathStrokeType (2.0f));

        // Draw AI curve (dashed)
        juce::PathStrokeType dashedStroke (2.0f, juce::PathStrokeType::beveled);
        dashedStroke.createDashedStroke (aiCurvePath, aiCurvePath, nullptr, 4.0f);
        g.strokePath (aiCurvePath, dashedStroke);
    }
};
```

### ML Hint Panel (Bottom-Right, Minimal)

**✓ SUBTLE:**
- Confidence bars (horizontal, small)
- Short text hints (1-2 words)
- No timeline takeover
- No MIDI projection

**❌ FORBIDDEN:**
- Large displays
- Detailed explanations
- Timeline overlays
- MIDI visualization

**Implementation:**
```cpp
// MLHintPanel.h - Minimal, non-intrusive
class MLHintPanel : public juce::Component
{
public:
    void setMLHints (float melodyConfidence, float harmonyConfidence,
                    const juce::String& hintText)
    {
        melodyBar.setValue (melodyConfidence);
        harmonyBar.setValue (harmonyConfidence);
        hintLabel.setText (hintText, juce::dontSendNotification);
    }

private:
    juce::ProgressBar melodyBar;   // Small confidence indicator
    juce::ProgressBar harmonyBar;  // Small confidence indicator
    juce::Label hintLabel;         // 1-2 word hint, e.g. "Strong harmony"
};
```

## Persistence Rules

### Standalone App Persistence

**✓ PERSISTENT:**
- Panel layout (which panels visible, sizes)
- Inspector visibility state
- Bottom panel type (mixer/editor/none)
- Emotion history (optional, user-controlled)

**❌ NOT PERSISTENT:**
- Current playback position
- Temporary UI states
- Modal dialog positions

**Implementation:**
```cpp
// LayoutPersistence.h
class LayoutPersistence
{
public:
    void saveLayout (const LayoutState& state) {
        juce::PropertiesFile::Options options;
        options.applicationName = "KmiDi";
        options.filenameSuffix = "layout";

        auto props = juce::PropertiesFile (options);
        props.setValue ("inspectorVisible", state.inspectorVisible);
        props.setValue ("bottomPanelType", static_cast<int>(state.bottomPanelType));
        props.setValue ("browserWidth", state.browserWidth);
    }

    LayoutState loadLayout() {
        // Load and return persisted state
    }
};
```

### Plug-in Persistence

**✓ PERSISTENT:**
- Parameters (handled by host)
- UI state (if allowed by host)

**❌ NOT PERSISTENT:**
- Emotion history
- Intent schema state
- ML processing state

**Implementation:**
```cpp
// Plugin persistence is handled by host
// Plugin just exposes parameters
void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // Only parameters, no UI state
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}
```

## Hierarchy & Visual Priority Spec

### Primary (Always Visible, High Contrast)

**Standalone:**
- Timeline (center) - primary workspace
- Transport controls (top) - always accessible

**Plug-in:**
- Parameter controls - primary interaction
- EQ curve - primary visual feedback

### Secondary (Contextual, Medium Contrast)

**Standalone:**
- Inspector panels (when visible)
- Browser content
- Bottom panels

**Plug-in:**
- Emotion bars (informational)
- ML hints (subtle)

### Tertiary (Optional, Low Contrast)

**Standalone:**
- ML overlays (background)
- Status indicators
- Help hints

**Plug-in:**
- Preset names
- Version info

### Inactive State Rules

**✓ WHEN INACTIVE:**
- Fade to 60% opacity
- Reduce saturation
- Maintain readability
- Clear inactive indication

**❌ FORBIDDEN:**
- Complete hiding
- Color inversion
- Illegible text

**Implementation:**
```cpp
// VisualPriority.h
class VisualPriority
{
public:
    enum Level { Primary, Secondary, Tertiary };

    void setPriority (juce::Component& component, Level level) {
        switch (level) {
            case Primary:
                component.setAlpha (1.0f);
                // High contrast, full saturation
                break;
            case Secondary:
                component.setAlpha (0.8f);
                // Medium contrast
                break;
            case Tertiary:
                component.setAlpha (0.6f);
                // Low contrast, desaturated
                break;
        }
    }
};
```

## Audit Checklist

### Standalone Layout Compliance
- [ ] One main window with AppKit chrome
- [ ] Timeline center, always visible, primary focus
- [ ] Inspector left: read-only, no sound-changing controls
- [ ] Browser right: presets/patterns/analysis, no DSP
- [ ] Bottom panel: optional, docked, persistent state
- [ ] No floating inspectors by default

### Plug-in Layout Compliance
- [ ] JUCE only, no AppKit/SwiftUI
- [ ] Fixed regions, no detachable panels
- [ ] Header: name, bypass, preset selector
- [ ] Emotion panel: read-only bars only
- [ ] Parameter panel: automatable controls, smoothed
- [ ] Master EQ: user curve solid, AI curve ghost, explicit apply
- [ ] ML hints: minimal, non-intrusive
- [ ] No dynamic window creation

### Persistence Compliance
- [ ] Standalone: panel layout and visibility persist
- [ ] Standalone: bottom panel type persists
- [ ] Plug-in: only parameters persist (host handles)
- [ ] No emotion history in plug-in

### Visual Priority Compliance
- [ ] Primary elements: high contrast, always visible
- [ ] Secondary elements: medium contrast, contextual
- [ ] Tertiary elements: low contrast, optional
- [ ] Inactive states: 60% opacity, maintain readability

## Code Examples

### ✅ CORRECT: Standalone Layout
```cpp
// StandaloneApp.h
class StandaloneApp : public juce::Component
{
public:
    StandaloneApp() {
        // AppKit window (handled by NSApplication)

        // JUCE timeline - center, primary
        addAndMakeVisible (timeline);
        timeline.setBounds (250, 0, getWidth()-500, getHeight()-100);

        // SwiftUI inspector - left, read-only
        // (Handled by SwiftUI bridge)

        // AppKit browser - right, utility
        // (Handled by NSViewController)

        // Optional bottom panel
        if (showBottomPanel)
            addAndMakeVisible (bottomPanel);
    }

private:
    TimelineComponent timeline;     // JUCE - primary
    // Inspector and browser handled by platform bridges
    juce::Component bottomPanel;    // Optional, docked
};
```

### ✅ CORRECT: Plugin Layout
```cpp
// PluginEditor.h
class PluginEditor : public juce::AudioProcessorEditor
{
public:
    PluginEditor (PluginProcessor& p) : AudioProcessorEditor (p)
    {
        setSize (800, 600); // Fixed size

        // Header - top
        addAndMakeVisible (header);
        header.setBounds (0, 0, 800, 50);

        // Emotion panel - read-only
        addAndMakeVisible (emotionPanel);
        emotionPanel.setBounds (0, 50, 800, 80);

        // Parameter panel - interactive
        addAndMakeVisible (parameterPanel);
        parameterPanel.setBounds (0, 130, 800, 200);

        // Master EQ - bottom
        addAndMakeVisible (masterEQ);
        masterEQ.setBounds (0, 330, 600, 200);

        // ML hints - bottom-right, minimal
        addAndMakeVisible (mlHints);
        mlHints.setBounds (600, 330, 200, 200);
    }

private:
    PluginHeader header;
    EmotionPanel emotionPanel;      // Read-only bars
    ParameterPanel parameterPanel;  // Interactive controls
    MasterEQPanel masterEQ;         // User-controlled EQ
    MLHintPanel mlHints;           // Minimal hints
};
```

### ❌ WRONG: Dynamic Plugin Layout
```cpp
// WRONG - No dynamic windows in plugin
class BadPluginEditor : public juce::AudioProcessorEditor
{
    void showFloatingInspector() {
        // WRONG: Creates floating window
        auto* window = new juce::DocumentWindow ("Inspector",
                                               juce::Colours::darkgrey,
                                               juce::DocumentWindow::allButtons);
        window->setContentOwned (new InspectorComponent(), true);
        window->setVisible (true); // WRONG!
    }
};
```

## Non-Compliance Fixes

### If Floating Panels Found:
1. Convert to docked panels
2. Use tabbed interfaces
3. Make optional with user preference

### If Interactive Inspector Found:
1. Remove all controls that change sound
2. Convert to read-only displays
3. Move interactive controls to main parameter panel

### If No Persistence Found:
1. Add juce::PropertiesFile for standalone
2. Ensure host handles plugin state
3. Add user preference system

### If Poor Visual Hierarchy Found:
1. Define primary/secondary/tertiary components
2. Adjust contrast and opacity levels
3. Ensure inactive states maintain readability

### If Plug-in Uses AppKit/SwiftUI:
1. Convert to JUCE equivalents
2. Remove platform-specific code
3. Use JUCE look and feel system