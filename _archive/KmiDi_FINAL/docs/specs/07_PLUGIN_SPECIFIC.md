# 07. Plugin-Specific Specs

## Overview

DAWs are hostile environments. These specs ensure plugins work correctly within DAW hosts and provide the right level of control.

## Parameter Exposure & Automation Spec

### Which Parameters Exist

**✓ REQUIRED PARAMETERS:**
- `ml_intensity` (0-100%) - Overall AI influence strength
- `melody_influence` (0-100%) - How much AI affects melody generation
- `harmony_influence` (0-100%) - How much AI affects harmony choices
- `groove_influence` (0-100%) - How much AI affects rhythmic feel
- `dynamics_influence` (0-100%) - How much AI affects volume/envelope shaping

**❌ FORBIDDEN PARAMETERS:**
- Direct audio processing parameters (these are internal)
- File paths or system settings
- Host-controlled parameters (tempo, transport, etc.)

**Implementation:**
```cpp
// PluginParameterLayout.h - Correct parameter exposure
void DAiWPluginProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // ML Control Parameters - all automatable
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "ml_intensity", "ML Intensity",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 50.0f,
        juce::String(), juce::AudioProcessorParameter::genericParameter,
        [] (float value, int) { return juce::String (value, 0) + "%"; }));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "melody_influence", "Melody Influence",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 75.0f,
        juce::String(), juce::AudioProcessorParameter::genericParameter,
        [] (float value, int) { return juce::String (value, 0) + "%"; }));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "harmony_influence", "Harmony Influence",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 75.0f,
        juce::String(), juce::AudioProcessorParameter::genericParameter,
        [] (float value, int) { return juce::String (value, 0) + "%"; }));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "groove_influence", "Groove Influence",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 75.0f,
        juce::String(), juce::AudioProcessorParameter::genericParameter,
        [] (float value, int) { return juce::String (value, 0) + "%"; }));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "dynamics_influence", "Dynamics Influence",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 75.0f,
        juce::String(), juce::AudioProcessorParameter::genericParameter,
        [] (float value, int) { return juce::String (value, 0) + "%"; }));

    // Create the parameter state
    parameters = std::make_unique<juce::AudioProcessorValueTreeState> (
        *this, nullptr, "PARAMETERS", std::move (layout));
}
```

### Ranges and Units

**✓ CORRECT RANGES:**
- All influence parameters: 0-100% (linear)
- Default values: 50-75% (balanced, not extreme)
- Steps: 1.0 (fine control for automation)
- Units: Percentage (%)

**❌ FORBIDDEN:**
- Non-linear ranges without clear justification
- Extreme defaults (0% or 100%)
- No units or unclear units
- Ranges that don't make sense for automation

**Implementation:**
```cpp
// ParameterRanges.h - Proper parameter definitions
struct ParameterRanges
{
    static juce::NormalisableRange<float> getMLIntensityRange() {
        return juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f, 1.0f, false);
        // Min: 0%, Max: 100%, Step: 1%, Skew: 1.0 (linear), Symmetric: false
    }

    static juce::NormalisableRange<float> getInfluenceRange() {
        return juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f, 1.0f, false);
        // Same for all influence parameters
    }

    static float getDefaultMLIntensity() { return 50.0f; }  // Balanced default
    static float getDefaultInfluence() { return 75.0f; }    // Slightly engaged

    static juce::String getParameterUnit (const juce::String& paramID) {
        if (paramID.contains ("intensity") || paramID.contains ("influence")) {
            return "%";
        }
        return juce::String();
    }

    static juce::String getParameterLabel (const juce::String& paramID) {
        if (paramID == "ml_intensity") return "ML Intensity";
        if (paramID == "melody_influence") return "Melody";
        if (paramID == "harmony_influence") return "Harmony";
        if (paramID == "groove_influence") return "Groove";
        if (paramID == "dynamics_influence") return "Dynamics";
        return paramID;
    }
};
```

### Smoothing Rules

**✓ REQUIRED:**
- All parameters smoothed to prevent zipper noise
- Smoothing time: 10-50ms (depends on parameter)
- Host-provided smoothing where available
- No audio artifacts from parameter changes

**❌ FORBIDDEN:**
- Instant parameter changes during audio processing
- No smoothing on automatable parameters
- Smoothing that causes latency issues

**Implementation:**
```cpp
// ParameterSmoothing.h - Smooth parameter changes
class ParameterSmoothing
{
public:
    void setTargetValue (int parameterIndex, float target) {
        // Start smoothing to target value
        parameterTargets[parameterIndex] = target;

        // Use appropriate smoothing time based on parameter
        float smoothingTime = getSmoothingTime (parameterIndex);
        parameterSmoothers[parameterIndex].setTime (smoothingTime);
    }

    float getCurrentValue (int parameterIndex) {
        float target = parameterTargets[parameterIndex];
        return parameterSmoothers[parameterIndex].smooth (target);
    }

private:
    std::map<int, float> parameterTargets;
    std::map<int, juce::SmoothedValue<float>> parameterSmoothers;

    float getSmoothingTime (int parameterIndex) {
        // Different smoothing times for different parameters
        switch (parameterIndex) {
            case MLIntensity: return 50.0f;  // Slow for overall intensity
            case MelodyInfluence: return 20.0f;  // Medium for melody
            case HarmonyInfluence: return 30.0f; // Medium-slow for harmony
            case GrooveInfluence: return 10.0f;  // Fast for groove
            case DynamicsInfluence: return 15.0f; // Fast-medium for dynamics
            default: return 20.0f;
        }
    }
};
```

### Host Automation Behavior

**✓ REQUIRED:**
- All parameters respond to host automation
- Smooth automation playback
- No artifacts during automation
- Proper parameter values in host UI

**❌ FORBIDDEN:**
- Parameters that can't be automated
- Automation causing audio glitches
- Parameters not visible in host automation

**Implementation:**
```cpp
// HostAutomationSupport.h - Proper automation behavior
class HostAutomationSupport
{
public:
    void processBlock (juce::AudioBuffer<float>& buffer,
                      juce::MidiBuffer& midiMessages) override
    {
        // Get smoothed parameter values for this block
        float mlIntensity = parameterSmoothing.getCurrentValue (MLIntensity);
        float melodyInfluence = parameterSmoothing.getCurrentValue (MelodyInfluence);
        float harmonyInfluence = parameterSmoothing.getCurrentValue (HarmonyInfluence);
        float grooveInfluence = parameterSmoothing.getCurrentValue (GrooveInfluence);
        float dynamicsInfluence = parameterSmoothing.getCurrentValue (DynamicsInfluence);

        // Use smoothed values for processing
        // No zipper noise, smooth automation response
        processAudioWithParameters (buffer, mlIntensity, melodyInfluence,
                                   harmonyInfluence, grooveInfluence, dynamicsInfluence);
    }

    void parameterChanged (const juce::String& parameterID, float newValue) override
    {
        // Update smoothing target when parameter changes
        int paramIndex = getParameterIndex (parameterID);
        parameterSmoothing.setTargetValue (paramIndex, newValue);
    }

private:
    ParameterSmoothing parameterSmoothing;
};
```

## Master EQ UI Spec (Plug-in Only)

### Purpose (Final Tonal Shaping, User-Owned, AI-Assisted)

**✓ CORE PRINCIPLES:**
- User curve = truth (what gets applied)
- AI curve = suggestion (visual only)
- Explicit apply for AI suggestions
- No auto-mastering

**❌ FORBIDDEN:**
- AI automatically changing the mix
- Hidden EQ processing
- "Smart" EQ that overrides user intent

### Structure (Parametric EQ with AI Suggestions)

**✓ REQUIRED ELEMENTS:**
- User EQ curve (solid line, interactive)
- AI EQ curve (dashed/ghost line, read-only)
- Apply AI button (explicit action)
- A/B comparison functionality
- Clear labeling of what's user vs AI

**Implementation:**
```cpp
// MasterEQComponent.h - User-controlled EQ with AI suggestions
class MasterEQComponent : public juce::Component
{
public:
    MasterEQComponent() {
        // User EQ controls (interactive)
        addAndMakeVisible (freqSlider);
        addAndMakeVisible (gainSlider);
        addAndMakeVisible (qSlider);

        // AI suggestion display (read-only)
        addAndMakeVisible (aiSuggestionLabel);
        aiSuggestionLabel.setText ("AI Suggestion Available", juce::dontSendNotification);

        // Explicit apply button
        addAndMakeVisible (applyAIButton);
        applyAIButton.setButtonText ("Apply AI");
        applyAIButton.onClick = [this]() { applyAISuggestion(); };

        // A/B toggle
        addAndMakeVisible (abToggle);
        abToggle.setButtonText ("A/B");
        abToggle.onClick = [this]() { toggleAB(); };
    }

    void setAISuggestion (const EQCurve& aiCurve) {
        aiSuggestion = aiCurve;
        aiSuggestionLabel.setVisible (true);
        applyAIButton.setVisible (true);
        repaint();
    }

private:
    juce::Slider freqSlider, gainSlider, qSlider;
    juce::Label aiSuggestionLabel;
    juce::TextButton applyAIButton, abToggle;

    EQCurve userCurve;
    EQCurve aiSuggestion;
    bool showingAI = false;

    void applyAISuggestion() {
        userCurve = aiSuggestion; // Explicit user action
        updateEQParameters();
        aiSuggestionLabel.setVisible (false);
        applyAIButton.setVisible (false);
        repaint();
    }

    void toggleAB() {
        showingAI = !showingAI;
        repaint();
    }

    void paint (juce::Graphics& g) override {
        // Draw user curve (solid, always visible)
        drawEQCurve (g, userCurve, juce::Colours::white, true);

        // Draw AI curve (dashed, when available)
        if (aiSuggestion.isValid()) {
            drawEQCurve (g, showingAI ? aiSuggestion : userCurve,
                        juce::Colours::blue.withAlpha (0.5f), false);
        }
    }

    void drawEQCurve (juce::Graphics& g, const EQCurve& curve,
                     juce::Colour color, bool solid) {
        juce::Path path;
        // Draw frequency response curve
        // Solid for user, dashed for AI
    }
};
```

### AI Behavior (Bias Don't Replace, Small Moves)

**✓ AI CONSTRAINTS:**
- ±1.5 dB maximum adjustment (small moves philosophy)
- Shelf-first preference (gentle, musical)
- Back off when user touches controls
- Suggest, don't impose

**❌ FORBIDDEN AI BEHAVIOR:**
- Large EQ changes
- Peak-focused adjustments
- Ignoring user curve adjustments
- Automatic application

**Implementation:**
```cpp
// EQAISuggestions.h - Constrained AI suggestions
class EQAISuggestions
{
public:
    EQCurve suggestEQAdjustments (const AudioAnalysis& analysis,
                                 const EQCurve& userCurve)
    {
        EQCurve suggestion = userCurve; // Start with user curve

        // Small adjustments only (±1.5 dB max)
        const float maxAdjustment = 1.5f;

        // Analyze frequency content
        auto frequencyBalance = analysis.getFrequencyBalance();

        // Gentle shelf adjustments (preferred over peaks)
        if (frequencyBalance.lowEnd < 0.3f) {
            // Suggest slight low shelf boost
            suggestion.adjustLowShelf (juce::jlimit (-maxAdjustment, maxAdjustment, 0.8f));
        }

        if (frequencyBalance.highEnd < 0.3f) {
            // Suggest slight high shelf boost
            suggestion.adjustHighShelf (juce::jlimit (-maxAdjustment, maxAdjustment, 0.8f));
        }

        // Check if user has touched controls recently
        if (userRecentlyAdjustedControls()) {
            // Back off - reduce suggestion intensity
            suggestion.scaleAdjustments (0.5f);
        }

        return suggestion;
    }

private:
    static constexpr float maxAdjustment = 1.5f; // Small moves only

    bool userRecentlyAdjustedControls() {
        // Check if user touched EQ controls in last N seconds
        return juce::Time::getCurrentTime() - lastUserAdjustment < 5000; // 5 seconds
    }
};
```

### Blending Behavior (User Curve = Truth)

**✓ USER PRIORITY:**
- User curve always takes precedence
- AI suggestions are secondary
- Clear visual distinction
- User intent respected

**Implementation:**
```cpp
// EQBlending.h - User curve always wins
class EQBlending
{
public:
    EQParameters getBlendedParameters() {
        // User curve is always applied
        EQParameters params = userCurve.getParameters();

        // AI suggestions are visual only
        // They don't affect audio processing unless explicitly applied
        return params;
    }

    void applyAISuggestion() {
        // Only when user explicitly clicks "Apply AI"
        userCurve = aiSuggestion;
        aiSuggestion.reset(); // Clear suggestion after application
    }

private:
    EQCurve userCurve;     // What actually gets applied
    EQCurve aiSuggestion;  // Visual suggestion only
};
```

### Apply vs Suggest Semantics (Explicit AI Application)

**✓ EXPLICIT ACTIONS:**
- AI suggestions are passive until applied
- "Apply AI" button required for all AI actions
- Clear before/after states
- Undo capability for applied suggestions

**❌ FORBIDDEN:**
- Automatic AI application
- Ambiguous apply/suggest states
- No way to revert AI applications

**Implementation:**
```cpp
// ExplicitAIApplication.h - User must explicitly apply AI
class ExplicitAIApplication
{
public:
    void showAISuggestion (const EQCurve& suggestion) {
        aiSuggestion = suggestion;
        applyButton.setVisible (true);
        suggestionLabel.setText ("AI suggests EQ adjustments. Apply?",
                                juce::dontSendNotification);
        // Visual indication but no audio change
    }

    void applyAISuggestion() {
        // Explicit user action only
        userCurve = aiSuggestion;
        aiSuggestion.reset();

        // Add to undo history
        undoManager.addAction (ActionType::ApplyAISuggestion,
                              userCurve, aiSuggestion);

        applyButton.setVisible (false);
        suggestionLabel.setText ("", juce::dontSendNotification);

        // Update audio processing
        updateEQParameters (userCurve);
    }

private:
    EQCurve aiSuggestion;  // Pending suggestion
    juce::TextButton applyButton;
    juce::Label suggestionLabel;
    UndoManager undoManager;
};
```

### A/B Rules (User Can Compare)

**✓ COMPARISON FEATURES:**
- Toggle between A (user) and B (AI suggestion)
- Clear labeling of current state
- No automatic switching
- User-controlled comparison

**Implementation:**
```cpp
// EQABComparison.h - User-controlled comparison
class EQABComparison
{
public:
    void toggleAB() {
        showingAI = !showingAI;
        updateDisplay();
        updateButtonText();
    }

    EQCurve getDisplayedCurve() const {
        return showingAI ? aiSuggestion : userCurve;
    }

    juce::String getCurrentStateLabel() const {
        return showingAI ? "Showing: AI Suggestion" : "Showing: Your EQ";
    }

private:
    bool showingAI = false;
    juce::TextButton abButton;

    void updateButtonText() {
        abButton.setButtonText (showingAI ? "Show A (Yours)" : "Show B (AI)");
    }

    void updateDisplay() {
        // Update visual display to show selected curve
        // Audio processing always uses userCurve
        repaint();
    }
};
```

## EQ + Chain Coexistence Spec

### Small-Move Philosophy

**✓ GENTLE ADJUSTMENTS:**
- Maximum ±1.5 dB adjustments
- Shelf filters preferred over peaks
- Subtle, musical changes
- User intent preservation

**❌ FORBIDDEN:**
- Large, obvious changes
- Peak filters that create resonances
- Changes that fight user intent

### Shelf vs Bell Preference

**✓ SHELF PREFERENCE:**
- Low shelf for bass adjustments
- High shelf for air/brightness
- Gentle, broad changes
- Musical, not technical

**❌ AVOID BELLS:**
- Narrow peaks create resonances
- Can sound unnatural
- Technical rather than musical

**Implementation:**
```cpp
// EQFilterPreferences.h - Shelf-first philosophy
class EQFilterPreferences
{
public:
    enum FilterType { LowShelf, HighShelf, Bell };

    FilterType chooseFilterType (float frequency, float adjustment) {
        // Prefer shelves for broad adjustments
        if (frequency < 200.0f && adjustment > 0) {
            return LowShelf;  // Bass boost
        }
        if (frequency > 5000.0f && adjustment > 0) {
            return HighShelf; // Air boost
        }

        // Use bells only for specific problems (rare)
        return Bell;
    }

    float getShelfSlope() {
        return 6.0f; // Gentle 6dB/octave slope
    }

    float getMaxBellWidth() {
        return 2.0f; // Octaves - keep bells broad
    }
};
```

### What AI Must Never "Correct"

**❌ NEVER AUTO-CORRECT:**
- User-established frequency balance
- Intentional mixing decisions
- Creative EQ choices
- Technical limitations of source material

**✓ AI CAN SUGGEST:**
- Subtle improvements
- Alternative perspectives
- Technical corrections (with user consent)

### How to Back Off When User Touches Something

**✓ RESPONSIVE BACKING OFF:**
- Detect user control adjustments
- Reduce AI suggestion intensity
- Clear existing suggestions
- Respect user workflow

**Implementation:**
```cpp
// UserInteractionDetection.h - Back off when user touches controls
class UserInteractionDetection
{
public:
    void onUserAdjustedControl (EQControlType control) {
        lastUserAdjustment = juce::Time::getCurrentTime();
        recentlyAdjustedControls.insert (control);

        // Back off AI suggestions
        reduceAISuggestionIntensity();
        clearConflictingSuggestions (control);
    }

    bool shouldApplyAISuggestion (EQControlType control) {
        // Don't apply AI if user recently touched this control
        if (recentlyAdjustedControls.contains (control)) {
            auto timeSinceAdjustment = juce::Time::getCurrentTime() - lastUserAdjustment;
            if (timeSinceAdjustment < 10000) { // 10 seconds
                return false; // Back off
            }
        }
        return true;
    }

    void reduceAISuggestionIntensity() {
        // Make AI suggestions more subtle after user interaction
        aiSuggestionScale *= 0.7f; // Reduce intensity
    }

private:
    juce::Time lastUserAdjustment;
    juce::HashSet<EQControlType> recentlyAdjustedControls;
    float aiSuggestionScale = 1.0f;
};
```

## Audit Checklist

### Parameter Exposure Compliance
- [ ] All required parameters exist (ml_intensity, melody_influence, etc.)
- [ ] All parameters use correct ranges (0-100%)
- [ ] All parameters have appropriate defaults (balanced, not extreme)
- [ ] All parameters are properly smoothed
- [ ] All parameters respond to host automation

### Master EQ UI Compliance (Plugin Only)
- [ ] User EQ curve is solid and interactive (truth)
- [ ] AI EQ curve is ghost/dashed and read-only (suggestion)
- [ ] Explicit "Apply AI" button required for AI application
- [ ] A/B comparison functionality exists
- [ ] AI adjustments limited to ±1.5 dB maximum
- [ ] Shelf filters preferred over bell filters

### EQ + Chain Coexistence Compliance
- [ ] Small-move philosophy enforced (±1.5 dB max)
- [ ] Shelf filters used for broad adjustments
- [ ] AI backs off when user touches controls
- [ ] No auto-correction of user intent
- [ ] User curve always takes precedence

## Code Examples

### ✅ CORRECT: Plugin Parameter Setup
```cpp
// PluginProcessor.cpp - Proper parameter exposure
void DAiWPluginProcessor::createParameters()
{
    auto layout = std::make_unique<juce::AudioProcessorValueTreeState::ParameterLayout>();

    // All influence parameters - automatable, smoothed, 0-100%
    layout->add (std::make_unique<juce::AudioParameterFloat> (
        "ml_intensity", "ML Intensity",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 50.0f));

    layout->add (std::make_unique<juce::AudioParameterFloat> (
        "melody_influence", "Melody Influence",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 75.0f));

    // ... other parameters ...

    parameters = std::make_unique<juce::AudioProcessorValueTreeState> (
        *this, nullptr, "PARAMETERS", std::move (*layout));
}
```

### ✅ CORRECT: Master EQ with AI Suggestions
```cpp
// MasterEQComponent.cpp - User-controlled with AI suggestions
void MasterEQComponent::paint (juce::Graphics& g)
{
    // Always draw user curve (solid, primary)
    drawEQCurve (g, userCurve, juce::Colours::white, juce::PathStrokeType (2.0f));

    // Draw AI suggestion (dashed, secondary, only if available)
    if (aiSuggestion.isValid() && showAISuggestion) {
        juce::PathStrokeType dashedStroke (2.0f);
        std::vector<float> dashLengths = { 4.0f, 4.0f };
        dashedStroke.createDashedStroke (aiCurvePath, aiCurvePath, dashLengths.data(), 2);

        g.setColour (juce::Colours::blue.withAlpha (0.6f));
        g.strokePath (aiCurvePath, dashedStroke);
    }
}

void MasterEQComponent::applyAISuggestion()
{
    // Explicit user action only
    userCurve = aiSuggestion;
    aiSuggestion.reset();

    // Update audio processing
    updateEQFromCurve (userCurve);

    // Hide suggestion UI
    applyButton.setVisible (false);
    aiLabel.setVisible (false);
}
```

### ❌ WRONG: Auto-Applying EQ AI
```cpp
// WRONG - AI automatically changes EQ
void BadMasterEQ::analyzeAndApplyEQ()
{
    // WRONG: Auto-apply AI decisions
    EQCurve aiCurve = aiAnalyzer.suggestEQ();
    userCurve = aiCurve; // No user consent!
    updateEQFromCurve (userCurve); // Changes audio immediately!
}
```

## Non-Compliance Fixes

### If Parameters Not Automatable:
1. Ensure all parameters use AudioProcessorValueTreeState
2. Add proper parameter layout with ranges
3. Implement parameterChanged() callback
4. Test automation in host DAW

### If Master EQ Auto-Applies AI:
1. Make AI suggestions visual-only
2. Add explicit "Apply AI" button
3. Separate AI curve display from user curve
4. Add A/B comparison functionality

### If Large EQ Adjustments Found:
1. Cap all AI adjustments to ±1.5 dB
2. Prefer shelf over bell filters
3. Implement back-off when user touches controls
4. Add user preference for AI adjustment intensity

### If No Parameter Smoothing:
1. Add parameter smoothing to prevent zipper noise
2. Use appropriate smoothing times per parameter
3. Test with fast automation to ensure no artifacts
4. Consider host-provided parameter smoothing