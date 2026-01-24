# 05. AI / ML Visibility Specs

## Overview

This is the dangerous part. Specs keep AI tame and trustworthy.

## AI/ML Visibility Rules (EVERYWHERE)

### What AI Can Do

**✓ ALLOWED:**
- **Suggest:** Present options to user
- **Visualize:** Show possibilities graphically
- **Explain briefly:** Provide short, factual descriptions
- **Bias user decisions:** Make good options more prominent

**❌ FORBIDDEN:**
- **Auto-apply edits:** Never change user work without permission
- **Override user intent:** Never contradict explicit user choices
- **Hijack input:** Never intercept or redirect user interactions
- **Speak emotionally authoritative:** Never use language like "you should" or "this feels"

**Implementation:**
```cpp
// AIBehaviorRules.h - What AI can and cannot do
class AIBehaviorRules
{
public:
    static bool canAutoApply (AIFeature feature) {
        // NEVER auto-apply
        return false;
    }

    static bool canSuggest (AIFeature feature) {
        // Suggestions are OK
        return true;
    }

    static bool canVisualize (AIFeature feature) {
        // Visualization is OK
        return feature != RealTimeProcessing; // But not during live performance
    }

    static juce::String createExplanation (AIFeature feature, float confidence) {
        // Short, factual explanations only
        if (confidence > 0.8f) {
            return "High confidence match";
        } else if (confidence > 0.5f) {
            return "Possible match";
        } else {
            return "Low confidence suggestion";
        }
        // NEVER: "This will sound amazing!"
        // NEVER: "You should try this"
    }

private:
    enum AIFeature { HarmonySuggestion, GrooveSuggestion, EQSuggestion, RealTimeProcessing };
};
```

### Update Rate (Throttled, Calm)

**✓ REQUIRED:**
- Maximum 10 updates per second
- Smooth interpolation between states
- No sudden changes
- Throttled to 60fps maximum

**❌ FORBIDDEN:**
- Real-time updates during performance
- Janky animations
- Constant visual noise

**Implementation:**
```cpp
// AIUpdateThrottler.h - Calm, throttled updates
class AIUpdateThrottler : private juce::Timer
{
public:
    void requestUpdate() {
        if (!updatePending) {
            updatePending = true;
            startTimer (100); // Max 10 updates/second
        }
    }

    void timerCallback() override {
        if (updatePending) {
            performSmoothUpdate();
            updatePending = false;
        }
        stopTimer();
    }

private:
    bool updatePending = false;
    float currentValue = 0.0f;
    float targetValue = 0.0f;

    void performSmoothUpdate() {
        // Smooth interpolation over multiple frames
        float delta = targetValue - currentValue;
        currentValue += delta * 0.1f; // 10% interpolation per update

        if (std::abs (delta) > 0.001f) {
            // More updates needed for smooth transition
            startTimer (16); // ~60fps for smooth animation
        }

        updateUI();
    }
};
```

## Emotion Inspector Spec (All Surfaces)

### Data Display

**✓ REQUIRED:**
- **Valence:** -1.0 to +1.0 (negative to positive)
- **Arousal:** 0.0 to 1.0 (calm to excited)
- **Dominance:** 0.0 to 1.0 (submissive to dominant)
- **Complexity:** 0.0 to 1.0 (simple to complex)

**Display Format:**
- Horizontal bars only
- Neutral midpoint for valence (centered)
- Muted colors (not bright/attention-grabbing)
- Optional numeric labels

**Implementation:**
```cpp
// EmotionInspector.h - Read-only bars
class EmotionInspector : public juce::Component
{
public:
    void setEmotion (float valence, float arousal,
                    float dominance, float complexity)
    {
        this->valence = valence;
        this->arousal = arousal;
        this->dominance = dominance;
        this->complexity = complexity;
        repaint();
    }

private:
    float valence = 0.0f;   // -1.0 to 1.0
    float arousal = 0.0f;   // 0.0 to 1.0
    float dominance = 0.0f; // 0.0 to 1.0
    float complexity = 0.0f; // 0.0 to 1.0

    void paint (juce::Graphics& g) override {
        auto bounds = getLocalBounds();

        // Valence bar - bipolar, centered
        drawBipolarBar (g, bounds.removeFromTop (30), valence,
                       juce::Colour::fromRGB (120, 120, 140), "Valence");

        bounds.removeFromTop (10); // Spacing

        // Arousal bar - unipolar
        drawUnipolarBar (g, bounds.removeFromTop (30), arousal,
                        juce::Colour::fromRGB (140, 120, 120), "Arousal");

        bounds.removeFromTop (10);

        // Dominance bar - unipolar
        drawUnipolarBar (g, bounds.removeFromTop (30), dominance,
                        juce::Colour::fromRGB (120, 140, 120), "Dominance");

        bounds.removeFromTop (10);

        // Complexity bar - unipolar
        drawUnipolarBar (g, bounds.removeFromTop (30), complexity,
                        juce::Colour::fromRGB (140, 140, 120), "Complexity");
    }

    void drawBipolarBar (juce::Graphics& g, juce::Rectangle<float> bounds,
                        float value, juce::Colour color, const juce::String& label)
    {
        // Center line at midpoint
        float centerY = bounds.getCentreY();
        g.setColour (color.withAlpha (0.3f));
        g.drawLine (bounds.getX(), centerY, bounds.getRight(), centerY, 1.0f);

        // Value bar from center
        float barWidth = std::abs (value) * bounds.getWidth() * 0.4f;
        float barStart = bounds.getCentreX() + (value >= 0 ? 0 : -barWidth);
        juce::Rectangle<float> bar (barStart, bounds.getY() + 5,
                                   barWidth, bounds.getHeight() - 10);
        g.setColour (color);
        g.fillRect (bar);

        // Label
        g.setColour (juce::Colours::white);
        g.drawText (label, bounds, juce::Justification::left);
    }

    void drawUnipolarBar (juce::Graphics& g, juce::Rectangle<float> bounds,
                         float value, juce::Colour color, const juce::String& label)
    {
        // Background bar
        g.setColour (color.withAlpha (0.2f));
        g.fillRect (bounds);

        // Value fill
        juce::Rectangle<float> fill (bounds.getX(), bounds.getY(),
                                    value * bounds.getWidth(), bounds.getHeight());
        g.setColour (color);
        g.fillRect (fill);

        // Label
        g.setColour (juce::Colours::white);
        g.drawText (label, bounds, juce::Justification::left);
    }
};
```

### Read-Only Rules

**❌ FORBIDDEN:**
- Sliders or knobs
- "Set mood" buttons
- Interactive controls
- Direct manipulation

**✓ REQUIRED:**
- Display only
- No input handling
- Passive observation
- Optional visibility toggle

**Implementation:**
```cpp
// EmotionInspector - No input handling
class EmotionInspector : public juce::Component
{
public:
    // NO mouse/keyboard handling methods

    bool hitTest (int x, int y) override {
        // Allow mouse events to pass through
        return false;
    }

    void mouseDown (const juce::MouseEvent&) override {
        // Do nothing - read-only
    }

    // NO parameter setting methods
    // This is display-only
};
```

### Update Cadence

**✓ REQUIRED:**
- Throttled updates (max 10Hz)
- Smooth transitions
- No sudden jumps
- Optional pause during performance

**❌ FORBIDDEN:**
- Real-time updates
- Jittery animations
- Performance interruptions

**Implementation:**
```cpp
// EmotionUpdateManager.h - Throttled updates
class EmotionUpdateManager : private juce::Timer
{
public:
    void setEmotionUpdateCallback (std::function<void()> callback) {
        updateCallback = callback;
    }

    void requestEmotionUpdate() {
        if (!updateQueued) {
            updateQueued = true;
            startTimerHz (10); // Max 10 updates/second
        }
    }

    void timerCallback() override {
        if (updateQueued && updateCallback) {
            updateCallback();
            updateQueued = false;
        }
    }

private:
    bool updateQueued = false;
    std::function<void()> updateCallback;
};
```

### No Emotional Authority Claims

**❌ FORBIDDEN LANGUAGE:**
- "This song feels sad"
- "You should feel angry here"
- "This captures grief perfectly"
- Any emotional interpretation

**✓ ALLOWED LANGUAGE:**
- "Valence: -0.3"
- "Arousal: 0.7"
- "Dominance: 0.4"
- "Complexity: 0.6"

**Implementation:**
```cpp
// EmotionalLanguageFilter.h - Remove authority claims
class EmotionalLanguageFilter
{
public:
    static juce::String filterExplanation (const juce::String& rawText) {
        // Remove emotional authority claims
        juce::String filtered = rawText;

        filtered = filtered.replace ("feels", "measures");
        filtered = filtered.replace ("should feel", "could be");
        filtered = filtered.replace ("captures", "indicates");
        filtered = filtered.replace ("This song", "Current reading");

        return filtered;
    }

    static juce::String createNeutralDescription (float valence, float arousal) {
        // Factual only
        return juce::String ("Valence: ") + juce::String (valence, 2) +
               ", Arousal: " + juce::String (arousal, 2);
        // NEVER: "This feels melancholic and intense"
    }
};
```

## ML Overlay Rendering Spec (Standalone Only)

### Where Overlays Appear

**✓ ALLOWED LOCATIONS:**
- Timeline background (behind clips)
- Clip-adjacent areas
- Never foreground blocking content
- Optional visibility toggle

**❌ FORBIDDEN LOCATIONS:**
- Over active timeline areas
- Blocking user interaction
- Modal overlays
- Always-visible elements

**Implementation:**
```cpp
// MLOverlayManager.h - Background-only overlays
class MLOverlayManager : public juce::Component
{
public:
    enum OverlayType { MelodyGhost, HarmonyRegions, GrooveHints, EnergyContour };

    void setOverlayVisible (OverlayType type, bool visible) {
        overlayVisibility[type] = visible;
        repaint();
    }

    bool hitTest (int x, int y) override {
        // NEVER intercept mouse events
        return false;
    }

private:
    std::map<OverlayType, bool> overlayVisibility;

    void paint (juce::Graphics& g) override {
        // Draw behind everything else
        g.setColour (juce::Colours::black.withAlpha (0.1f));

        if (overlayVisibility[MelodyGhost]) {
            drawMelodyGhost (g);
        }

        if (overlayVisibility[HarmonyRegions]) {
            drawHarmonyRegions (g);
        }

        // Always low opacity, never blocking
    }

    void drawMelodyGhost (juce::Graphics& g) {
        // Semi-transparent melody suggestions
        g.setColour (juce::Colours::blue.withAlpha (0.3f));
        // Draw ghost notes behind user content
    }
};
```

### Overlay Types

**Melody Ghost Notes:**
- Semi-transparent note representations
- Behind user MIDI clips
- Toggleable visibility
- No interaction

**Harmony Regions:**
- Colored background zones
- Indicate chord suggestions
- Low opacity rectangles
- Zoom-gated visibility

**Groove Timing Hints:**
- Subtle timing markers
- Background grid enhancements
- Rhythmic accent indicators
- Performance-gated

**Energy Contours:**
- Smooth curves showing intensity
- Background waveforms
- Optional display
- Non-interactive

**Implementation:**
```cpp
// OverlayRenderer.h - Different overlay types
class OverlayRenderer
{
public:
    void drawMelodyGhost (juce::Graphics& g, const juce::Rectangle<float>& bounds) {
        g.setColour (juce::Colours::blue.withAlpha (0.2f));
        // Draw semi-transparent note stems and heads
        for (auto& ghostNote : melodySuggestions) {
            drawGhostNote (g, ghostNote);
        }
    }

    void drawHarmonyRegions (juce::Graphics& g, const juce::Rectangle<float>& bounds) {
        for (auto& region : harmonyRegions) {
            juce::Colour regionColor = getHarmonyColor (region.chordType);
            g.setColour (regionColor.withAlpha (0.15f));
            g.fillRect (region.bounds);
        }
    }

    void drawGrooveHints (juce::Graphics& g, const juce::Rectangle<float>& bounds) {
        g.setColour (juce::Colours::green.withAlpha (0.1f));
        // Draw subtle timing markers
        for (auto& hint : grooveHints) {
            g.drawLine (hint.x, bounds.getY(), hint.x, bounds.getBottom(), 1.0f);
        }
    }

private:
    juce::Colour getHarmonyColor (ChordType type) {
        // Subtle colors for harmony suggestions
        switch (type) {
            case Major: return juce::Colour::fromRGB (100, 150, 200);
            case Minor: return juce::Colour::fromRGB (150, 100, 200);
            default: return juce::Colour::fromRGB (200, 150, 100);
        }
    }
};
```

### Visibility Thresholds

**✓ REQUIRED:**
- Zoom-dependent visibility
- Performance state awareness
- User preference controls
- Smooth fade transitions

**❌ FORBIDDEN:**
- Always-visible overlays
- Performance-disrupting updates
- No user control

**Implementation:**
```cpp
// OverlayVisibilityController.h - Smart visibility
class OverlayVisibilityController
{
public:
    bool shouldShowOverlay (OverlayType type, float zoomLevel, bool isPlaying) {
        // Zoom threshold - only show when zoomed in enough
        if (zoomLevel < getMinZoomForType (type)) {
            return false;
        }

        // Performance threshold - hide during playback if distracting
        if (isPlaying && isDistractingDuringPlayback (type)) {
            return false;
        }

        // User preference
        return userWantsOverlay (type);
    }

private:
    float getMinZoomForType (OverlayType type) {
        switch (type) {
            case MelodyGhost: return 2.0f;    // Show when zoomed in 2x
            case HarmonyRegions: return 1.0f; // Show at normal zoom
            case GrooveHints: return 3.0f;    // Show when zoomed in 3x
            default: return 1.0f;
        }
    }

    bool isDistractingDuringPlayback (OverlayType type) {
        // Some overlays are too distracting during live performance
        return type == MelodyGhost; // Too busy during playback
    }
};
```

### Interaction Rules (Mostly "Don't")

**❌ FORBIDDEN INTERACTIONS:**
- Clicking overlays
- Dragging overlay elements
- Hover effects on overlays
- Context menus from overlays

**✓ ALLOWED INTERACTIONS:**
- Toggle visibility (via UI controls)
- Adjust opacity (via preferences)
- Zoom to show/hide
- Performance pause/play gating

**Implementation:**
```cpp
// OverlayInteractionRules.h - No interaction
class OverlayInteractionRules
{
public:
    static bool canInteractWithOverlay (OverlayType type) {
        // NEVER allow direct interaction with overlays
        return false;
    }

    static bool canHoverOverlay (OverlayType type) {
        // No hover effects
        return false;
    }

    static juce::MouseCursor getOverlayCursor (OverlayType type) {
        // Always default cursor - no special cursors
        return juce::MouseCursor::NormalCursor;
    }

    // ALLOWED: Global controls for overlay visibility
    static void setGlobalOverlayOpacity (float opacity) {
        globalOpacity = juce::jlimit (0.0f, 1.0f, opacity);
    }

private:
    static float globalOpacity;
};
```

## AI Explanation / Teaching Spec

### Language Constraints

**❌ FORBIDDEN:**
- "This will sound better"
- "You should try this"
- "This captures the emotion perfectly"
- Prescriptive language

**✓ ALLOWED:**
- "High confidence suggestion"
- "Alternative option available"
- "Based on similar patterns"
- Factual descriptions only

**Implementation:**
```cpp
// ExplanationLanguageFilter.h - Neutral, factual language
class ExplanationLanguageFilter
{
public:
    static juce::String filterExplanation (const juce::String& rawExplanation) {
        juce::String filtered = rawExplanation;

        // Remove prescriptive language
        filtered = filtered.replace ("will sound", "might sound");
        filtered = filtered.replace ("should try", "could try");
        filtered = filtered.replace ("perfectly", "similarly");
        filtered = filtered.replace ("You should", "Consider");

        // Ensure factual tone
        if (!isFactual (filtered)) {
            return "Suggestion available";
        }

        return filtered;
    }

    static bool isFactual (const juce::String& text) {
        // Check for forbidden words
        juce::StringArray forbidden = {"should", "must", "will", "perfect", "amazing", "terrible"};
        for (auto& word : forbidden) {
            if (text.containsIgnoreCase (word)) {
                return false;
            }
        }
        return true;
    }
};
```

### When Explanations Appear

**✓ ALLOWED:**
- After user requests suggestions
- When confidence is high (>80%)
- In dedicated explanation panels
- On explicit user interaction

**❌ FORBIDDEN:**
- Automatically during workflow
- During performance/playback
- As tooltips on hover
- Without user initiation

**Implementation:**
```cpp
// ExplanationTriggerRules.h - Controlled appearance
class ExplanationTriggerRules
{
public:
    static bool shouldShowExplanation (ExplanationContext context, float confidence) {
        switch (context) {
            case UserRequested:
                return true; // Always show when requested

            case HighConfidence:
                return confidence > 0.8f; // Only very confident suggestions

            case ErrorCondition:
                return true; // Always explain errors

            case HoverTooltip:
                return false; // Never on hover

            case Automatic:
                return false; // Never automatic

            default:
                return false;
        }
    }

private:
    enum ExplanationContext {
        UserRequested, HighConfidence, ErrorCondition,
        HoverTooltip, Automatic
    };
};
```

### When Explanations Shut Up

**✓ REQUIRED:**
- Dismissible by user
- Auto-hide after 10 seconds
- Disappear on user action
- Respect "don't show again" preferences

**❌ FORBIDDEN:**
- Persistent explanations
- Undismissible messages
- Constant presence

**Implementation:**
```cpp
// ExplanationDismissalRules.h - Respectful disappearance
class ExplanationDismissalRules : private juce::Timer
{
public:
    void showExplanation (const juce::String& text) {
        currentExplanation = text;
        startTimer (10000); // Auto-hide after 10 seconds
        explanationVisible = true;
        updateUI();
    }

    void dismissExplanation() {
        explanationVisible = false;
        stopTimer();
        updateUI();
    }

    void timerCallback() override {
        // Auto-dismiss after timeout
        dismissExplanation();
    }

    bool shouldShowAgain (ExplanationType type) {
        // Check user preference for "don't show again"
        return !userDismissedPermanently (type);
    }

private:
    juce::String currentExplanation;
    bool explanationVisible = false;

    void updateUI() {
        // Trigger UI update to show/hide explanation
    }
};
```

### What Never Becomes Prescriptive

**❌ FORBIDDEN PRESCRIPTIONS:**
- "Change this note"
- "Use this chord"
- "Adjust the EQ here"
- "This is wrong"

**✓ ALLOWED SUGGESTIONS:**
- "Alternative available"
- "Other options exist"
- "Consider this variation"
- "Pattern detected"

**Implementation:**
```cpp
// PrescriptiveLanguageBlocker.h - Block prescriptions
class PrescriptiveLanguageBlocker
{
public:
    static juce::String makeSuggestion (const juce::String& action) {
        // Convert prescriptions to suggestions
        juce::String suggestion = action;

        suggestion = suggestion.replace ("Change", "Consider changing");
        suggestion = suggestion.replace ("Use", "Try using");
        suggestion = suggestion.replace ("Adjust", "You could adjust");
        suggestion = suggestion.replace ("Fix", "Consider fixing");

        // Add uncertainty markers
        if (!suggestion.startsWith ("Consider") &&
            !suggestion.startsWith ("Try") &&
            !suggestion.startsWith ("You could")) {
            suggestion = "Consider: " + suggestion;
        }

        return suggestion;
    }

    static bool isPrescriptive (const juce::String& text) {
        juce::StringArray prescriptiveWords = {
            "change", "use", "adjust", "fix", "correct",
            "should", "must", "need to", "have to"
        };

        for (auto& word : prescriptiveWords) {
            if (text.containsIgnoreCase (word + " ")) {
                return true;
            }
        }
        return false;
    }
};
```

## Audit Checklist

### AI/ML Visibility Rules Compliance
- [ ] AI can only suggest, visualize, explain briefly
- [ ] AI never auto-applies edits or overrides intent
- [ ] Update rate throttled (max 10Hz)
- [ ] No animation storms or constant visual noise

### Emotion Inspector Compliance
- [ ] Read-only bars for valence/arousal/dominance/complexity
- [ ] Neutral midpoint for valence, muted colors
- [ ] No sliders, knobs, or interactive controls
- [ ] Throttled updates, smooth transitions
- [ ] No emotional authority claims in text

### ML Overlay Compliance (Standalone Only)
- [ ] Overlays only in timeline background, never foreground
- [ ] Low opacity, toggleable visibility
- [ ] Zoom-gated and performance-aware
- [ ] Never intercept mouse events or block interaction
- [ ] Types: melody ghosts, harmony regions, groove hints, energy contours

### AI Explanation Compliance
- [ ] Language remains neutral and factual
- [ ] Explanations appear only when requested or high confidence
- [ ] Auto-dismiss after 10 seconds, user-dismissible
- [ ] Never prescriptive ("should", "must", "change this")

## Code Examples

### ✅ CORRECT: AI Suggestion Implementation
```cpp
// AISuggestionManager.h - Proper AI behavior
class AISuggestionManager
{
public:
    void showChordSuggestion (ChordSuggestion suggestion) {
        if (suggestion.confidence > 0.8f) {
            // Show high-confidence suggestion
            juce::String explanation = "High confidence chord match detected";
            explanationPanel.showExplanation (explanation);

            // Visualize only - don't apply
            harmonyOverlay.showSuggestion (suggestion.chord, suggestion.confidence);
        }
    }

    void userRequestedSuggestions() {
        // Show all available suggestions when explicitly requested
        for (auto& suggestion : availableSuggestions) {
            suggestionPanel.addSuggestion (suggestion);
        }
    }

private:
    // NEVER auto-apply
    // ALWAYS show suggestions for user consideration
    // NEVER override user choices
};
```

### ❌ WRONG: Auto-Applying AI
```cpp
// WRONG - Auto-applying AI
class BadAISystem
{
    void processAudio() {
        // WRONG: Auto-apply AI decisions
        if (aiSuggestsChange()) {
            applyChangeAutomatically(); // FORBIDDEN!
        }

        // WRONG: Override user intent
        if (userHasSetting()) {
            ignoreUserSetting(); // FORBIDDEN!
        }
    }
};
```

### ✅ CORRECT: ML Overlay Rendering
```cpp
// MLOverlayComponent.h - Background-only overlays
class MLOverlayComponent : public juce::Component
{
public:
    MLOverlayComponent() {
        // Never intercept mouse events
        setInterceptsMouseClicks (false, false);
    }

    void paint (juce::Graphics& g) override {
        // Always draw behind user content
        g.setColour (juce::Colours::blue.withAlpha (0.1f)); // Very subtle

        // Draw harmony suggestions
        for (auto& region : harmonySuggestions) {
            g.fillRect (region); // Background only
        }
    }

    bool hitTest (int x, int y) override {
        // Never claim mouse events
        return false;
    }
};
```

## Non-Compliance Fixes

### If AI Auto-Applies Found:
1. Remove all automatic application code
2. Convert to suggestion-only system
3. Add explicit user confirmation for all AI actions
4. Implement undo system for any accidental applications

### If Prescriptive Language Found:
1. Replace with suggestion language ("consider", "try", "could")
2. Remove authority claims ("should", "must", "perfect")
3. Add uncertainty markers to all suggestions
4. Implement language filter for all AI text

### If Overlays Block Interaction:
1. Set hitTest to always return false
2. Move overlays to background z-order
3. Reduce opacity to non-blocking levels
4. Add visibility toggles for user control

### If Real-Time AI Updates Found:
1. Implement throttling (max 10Hz)
2. Add smooth interpolation between states
3. Disable during performance/playback
4. Add user preference controls for update rate