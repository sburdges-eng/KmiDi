# 06. Control & Trust Specs

## Overview

These keep users from uninstalling. AI features must be clearly marked, controllable, and respectful of user intent.

## AI Trust & Consent Spec

### Global Enable/Disable

**✓ REQUIRED:**
- Master AI toggle (affects all AI features)
- Prominent placement in preferences/settings
- Immediate effect on all AI systems
- Clear visual indication when AI is disabled

**❌ FORBIDDEN:**
- Hidden AI controls
- Delayed effect of toggles
- AI features active when master toggle is off

**Implementation:**
```cpp
// AIGlobalToggle.h - Master AI control
class AIGlobalToggle : public juce::Component
{
public:
    AIGlobalToggle() {
        addAndMakeVisible (toggleButton);
        toggleButton.setButtonText ("Enable AI Features");
        toggleButton.onClick = [this]() {
            bool enabled = toggleButton.getToggleState();
            setAIGloballyEnabled (enabled);
            updateAllAIComponents (enabled);
        };

        // Load saved preference
        toggleButton.setToggleState (getSavedAIPreference(), juce::dontSendNotification);
    }

    static bool isAIGloballyEnabled() {
        return globalAIEnabled;
    }

private:
    juce::ToggleButton toggleButton;

    static bool globalAIEnabled;

    void setAIGloballyEnabled (bool enabled) {
        globalAIEnabled = enabled;
        saveAIPreference (enabled);

        // Immediate effect on all AI systems
        emotionAnalyzer.setEnabled (enabled);
        harmonySuggester.setEnabled (enabled);
        mlOverlayRenderer.setEnabled (enabled);
        // ... all AI components
    }

    void updateAllAIComponents (bool enabled) {
        // Update UI to reflect AI state
        if (!enabled) {
            // Clear all AI visualizations
            mlOverlays.setVisible (false);
            aiSuggestions.clear();
            emotionBars.setToNeutral();
        }
    }
};
```

### Per-Domain Toggles

**✓ REQUIRED:**
- Separate controls for different AI domains:
  - Melody AI (suggestions, generation)
  - Harmony AI (chord suggestions, progressions)
  - Groove AI (rhythm suggestions, timing)
  - Dynamics AI (mixing suggestions, automation)
  - EQ AI (frequency shaping suggestions)
- Independent of global toggle
- Persistent per-domain preferences

**❌ FORBIDDEN:**
- All-or-nothing AI control
- Domain controls that don't persist
- Hidden per-domain settings

**Implementation:**
```cpp
// AIDomainControls.h - Per-domain AI control
class AIDomainControls : public juce::Component
{
public:
    enum AIDomain { Melody, Harmony, Groove, Dynamics, EQ };

    AIDomainControls() {
        // Create toggle for each domain
        for (int i = 0; i < numDomains; ++i) {
            auto* toggle = new juce::ToggleButton (getDomainName ((AIDomain)i));
            addAndMakeVisible (toggle);
            domainToggles[i] = toggle;

            toggle->onClick = [this, i]() {
                setDomainEnabled ((AIDomain)i, domainToggles[i]->getToggleState());
            };

            // Load saved preference
            toggle->setToggleState (getSavedDomainPreference ((AIDomain)i), juce::dontSendNotification);
        }
    }

    static bool isDomainEnabled (AIDomain domain) {
        return domainEnabled[domain] && AIGlobalToggle::isAIGloballyEnabled();
    }

private:
    juce::ToggleButton* domainToggles[numDomains];

    static bool domainEnabled[numDomains];

    void setDomainEnabled (AIDomain domain, bool enabled) {
        domainEnabled[domain] = enabled;
        saveDomainPreference (domain, enabled);

        // Update relevant AI components
        switch (domain) {
            case Melody: melodyAI.setEnabled (enabled); break;
            case Harmony: harmonyAI.setEnabled (enabled); break;
            case Groove: grooveAI.setEnabled (enabled); break;
            case Dynamics: dynamicsAI.setEnabled (enabled); break;
            case EQ: eqAI.setEnabled (enabled); break;
        }
    }

    static const char* getDomainName (AIDomain domain) {
        switch (domain) {
            case Melody: return "Melody AI";
            case Harmony: return "Harmony AI";
            case Groove: return "Groove AI";
            case Dynamics: return "Dynamics AI";
            case EQ: return "EQ AI";
        }
        return "";
    }
};
```

### Immediate Effect Rules

**✓ REQUIRED:**
- All toggles take effect immediately
- No restart required
- Visual feedback within 100ms
- State changes propagated synchronously

**❌ FORBIDDEN:**
- Delayed toggle effects
- Restart requirements
- Asynchronous state changes
- Hidden state transitions

**Implementation:**
```cpp
// ImmediateEffectManager.h - Synchronous AI control
class ImmediateEffectManager
{
public:
    void setAIFeatureEnabled (AIFeature feature, bool enabled) {
        // Immediate synchronous update
        featureEnabled[feature] = enabled;

        // Direct component updates (no async)
        switch (feature) {
            case EmotionAnalysis:
                emotionComponent.setEnabled (enabled);
                emotionComponent.repaint(); // Immediate visual feedback
                break;

            case HarmonySuggestions:
                harmonyOverlay.setVisible (enabled);
                harmonyOverlay.repaint();
                break;

            case MLSuggestions:
                mlPanel.setEnabled (enabled);
                mlPanel.updateDisplay();
                break;
        }

        // Save preference immediately
        saveFeaturePreference (feature, enabled);

        // Log for debugging
        logFeatureStateChange (feature, enabled);
    }

    bool isFeatureEnabled (AIFeature feature) const {
        return featureEnabled[feature] && AIGlobalToggle::isAIGloballyEnabled();
    }

private:
    std::map<AIFeature, bool> featureEnabled;

    // All operations are synchronous and immediate
    // No timers, no message queues, no async callbacks
};
```

### Persistence Rules

**✓ REQUIRED:**
- All AI preferences persist across sessions
- Per-user, not global settings
- Exported with project files (optional)
- Clear indication of saved state

**❌ FORBIDDEN:**
- AI settings that reset on restart
- Global AI settings that affect all users
- Forced AI preferences

**Implementation:**
```cpp
// AIPersistenceManager.h - Persistent AI preferences
class AIPersistenceManager
{
public:
    void saveAIPreferences (const AIPreferences& prefs) {
        juce::PropertiesFile::Options options;
        options.applicationName = "KmiDi";
        options.filenameSuffix = "ai_prefs";

        auto propsFile = juce::PropertiesFile (options);

        // Save all AI preferences
        propsFile.setValue ("ai_global_enabled", prefs.globalEnabled);
        propsFile.setValue ("ai_melody_enabled", prefs.melodyEnabled);
        propsFile.setValue ("ai_harmony_enabled", prefs.harmonyEnabled);
        propsFile.setValue ("ai_groove_enabled", prefs.grooveEnabled);
        propsFile.setValue ("ai_dynamics_enabled", prefs.dynamicsEnabled);
        propsFile.setValue ("ai_eq_enabled", prefs.eqEnabled);

        // Force save
        propsFile.save();
    }

    AIPreferences loadAIPreferences() {
        juce::PropertiesFile::Options options;
        options.applicationName = "KmiDi";
        options.filenameSuffix = "ai_prefs";

        auto propsFile = juce::PropertiesFile (options);

        AIPreferences prefs;
        prefs.globalEnabled = propsFile.getBoolValue ("ai_global_enabled", true);
        prefs.melodyEnabled = propsFile.getBoolValue ("ai_melody_enabled", true);
        prefs.harmonyEnabled = propsFile.getBoolValue ("ai_harmony_enabled", true);
        prefs.grooveEnabled = propsFile.getBoolValue ("ai_groove_enabled", true);
        prefs.dynamicsEnabled = propsFile.getBoolValue ("ai_dynamics_enabled", true);
        prefs.eqEnabled = propsFile.getBoolValue ("ai_eq_enabled", true);

        return prefs;
    }

private:
    struct AIPreferences {
        bool globalEnabled = true;
        bool melodyEnabled = true;
        bool harmonyEnabled = true;
        bool grooveEnabled = true;
        bool dynamicsEnabled = true;
        bool eqEnabled = true;
    };
};
```

## Undo / History Semantics Spec

### What Is Undoable

**✓ UNDOABLE:**
- User-initiated actions (note edits, parameter changes)
- Explicit AI application ("Apply AI Suggestion")
- Manual adjustments to AI-suggested content
- State changes from UI controls

**❌ NOT UNDOABLE:**
- AI suggestions themselves (they're not actions)
- Automatic AI visualizations
- Background ML processing
- Real-time parameter automation

**Implementation:**
```cpp
// UndoManager.h - Selective undo system
class UndoManager
{
public:
    void performUserAction (UserAction action) {
        // All user actions are undoable
        actionHistory.push (action);
        executeAction (action);
        updateUndoMenu();
    }

    void performAIAction (AIAction action) {
        if (action.type == AIAction::ExplicitApply) {
            // Explicit AI application is undoable
            actionHistory.push (action);
            executeAIAction (action);
            updateUndoMenu();
        } else {
            // AI suggestions, visualizations are NOT undoable
            executeAIAction (action);
        }
    }

    void undo() {
        if (!actionHistory.empty()) {
            auto action = actionHistory.top();
            actionHistory.pop();
            reverseAction (action);
            updateUndoMenu();
        }
    }

private:
    std::stack<UserAction> actionHistory;

    enum ActionType { UserEdit, AIExplicitApply, AISuggestion, Visualization };

    struct UserAction {
        ActionType type;
        std::function<void()> execute;
        std::function<void()> reverse;
    };
};
```

### What Isn't Undoable

**❌ NOT UNDOABLE:**
- AI confidence meters
- Emotion bar updates
- ML overlay visibility changes
- Background processing results
- Automatic parameter smoothing

**✓ RATIONALE:**
- These are observations, not user actions
- Undoing observations doesn't make sense
- Users can't "undo" seeing an emotion reading

**Implementation:**
```cpp
// NonUndoableActions.h - What stays permanent
class NonUndoableActions
{
public:
    static bool isUndoable (ActionType type) {
        switch (type) {
            case NoteEdit: return true;
            case ParameterChange: return true;
            case ExplicitAISuggestion: return true;

            case EmotionUpdate: return false;      // Observation, not action
            case ConfidenceMeterUpdate: return false; // Display only
            case MLOverlayToggle: return false;    // UI preference
            case BackgroundProcessing: return false; // Internal processing

            default: return false;
        }
    }

    static juce::String getUndoDescription (ActionType type) {
        switch (type) {
            case NoteEdit: return "Edit Note";
            case ParameterChange: return "Change Parameter";
            case ExplicitAISuggestion: return "Apply AI Suggestion";

            // No undo descriptions for non-undoable actions
            default: return "";
        }
    }
};
```

### AI Suggestions vs Committed Actions

**✓ DISTINCTION:**
- **AI Suggestions:** Not undoable, can be dismissed, don't affect project
- **Committed Actions:** Undoable, become part of project history, user-initiated

**❌ FORBIDDEN:**
- Treating suggestions as committed actions
- Making suggestions undoable
- Committing suggestions without user consent

**Implementation:**
```cpp
// ActionTypeManager.h - Clear action categorization
class ActionTypeManager
{
public:
    enum ActionCategory { Suggestion, Committed };

    static ActionCategory getCategory (ActionType type) {
        switch (type) {
            case ShowAISuggestion:
            case DisplayHarmonyRegion:
            case UpdateConfidenceMeter:
                return Suggestion; // Not undoable, dismissible

            case ApplyAISuggestion:
            case AcceptChordChange:
            case CommitParameterEdit:
                return Committed; // Undoable, permanent

            default:
                return Suggestion;
        }
    }

    static bool canUndo (ActionType type) {
        return getCategory (type) == Committed;
    }

    static void handleAction (ActionType type, const ActionData& data) {
        if (getCategory (type) == Committed) {
            undoManager.addAction (type, data);
        } else {
            // Handle suggestions (show temporarily, allow dismissal)
            suggestionManager.showSuggestion (type, data);
        }
    }
};
```

### Transaction Boundaries

**✓ REQUIRED:**
- Clear transaction starts/ends
- Undo operates on complete transactions
- Partial transactions don't exist
- Transaction names in undo menu

**❌ FORBIDDEN:**
- Partial undo states
- Incomplete transactions
- Unnamed transactions

**Implementation:**
```cpp
// TransactionManager.h - Complete transaction handling
class TransactionManager
{
public:
    void beginTransaction (const juce::String& name) {
        currentTransaction = std::make_unique<Transaction>();
        currentTransaction->name = name;
        currentTransaction->actions.clear();
    }

    void addActionToTransaction (UserAction action) {
        if (currentTransaction) {
            currentTransaction->actions.push_back (action);
        } else {
            // Single action transaction
            commitSingleAction (action);
        }
    }

    void commitTransaction() {
        if (currentTransaction) {
            undoManager.addTransaction (*currentTransaction);
            currentTransaction.reset();
            updateUndoMenu();
        }
    }

    void cancelTransaction() {
        if (currentTransaction) {
            // Reverse any actions performed so far
            for (auto it = currentTransaction->actions.rbegin();
                 it != currentTransaction->actions.rend(); ++it) {
                reverseAction (*it);
            }
            currentTransaction.reset();
        }
    }

private:
    struct Transaction {
        juce::String name;
        std::vector<UserAction> actions;
    };

    std::unique_ptr<Transaction> currentTransaction;
};
```

## Audit Checklist

### AI Trust & Consent Compliance
- [ ] Global AI enable/disable toggle exists and works immediately
- [ ] Per-domain toggles (melody, harmony, groove, dynamics, EQ) are independent
- [ ] All toggles take effect immediately without restart
- [ ] AI preferences persist across sessions
- [ ] Clear visual indication when AI features are disabled

### Undo / History Semantics Compliance
- [ ] User actions (edits, parameter changes) are undoable
- [ ] Explicit AI application ("Apply AI Suggestion") is undoable
- [ ] AI suggestions, visualizations, and observations are NOT undoable
- [ ] Clear distinction between suggestions and committed actions
- [ ] Transaction boundaries are well-defined and complete

### Control & Trust Overall Compliance
- [ ] Users always maintain ultimate control over their work
- [ ] AI features are clearly marked and can be disabled
- [ ] No "surprise" AI behavior that overrides user intent
- [ ] Clear feedback for all AI state changes
- [ ] Respect for user workflow and creative process

## Code Examples

### ✅ CORRECT: AI Consent Implementation
```cpp
// AIConsentManager.h - User-controlled AI
class AIConsentManager
{
public:
    AIConsentManager() {
        // Load saved preferences
        loadPreferences();
    }

    bool canUseAI (AIDomain domain) {
        // Check global toggle first
        if (!globalAIEnabled) return false;

        // Then check domain-specific toggle
        return domainEnabled[domain];
    }

    void setGlobalAIEnabled (bool enabled) {
        globalAIEnabled = enabled;
        savePreference ("global_ai", enabled);

        // Immediate effect on all AI systems
        updateAllAISystems();
    }

    void setDomainEnabled (AIDomain domain, bool enabled) {
        domainEnabled[domain] = enabled;
        savePreference (getDomainKey (domain), enabled);

        // Immediate effect on specific domain
        updateDomainSystem (domain);
    }

private:
    bool globalAIEnabled = true;
    std::map<AIDomain, bool> domainEnabled;

    void updateAllAISystems() {
        // Synchronous update of all AI components
        melodyAI.setEnabled (globalAIEnabled && domainEnabled[Melody]);
        harmonyAI.setEnabled (globalAIEnabled && domainEnabled[Harmony]);
        // ... etc
    }
};
```

### ✅ CORRECT: Undo System Implementation
```cpp
// SelectiveUndoSystem.h - Appropriate undo behavior
class SelectiveUndoSystem
{
public:
    void performAction (ActionType type, const ActionData& data) {
        switch (type) {
            case UserNoteEdit:
            case UserParameterChange:
                // Undoable user actions
                addToUndoHistory (type, data);
                executeAction (type, data);
                break;

            case AISuggestionShown:
            case EmotionUpdated:
            case ConfidenceChanged:
                // NOT undoable - these are observations
                executeAction (type, data);
                break;

            case AIExplicitApply:
                // Undoable when user explicitly applies AI
                addToUndoHistory (type, data);
                executeAction (type, data);
                break;
        }
    }

    void undo() {
        if (!undoHistory.empty()) {
            auto& lastAction = undoHistory.back();
            reverseAction (lastAction);
            undoHistory.pop_back();
        }
    }

private:
    std::vector<UndoableAction> undoHistory;

    struct UndoableAction {
        ActionType type;
        ActionData data;
        std::function<void()> reverseFunc;
    };
};
```

### ❌ WRONG: Auto-Committing AI
```cpp
// WRONG - AI makes decisions for user
class BadAIControl
{
    void processAudio() {
        // WRONG: AI auto-commits changes
        if (aiThinksBetter()) {
            applyChangeWithoutUserConsent(); // FORBIDDEN!
            // No undo, no user control
        }
    }

    void showSuggestion() {
        // WRONG: Suggestions become committed
        displaySuggestion();
        addToUndoHistory(); // Makes suggestion undoable - wrong!
    }
};
```

## Non-Compliance Fixes

### If AI Auto-Applies Found:
1. Remove all automatic AI application code
2. Convert to explicit user confirmation flows
3. Add "Apply AI Suggestion" buttons for all AI actions
4. Implement proper undo for explicitly applied AI actions

### If Missing AI Controls Found:
1. Add global AI enable/disable toggle
2. Add per-domain toggles for all AI features
3. Ensure immediate effect of all toggles
4. Add persistent preference storage
5. Provide clear visual feedback for AI state

### If Incorrect Undo Behavior Found:
1. Remove undo capability from AI suggestions and observations
2. Ensure only user actions and explicit AI applications are undoable
3. Add transaction boundaries for complex operations
4. Provide clear undo descriptions for all undoable actions

### If Poor Trust Design Found:
1. Add explicit consent flows for AI features
2. Remove any "surprise" AI behavior
3. Add clear labeling of AI-generated content
4. Provide user control over all AI parameters
5. Implement "AI off" mode that completely disables all AI features