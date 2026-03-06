#include "plugin/PluginEditor.h"
#include "common/DebugLog.h"
#include <algorithm>
#include <cctype>

namespace kelly {

namespace {

std::string toLowerCopy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool hasJourneyToken(const juce::String& sideA, const juce::String& sideB) {
    const std::string lowered = toLowerCopy((sideA + " " + sideB).toStdString());
    return lowered.find("journey") != std::string::npos
        || (lowered.find("from ") != std::string::npos && lowered.find(" to ") != std::string::npos);
}

} // namespace

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(p), processor_(p)
{
    // #region agent log
    debugLog("PluginEditor.cpp:ctor", "startup", "build", "instrumented", "A");
    // #endregion
    setSize(500, 580);
    
    // =========================================================================
    // SIDE A - "Where you are"
    // =========================================================================
    
    sideALabel_.setText("SIDE A - Where You Are", juce::dontSendNotification);
    sideALabel_.setFont(juce::Font(16.0f, juce::Font::bold));
    sideALabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(sideALabel_);
    
    sideAInput_.setMultiLine(false);
    sideAInput_.setTextToShowWhenEmpty("Describe your current feeling...", juce::Colours::grey);
    addAndMakeVisible(sideAInput_);
    
    sideAIntensity_.setRange(0.0, 1.0, 0.01);
    sideAIntensity_.setValue(0.7);
    sideAIntensity_.setSliderStyle(juce::Slider::LinearHorizontal);
    sideAIntensity_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(sideAIntensity_);
    
    // =========================================================================
    // SIDE B - "Where you want to go"
    // =========================================================================
    
    sideBLabel_.setText("SIDE B - Where You Want To Go", juce::dontSendNotification);
    sideBLabel_.setFont(juce::Font(16.0f, juce::Font::bold));
    sideBLabel_.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(sideBLabel_);
    
    sideBInput_.setMultiLine(false);
    sideBInput_.setTextToShowWhenEmpty("Describe your desired state...", juce::Colours::grey);
    addAndMakeVisible(sideBInput_);
    
    sideBIntensity_.setRange(0.0, 1.0, 0.01);
    sideBIntensity_.setValue(0.5);
    sideBIntensity_.setSliderStyle(juce::Slider::LinearHorizontal);
    sideBIntensity_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(sideBIntensity_);
    
    // =========================================================================
    // CONTROL BUTTONS
    // =========================================================================
    
    generateButton_.setButtonText("GENERATE");
    generateButton_.onClick = [this] { onGenerate(); };
    addAndMakeVisible(generateButton_);
    
    playStopButton_.setButtonText("PLAY");
    playStopButton_.onClick = [this] { onPlayStop(); };
    playStopButton_.setEnabled(false);
    addAndMakeVisible(playStopButton_);
    
    exportButton_.setButtonText("EXPORT");
    exportButton_.onClick = [this] { onExport(); };
    exportButton_.setEnabled(false);
    addAndMakeVisible(exportButton_);

    suggestButton_.setButtonText("SUGGEST");
    suggestButton_.onClick = [this] { onSuggest(); };
    suggestButton_.setEnabled(false);
    addAndMakeVisible(suggestButton_);

    useAButton_.setButtonText("Use A");
    useAButton_.onClick = [this] { onUseVariation(0); };
    useAButton_.setVisible(false);
    addAndMakeVisible(useAButton_);

    useBButton_.setButtonText("Use B");
    useBButton_.onClick = [this] { onUseVariation(1); };
    useBButton_.setVisible(false);
    addAndMakeVisible(useBButton_);

    useCButton_.setButtonText("Use C");
    useCButton_.onClick = [this] { onUseVariation(2); };
    useCButton_.setVisible(false);
    addAndMakeVisible(useCButton_);

    showAnalysisToggle_.setButtonText("Show analysis");
    showAnalysisToggle_.setToggleState(false, juce::dontSendNotification);
    showAnalysisToggle_.onClick = [this] {
        const bool show = showAnalysisToggle_.getToggleState();
        emotionDisplay_.setVisible(show);
        if (show) {
            if (const auto intent = processor_.getLastIntentResult()) {
                updateEmotionDisplay(*intent);
            } else {
                emotionDisplay_.setText("No analysis available.", juce::dontSendNotification);
            }
        } else {
            emotionDisplay_.setText("", juce::dontSendNotification);
        }
    };
    addAndMakeVisible(showAnalysisToggle_);

    // Read-only branch comparison surface (D1/D2 non-actuating metadata only)
    suggestionComparisonView_.setMultiLine(true);
    suggestionComparisonView_.setReadOnly(true);
    suggestionComparisonView_.setScrollbarsShown(true);
    suggestionComparisonView_.setCaretVisible(false);
    suggestionComparisonView_.setPopupMenuEnabled(false);
    suggestionComparisonView_.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xFFF8FAFC));
    suggestionComparisonView_.setColour(juce::TextEditor::outlineColourId, juce::Colours::lightgrey);
    addAndMakeVisible(suggestionComparisonView_);
    
    // =========================================================================
    // STATUS DISPLAY
    // =========================================================================
    
    statusLabel_.setText("ABSTAIN until one side has text. Then click Generate.", juce::dontSendNotification);
    statusLabel_.setFont(juce::Font(12.0f));
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel_);
    
    emotionDisplay_.setText("", juce::dontSendNotification);
    emotionDisplay_.setFont(juce::Font(14.0f));
    emotionDisplay_.setColour(juce::Label::textColourId, juce::Colours::darkblue);
    addAndMakeVisible(emotionDisplay_);
    // LISTENING: Phase 3 — Hide analysis by default; gate behind "Show analysis" (original contract)
    emotionDisplay_.setVisible(false);
    updateSuggestionComparisonView();

    waveformView_.setAudioBuffer(nullptr, 0);
    addAndMakeVisible(waveformView_);
    
    // Start timer for UI updates
    startTimerHz(30);
}

PluginEditor::~PluginEditor() {
    stopTimer();
}

void PluginEditor::timerCallback() {
    updatePlayStopButton();
    updateSuggestionButtons();
    updateWaveform();
}

void PluginEditor::updatePlayStopButton() {
    if (processor_.isPlaying()) {
        playStopButton_.setButtonText("STOP");
        playStopButton_.setColour(juce::TextButton::buttonColourId, juce::Colours::indianred);
    } else {
        playStopButton_.setButtonText("PLAY");
        playStopButton_.setColour(juce::TextButton::buttonColourId, 
            getLookAndFeel().findColour(juce::TextButton::buttonColourId));
    }
}

void PluginEditor::paint(juce::Graphics& g) {
    // Background
    g.fillAll(juce::Colour(0xFFF5F5F5));
    
    // Cassette visual divider
    g.setColour(juce::Colours::lightgrey);
    g.drawLine(0, getHeight() / 2.0f - 30, static_cast<float>(getWidth()), getHeight() / 2.0f - 30, 2.0f);
    
    // Header
    g.setColour(juce::Colours::darkgrey);
    g.setFont(juce::Font(24.0f, juce::Font::bold));
    g.drawText("KELLY", getLocalBounds().removeFromTop(35), juce::Justification::centred);
    
    // Subtitle
    g.setFont(juce::Font(11.0f));
    g.setColour(juce::Colours::grey);
    g.drawText("Therapeutic MIDI Companion", getLocalBounds().removeFromTop(50).translated(0, 30), 
               juce::Justification::centred);
}

void PluginEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    
    // Header space
    bounds.removeFromTop(50);
    
    // SIDE A (top section)
    auto sideAArea = bounds.removeFromTop(90);
    sideALabel_.setBounds(sideAArea.removeFromTop(25));
    sideAInput_.setBounds(sideAArea.removeFromTop(30).reduced(0, 2));
    sideAIntensity_.setBounds(sideAArea.removeFromTop(30));
    
    // Divider gap
    bounds.removeFromTop(20);
    
    // SIDE B (middle section)
    auto sideBArea = bounds.removeFromTop(90);
    sideBLabel_.setBounds(sideBArea.removeFromTop(25));
    sideBInput_.setBounds(sideBArea.removeFromTop(30).reduced(0, 2));
    sideBIntensity_.setBounds(sideBArea.removeFromTop(30));
    
    // Buttons (4 columns)
    bounds.removeFromTop(15);
    auto buttonArea = bounds.removeFromTop(40);
    int buttonWidth = buttonArea.getWidth() / 4;
    generateButton_.setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(5));
    playStopButton_.setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(5));
    exportButton_.setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(5));
    suggestButton_.setBounds(buttonArea.reduced(5));

    // Suggestion choices (Use A / B / C) - shown when suggestions exist
    auto suggestChoiceArea = bounds.removeFromTop(32);
    int choiceWidth = suggestChoiceArea.getWidth() / 3;
    useAButton_.setBounds(suggestChoiceArea.removeFromLeft(choiceWidth).reduced(3));
    useBButton_.setBounds(suggestChoiceArea.removeFromLeft(choiceWidth).reduced(3));
    useCButton_.setBounds(suggestChoiceArea.reduced(3));

    // Read-only branch comparison text
    bounds.removeFromTop(6);
    suggestionComparisonView_.setBounds(bounds.removeFromTop(58));
    
    // Status area
    bounds.removeFromTop(8);
    showAnalysisToggle_.setBounds(bounds.removeFromTop(24));
    emotionDisplay_.setBounds(bounds.removeFromTop(25));
    statusLabel_.setBounds(bounds.removeFromTop(20));

    // D2: Waveform (read-only; display only)
    bounds.removeFromTop(6);
    waveformView_.setBounds(bounds.removeFromTop(74));
}

void PluginEditor::onGenerate() {
    // TRUST MVP: ONE call site for runTrustMVP. No other generation path.
    juce::String sideA = sideAInput_.getText().trimStart().trimEnd();
    juce::String sideB = sideBInput_.getText().trimStart().trimEnd();
    const bool journeyRequested = hasJourneyToken(sideA, sideB);

    // #region agent log
    debugLogB("PluginEditor.cpp:onGenerate", "onGenerate entry", "sideA_empty", sideA.isEmpty(), "D");
    debugLogB("PluginEditor.cpp:onGenerate", "onGenerate entry", "sideB_empty", sideB.isEmpty(), "D");
    // #endregion

    // Step 2: Empty input → ABSTAIN
    if (sideA.isEmpty() && sideB.isEmpty()) {
        // #region agent log
        debugLog("PluginEditor.cpp:onGenerate", "ABSTAIN path", "reason", "both_empty", "D");
        // #endregion
        statusLabel_.setText("ABSTAIN: Enter text in Side A or Side B.", juce::dontSendNotification);
        emotionDisplay_.setText("", juce::dontSendNotification);
        return;
    }
    // Step 2: Multiple inputs require explicit journey request.
    if (!sideA.isEmpty() && !sideB.isEmpty() && !journeyRequested) {
        // #region agent log
        debugLog("PluginEditor.cpp:onGenerate", "ABSTAIN path", "reason", "both_filled", "D");
        // #endregion
        statusLabel_.setText("ABSTAIN: One input only unless you explicitly request a journey.", juce::dontSendNotification);
        emotionDisplay_.setText("", juce::dontSendNotification);
        return;
    }

    suggestionComparisonView_.setText("Branch comparison is read-only. Click SUGGEST to refresh options.", juce::dontSendNotification);

    GeneratedMidi midiResult;
    if (!sideA.isEmpty() && !sideB.isEmpty()) {
        SideA current{sideA.toStdString(), static_cast<float>(sideAIntensity_.getValue()), std::nullopt};
        SideB desired{sideB.toStdString(), static_cast<float>(sideBIntensity_.getValue()), std::nullopt};
        midiResult = processor_.generateFromJourney(current, desired);
    } else {
        UserInput input;
        input.text = sideA.isEmpty() ? sideB.toStdString() : sideA.toStdString();
        ParameterDelta intensityParam;
        intensityParam.id = "intensity";
        intensityParam.value = static_cast<float>(sideA.isEmpty() ? sideBIntensity_.getValue() : sideAIntensity_.getValue());
        input.parameterDelta = intensityParam;

        // #region agent log
        debugLog("PluginEditor.cpp:onGenerate", "calling runTrustMVP", "input_text", input.text, "D");
        // #endregion

        // LISTENING_GUARDRAIL: I-3, R-1 — empty input never triggers generation
        jassert(!input.text.empty());
        midiResult = processor_.runTrustMVP(input);
    }

    // #region agent log
    debugLogI("PluginEditor.cpp:onGenerate", "runTrustMVP result", "chord_count", (int)midiResult.chords.size(), "A");
    // #endregion

    if (showAnalysisToggle_.getToggleState()) {
        if (const auto intent = processor_.getLastIntentResult()) {
            updateEmotionDisplay(*intent);
        } else {
            emotionDisplay_.setText("", juce::dontSendNotification);
        }
    }

    // ABSTAIN = nothing happens. Success condition, not a bug.
    if (midiResult.chords.empty()) {
        statusLabel_.setText("ABSTAIN: No output. Try: chord change, darker, more energy, or smoother.", juce::dontSendNotification);
        return;
    }

    playStopButton_.setEnabled(true);
    exportButton_.setEnabled(true);
    suggestButton_.setEnabled(true);
    updateSuggestionButtons();
    updateSuggestionComparisonView();
    updateWaveform();
    statusLabel_.setText("Generated minimal response. Press PLAY to hear it.", juce::dontSendNotification);
}

void PluginEditor::onPlayStop() {
    if (processor_.isPlaying()) {
        processor_.stopPlayback();
        statusLabel_.setText("Stopped.", juce::dontSendNotification);
    } else {
        processor_.startPlayback(true);  // User-initiated only (P-1)
        statusLabel_.setText("Playing - press PLAY in Logic transport!", juce::dontSendNotification);
    }
    updatePlayStopButton();
}

void PluginEditor::onSuggest() {
    auto variations = processor_.requestSuggestions();
    if (variations.empty()) {
        updateSuggestionComparisonView();
        statusLabel_.setText("ABSTAIN: Generate first, then Suggest.", juce::dontSendNotification);
        return;
    }
    updateSuggestionButtons();
    updateSuggestionComparisonView();
    juce::String msg = "2 voicing options";
    if (variations.size() == 3)
        msg = "3 voicing options";
    statusLabel_.setText(msg + ". Comparison panel is read-only. Use A, B, or C to apply.", juce::dontSendNotification);
}

void PluginEditor::onUseVariation(int index) {
    if (processor_.applySuggestion(static_cast<size_t>(index))) {
        updateSuggestionButtons();
        updateSuggestionComparisonView();
        updateWaveform();
        playStopButton_.setEnabled(true);
        exportButton_.setEnabled(true);
        statusLabel_.setText("Applied variation. Press PLAY to hear.", juce::dontSendNotification);
    }
}

void PluginEditor::updateSuggestionButtons() {
    const auto n = processor_.getPendingSuggestionsCount();
    useAButton_.setVisible(n >= 1);
    useBButton_.setVisible(n >= 2);
    useCButton_.setVisible(n >= 3);
}

void PluginEditor::updateSuggestionComparisonView() {
    const auto comparisons = processor_.getPendingSuggestionComparisons();
    if (comparisons.empty()) {
        suggestionComparisonView_.setText(
            "Branch comparison is read-only. Generate, then click SUGGEST to inspect A/B/C differences.",
            juce::dontSendNotification);
        return;
    }

    juce::String text = "Read-only branch comparison:\n";
    for (const auto& c : comparisons) {
        juce::String option = juce::String::charToString(
            static_cast<juce::juce_wchar>('A' + static_cast<int>(c.index)));
        text << option
             << ": changed chords=" << c.changedChords
             << ", max shift=" << c.maxSemitoneShift << " st, voices="
             << (c.voiceCountPreserved ? "preserved" : "changed")
             << ", timing(max/mean)="
             << juce::String(c.maxStartBeatShift, 3) << "/"
             << juce::String(c.meanStartBeatShift, 3) << " beats"
             << ", duration mean=" << juce::String(c.meanDurationShift, 3)
             << "\n";
    }
    suggestionComparisonView_.setText(text.trimEnd(), juce::dontSendNotification);
}

void PluginEditor::onExport() {
    auto chooser = std::make_shared<juce::FileChooser>("Save MIDI File",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        "*.mid");

    chooser->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc) {
        auto file = fc.getResult();
        if (file != juce::File{}) {
            if (processor_.exportMidiToFile(file)) {
                statusLabel_.setText("Saved: " + file.getFileName(), juce::dontSendNotification);
            } else {
                statusLabel_.setText("Export failed!", juce::dontSendNotification);
            }
        }
    });
}

void PluginEditor::updateWaveform() {
    auto pcm = processor_.getRenderedAudioForDisplay();
    waveformView_.setAudioBuffer(pcm);
}

void PluginEditor::updateEmotionDisplay(const IntentResult& result) {
    if (!showAnalysisToggle_.getToggleState()) {
        emotionDisplay_.setText("", juce::dontSendNotification);
        return;
    }
    // LISTENING_GUARDRAIL: Do not display analysis for ABSTAIN/INVALID
    if (!result.isValid()) {
        emotionDisplay_.setText("ABSTAIN/INVALID (analysis hidden for non-valid intents).", juce::dontSendNotification);
        return;
    }
    juce::String display;
    display << "Emotion: " << result.emotion.name;
    display << " | Mode: " << result.mode;
    display << " | Tempo: " << juce::String(result.tempo, 2) << "x";
    emotionDisplay_.setText(display, juce::dontSendNotification);
}

} // namespace kelly
