#include "plugin/PluginEditor.h"

namespace kelly {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(p), processor_(p)
{
    setSize(500, 400);
    
    // =========================================================================
    // SIDE A - "Where you are"
    // =========================================================================
    
    sideALabel_.setText("SIDE A - Where You Are", juce::dontSendNotification);
    sideALabel_.setFont(juce::FontOptions(16.0f).withStyle("Bold"));
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
    sideBLabel_.setFont(juce::FontOptions(16.0f).withStyle("Bold"));
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
    
    exportButton_.setButtonText("EXPORT MIDI");
    exportButton_.onClick = [this] { onExport(); };
    exportButton_.setEnabled(false);
    addAndMakeVisible(exportButton_);
    
    // =========================================================================
    // STATUS DISPLAY
    // =========================================================================
    
    statusLabel_.setText("Enter emotions and click Generate", juce::dontSendNotification);
    statusLabel_.setFont(juce::FontOptions(12.0f));
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel_);
    
    emotionDisplay_.setText("", juce::dontSendNotification);
    emotionDisplay_.setFont(juce::FontOptions(14.0f));
    emotionDisplay_.setColour(juce::Label::textColourId, juce::Colours::darkblue);
    addAndMakeVisible(emotionDisplay_);
}

void PluginEditor::paint(juce::Graphics& g) {
    // Background
    g.fillAll(juce::Colour(0xFFF5F5F5));
    
    // Cassette visual divider
    g.setColour(juce::Colours::lightgrey);
    g.drawLine(0, getHeight() / 2.0f - 20, static_cast<float>(getWidth()), getHeight() / 2.0f - 20, 2.0f);
    
    // Header
    g.setColour(juce::Colours::darkgrey);
    g.setFont(juce::FontOptions(24.0f).withStyle("Bold"));
    g.drawText("KELLY", getLocalBounds().removeFromTop(35), juce::Justification::centred);
    
    // Subtitle
    g.setFont(juce::FontOptions(11.0f));
    g.setColour(juce::Colours::grey);
    g.drawText("Therapeutic MIDI Companion", getLocalBounds().removeFromTop(50).translated(0, 30), 
               juce::Justification::centred);
}

void PluginEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    
    // Header space
    bounds.removeFromTop(50);
    
    // SIDE A (top section)
    auto sideAArea = bounds.removeFromTop(85);
    sideALabel_.setBounds(sideAArea.removeFromTop(22));
    sideAInput_.setBounds(sideAArea.removeFromTop(28).reduced(0, 2));
    sideAIntensity_.setBounds(sideAArea.removeFromTop(28));
    
    // Divider gap
    bounds.removeFromTop(15);
    
    // SIDE B (middle section)
    auto sideBArea = bounds.removeFromTop(85);
    sideBLabel_.setBounds(sideBArea.removeFromTop(22));
    sideBInput_.setBounds(sideBArea.removeFromTop(28).reduced(0, 2));
    sideBIntensity_.setBounds(sideBArea.removeFromTop(28));
    
    // Buttons (2 columns now)
    bounds.removeFromTop(15);
    auto buttonArea = bounds.removeFromTop(40);
    generateButton_.setBounds(buttonArea.removeFromLeft(buttonArea.getWidth() / 2).reduced(5));
    exportButton_.setBounds(buttonArea.reduced(5));
    
    // Status area
    bounds.removeFromTop(10);
    emotionDisplay_.setBounds(bounds.removeFromTop(22));
    statusLabel_.setBounds(bounds.removeFromTop(20));
}

void PluginEditor::onGenerate() {
    SideA current{
        sideAInput_.getText().toStdString(),
        static_cast<float>(sideAIntensity_.getValue()),
        std::nullopt
    };
    
    SideB desired{
        sideBInput_.getText().toStdString(),
        static_cast<float>(sideBIntensity_.getValue()),
        std::nullopt
    };
    
    IntentResult result;
    
    // Use Side A alone if Side B is empty
    if (sideBInput_.getText().isEmpty()) {
        processor_.generateFromWound(current.description, current.intensity);
        
        Wound wound{current.description, current.intensity, "ui"};
        result = processor_.getIntentPipeline().process(wound);
    } else {
        processor_.generateFromJourney(current, desired);
        result = processor_.getIntentPipeline().processJourney(current, desired);
    }
    
    updateEmotionDisplay(result);
    
    exportButton_.setEnabled(true);
    
    statusLabel_.setText("Ready! Press PLAY in Logic to hear.", juce::dontSendNotification);
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

void PluginEditor::updateEmotionDisplay(const IntentResult& result) {
    juce::String display;
    display << "Emotion: " << result.emotion.name;
    display << " | Mode: " << result.mode;
    display << " | Tempo: " << juce::String(result.tempo, 2) << "x";
    
    if (!result.ruleBreaks.empty()) {
        display << " | Rules broken: " << static_cast<int>(result.ruleBreaks.size());
    }
    
    emotionDisplay_.setText(display, juce::dontSendNotification);
}

} // namespace kelly
