/*
  ==============================================================================

    PluginEditor.cpp
    Created: 2025
    Author: DAiW Team

    DAiW Bridge Plugin Editor - UI for connecting to Python brain

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
DAiWBridgeAudioProcessorEditor::DAiWBridgeAudioProcessorEditor (DAiWBridgeAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Set plugin window size
    setSize (500, 400);
    
    // Title
    titleLabel.setText("DAiW Bridge", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(24.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Text editor for emotional input
    textEditor.setMultiLine(true);
    textEditor.setReturnKeyStartsNewLine(true);
    textEditor.setText("I feel...");
    textEditor.setFont(juce::Font(14.0f));
    addAndMakeVisible(textEditor);
    
    // Generate button
    generateButton.setButtonText("Generate MIDI");
    generateButton.onClick = [this] { generateButtonClicked(); };
    addAndMakeVisible(generateButton);
    
    // Motivation slider
    motivationSlider.setRange(1.0, 10.0, 1.0);
    motivationSlider.setValue(7.0);
    motivationSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(motivationSlider);
    motivationLabel.setText("Motivation", juce::dontSendNotification);
    motivationLabel.attachToComponent(&motivationSlider, true);
    addAndMakeVisible(motivationLabel);
    
    // Chaos slider
    chaosSlider.setRange(1.0, 10.0, 1.0);
    chaosSlider.setValue(5.0);
    chaosSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(chaosSlider);
    chaosLabel.setText("Chaos", juce::dontSendNotification);
    chaosLabel.attachToComponent(&chaosSlider, true);
    addAndMakeVisible(chaosLabel);
    
    // Vulnerability slider
    vulnerabilitySlider.setRange(1.0, 10.0, 1.0);
    vulnerabilitySlider.setValue(5.0);
    vulnerabilitySlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(vulnerabilitySlider);
    vulnerabilityLabel.setText("Vulnerability", juce::dontSendNotification);
    vulnerabilityLabel.attachToComponent(&vulnerabilitySlider, true);
    addAndMakeVisible(vulnerabilityLabel);
    
    // Status label
    statusLabel.setText("Status: Not Connected", juce::dontSendNotification);
    statusLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(statusLabel);
    
    // Check connection status
    updateConnectionStatus();
    
    // Send initial ping
    audioProcessor.sendPing();
    
    // Start timer to periodically check connection
    startTimer(1000); // Check every second
}

DAiWBridgeAudioProcessorEditor::~DAiWBridgeAudioProcessorEditor()
{
}

//==============================================================================
void DAiWBridgeAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
    g.setColour (juce::Colours::white);
    g.setFont (14.0f);
}

void DAiWBridgeAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();
    
    // Title
    titleLabel.setBounds(area.removeFromTop(40).reduced(10));
    
    // Status
    statusLabel.setBounds(area.removeFromTop(25).reduced(10));
    
    // Text editor
    textEditor.setBounds(area.removeFromTop(120).reduced(10));
    
    // Sliders
    auto sliderArea = area.removeFromTop(120);
    motivationSlider.setBounds(sliderArea.removeFromTop(40).reduced(10, 5));
    chaosSlider.setBounds(sliderArea.removeFromTop(40).reduced(10, 5));
    vulnerabilitySlider.setBounds(sliderArea.removeFromTop(40).reduced(10, 5));
    
    // Generate button
    generateButton.setBounds(area.removeFromTop(50).reduced(10));
}

void DAiWBridgeAudioProcessorEditor::generateButtonClicked()
{
    juce::String text = textEditor.getText();
    float motivation = (float)motivationSlider.getValue();
    float chaos = (float)chaosSlider.getValue();
    float vulnerability = (float)vulnerabilitySlider.getValue();
    
    if (text.trim().isEmpty())
    {
        statusLabel.setText("Status: Please enter some text", juce::dontSendNotification);
        return;
    }
    
    if (!audioProcessor.isConnected())
    {
        statusLabel.setText("Status: Not connected to brain server", juce::dontSendNotification);
        return;
    }
    
    statusLabel.setText("Status: Generating...", juce::dontSendNotification);
    
    audioProcessor.sendGenerateRequest(text, motivation, chaos, vulnerability);
}

void DAiWBridgeAudioProcessorEditor::updateConnectionStatus()
{
    if (audioProcessor.isConnected())
    {
        statusLabel.setText("Status: Connected to brain server", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
    }
    else
    {
        statusLabel.setText("Status: Not connected (start brain_server.py)", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    }
}

void DAiWBridgeAudioProcessorEditor::timerCallback()
{
    updateConnectionStatus();
    
    // Send periodic ping to check connection
    audioProcessor.sendPing();
}

