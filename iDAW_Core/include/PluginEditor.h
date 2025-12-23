/*
  ==============================================================================

    PluginEditor.h
    Created: 2025
    Author: DAiW Team

    DAiW Bridge Plugin Editor - UI for connecting to Python brain

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
    DAiW Bridge Audio Processor Editor

    Simple UI with:
    - Text area for emotional input
    - Generate button
    - Parameter sliders (motivation, chaos, vulnerability)
    - Connection status indicator
*/
class DAiWBridgeAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                         public juce::Timer
{
public:
    DAiWBridgeAudioProcessorEditor (DAiWBridgeAudioProcessor&);
    ~DAiWBridgeAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    //==============================================================================
    DAiWBridgeAudioProcessor& audioProcessor;
    
    // UI Components
    juce::TextEditor textEditor;
    juce::TextButton generateButton;
    juce::Slider motivationSlider;
    juce::Slider chaosSlider;
    juce::Slider vulnerabilitySlider;
    juce::Label motivationLabel;
    juce::Label chaosLabel;
    juce::Label vulnerabilityLabel;
    juce::Label statusLabel;
    juce::Label titleLabel;
    
    // Callbacks
    void generateButtonClicked();
    void updateConnectionStatus();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DAiWBridgeAudioProcessorEditor)
};

