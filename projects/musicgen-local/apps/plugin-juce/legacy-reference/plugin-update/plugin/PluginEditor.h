#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "plugin/PluginProcessor.h"

namespace kelly {

/**
 * The Cassette UI - Side A / Side B interface.
 * 
 * Side A: "Where you are" - current emotional state
 * Side B: "Where you want to go" - desired emotional state
 */
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::Timer {
public:
    explicit PluginEditor(PluginProcessor& processor);
    ~PluginEditor() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
private:
    PluginProcessor& processor_;
    
    // Side A components
    juce::TextEditor sideAInput_;
    juce::Slider sideAIntensity_;
    juce::Label sideALabel_;
    
    // Side B components  
    juce::TextEditor sideBInput_;
    juce::Slider sideBIntensity_;
    juce::Label sideBLabel_;
    
    // Control buttons
    juce::TextButton generateButton_;
    juce::TextButton playStopButton_;
    juce::TextButton exportButton_;
    
    // Status display
    juce::Label statusLabel_;
    juce::Label emotionDisplay_;
    
    void onGenerate();
    void onPlayStop();
    void onExport();
    void updateEmotionDisplay(const IntentResult& result);
    void updatePlayStopButton();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
