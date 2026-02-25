#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "plugin/PluginProcessor.h"

namespace kelly {

class PluginEditor : public juce::AudioProcessorEditor {
public:
    explicit PluginEditor(PluginProcessor& processor);
    ~PluginEditor() override = default;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
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
    juce::TextButton exportButton_;
    
    // Status display
    juce::Label statusLabel_;
    juce::Label emotionDisplay_;
    
    void onGenerate();
    void onExport();
    void updateEmotionDisplay(const IntentResult& result);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
