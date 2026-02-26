#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "plugin/PluginProcessor.h"
#include "plugin/WaveformView.h"

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
    juce::TextButton suggestButton_;
    juce::TextButton useAButton_;
    juce::TextButton useBButton_;
    juce::TextButton useCButton_;
    juce::ToggleButton showAnalysisToggle_;
    
    // Status display
    juce::Label statusLabel_;
    juce::Label emotionDisplay_;
    juce::TextEditor suggestionComparisonView_;

    // D2: Read-only waveform (consumes PCM only; no feedback)
    WaveformView waveformView_;
    
    void onGenerate();
    void onPlayStop();
    void onExport();
    void onSuggest();
    void onUseVariation(int index);
    void updateSuggestionButtons();
    void updateSuggestionComparisonView();
    void updateEmotionDisplay(const IntentResult& result);
    void updatePlayStopButton();
    void updateWaveform();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
