#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "Types.h"
#include <atomic>
#include <vector>

namespace kelly {

class PluginProcessor : public juce::AudioProcessor {
public:
    PluginProcessor();
    ~PluginProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Kelly Emotion Processor"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    
    //=========================================================================
    // EMOTION STATE
    //=========================================================================
    
    struct EmotionInput {
        juce::String description;
        float intensity = 0.5f;
    };
    
    EmotionInput sideA{"", 0.7f};
    EmotionInput sideB{"", 0.5f};
    
    //=========================================================================
    // GENERATION PARAMETERS
    //=========================================================================
    
    std::atomic<int> tempo{120};
    std::atomic<int> bars{8};
    juce::String keyRoot{"C"};
    juce::String mode{"Minor"};
    juce::String genre{"Ambient"};
    
    //=========================================================================
    // PLAYBACK STATE
    //=========================================================================
    
    std::atomic<bool> isPlaying{false};
    std::atomic<float> playbackProgress{0.0f};
    std::vector<MidiNote> generatedNotes;
    
    //=========================================================================
    // GENERATION API
    //=========================================================================
    
    void generateFromEmotion();
    void startPlayback();
    void stopPlayback();
    bool exportMidi(const juce::File& outputFile);

private:
    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;
    int playbackPosition = 0;
    
    void processGeneratedMidi(juce::MidiBuffer& midiMessages, int numSamples);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

} // namespace kelly
