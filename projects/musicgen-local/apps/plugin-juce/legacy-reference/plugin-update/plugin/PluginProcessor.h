#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "engine/IntentPipeline.h"
#include "common/Types.h"
#include <vector>
#include <mutex>

namespace kelly {

class PluginProcessor : public juce::AudioProcessor {
public:
    PluginProcessor();
    ~PluginProcessor() override = default;
    
    //==========================================================================
    // AudioProcessor overrides
    //==========================================================================
    
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    
    const juce::String getName() const override { return JucePlugin_Name; }
    
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return true; }
    double getTailLengthSeconds() const override { return 0.0; }
    
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;
    
    //==========================================================================
    // Kelly-specific API
    //==========================================================================
    
    /** Generate MIDI from wound description */
    GeneratedMidi generateFromWound(const std::string& description, float intensity);
    
    /** Generate MIDI from Side A/B cassette journey */
    GeneratedMidi generateFromJourney(const SideA& current, const SideB& desired);
    
    /** Get the intent pipeline for UI access */
    IntentPipeline& getIntentPipeline() { return intentPipeline_; }
    
    /** Get the last generated MIDI for drag-and-drop */
    const GeneratedMidi& getLastGeneratedMidi() const { return lastGenerated_; }
    
    /** Export last generated MIDI to a file */
    bool exportMidiToFile(const juce::File& file);
    
    /** Start real-time playback of generated MIDI */
    void startPlayback();
    
    /** Stop real-time playback */
    void stopPlayback();
    
    /** Is currently playing? */
    bool isPlaying() const { return isPlaying_; }
    
    //==========================================================================
    // Parameters
    //==========================================================================
    
    juce::AudioProcessorValueTreeState parameters;
    
private:
    IntentPipeline intentPipeline_;
    GeneratedMidi lastGenerated_;
    
    // Real-time playback state
    struct ScheduledNote {
        int pitch;
        int velocity;
        double startBeat;
        double endBeat;
        bool noteOnSent = false;
        bool noteOffSent = false;
    };
    
    std::vector<ScheduledNote> scheduledNotes_;
    std::mutex notesMutex_;
    
    std::atomic<bool> isPlaying_{false};
    std::atomic<double> playbackPositionBeats_{0.0};
    double lastHostBeat_ = 0.0;
    bool wasHostPlaying_ = false;
    
    double currentSampleRate_ = 44100.0;
    double currentBpm_ = 120.0;
    
    // Parameter IDs
    static constexpr const char* PARAM_INTENSITY = "intensity";
    static constexpr const char* PARAM_HUMANIZE = "humanize";
    static constexpr const char* PARAM_TEMPO_LOCK = "tempoLock";
    
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    
    void scheduleNotesForPlayback();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

} // namespace kelly
