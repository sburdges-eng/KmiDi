#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "engine/IntentPipeline.h"
#include "common/Types.h"
#include "midi/ChordGenerator.h"
#include "audio/AudioRenderer.h"
#include <vector>
#include <mutex>
#include <optional>

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
    
    /** TRUST MVP CHOKE POINT: The only path that may produce output.
        Week 3: supports text/performance/parameter deltas with R-3 precedence. */
    GeneratedMidi runTrustMVP(const UserInput& input);

    /** Bypassed: routes to runTrustMVP or returns {}. */
    GeneratedMidi generateFromWound(const std::string& description, float intensity);

    /** Bypassed: routes to runTrustMVP or returns {}. */
    GeneratedMidi generateFromJourney(const SideA& current, const SideB& desired);
    
    /** Get the intent pipeline for UI access */
    IntentPipeline& getIntentPipeline() { return intentPipeline_; }
    
    /** Get the last generated MIDI for drag-and-drop */
    const GeneratedMidi& getLastGeneratedMidi() const { return lastGenerated_; }
    
    /** Export last generated MIDI to a file */
    bool exportMidiToFile(const juce::File& file);

    /** C2: Request 2–3 voicing suggestions. Opt-in only; call only on explicit user action. */
    std::vector<GeneratedMidi> requestSuggestions();

    struct SuggestionComparison {
        size_t index = 0;
        int changedChords = 0;
        int maxSemitoneShift = 0;
        bool voiceCountPreserved = true;
        double maxStartBeatShift = 0.0;
        double meanStartBeatShift = 0.0;
        double meanDurationShift = 0.0;
    };

    /** C2/D1: Read-only suggestion metadata; never mutates generation state. */
    std::vector<SuggestionComparison> getPendingSuggestionComparisons() const;

    /** C2: Apply a chosen suggestion (0-based index). Replaces lastGenerated_. */
    bool applySuggestion(size_t index);

    /** C2: Number of pending suggestions (0 if none). */
    size_t getPendingSuggestionsCount() const { return pendingSuggestions_.size(); }

    /** D2: Rendered PCM for visualization only. Read-only; no feedback. Empty if no output. */
    std::vector<float> getRenderedAudioForDisplay() const;

    /** Last analyzed intent for optional UI inspection. */
    std::optional<IntentResult> getLastIntentResult() const { return lastIntent_; }

    /** Canonical boundary metadata from the most recent runTrustMVP call. */
    const std::string& getLastIntentSeed() const { return lastIntentSeed_; }
    const std::string& getLastIntentHash() const { return lastIntentHash_; }
    const std::string& getLastOutputMidiHash() const { return lastOutputMidiHash_; }
    
    /** Start real-time playback of generated MIDI. Must be user-initiated only (P-1). */
    void startPlayback(bool fromUserAction = false);
    
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
    ChordGenerator chordGenerator_;
    GeneratedMidi lastGenerated_;
    std::optional<IntentResult> lastIntent_;
    std::string lastIntentSeed_;
    std::string lastIntentHash_;
    std::string lastOutputMidiHash_;
    std::vector<GeneratedMidi> pendingSuggestions_;
    
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
