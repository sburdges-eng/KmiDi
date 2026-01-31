#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "engine/IntentPipeline.h"
#include "common/Types.h"
#include "biometric/BiometricInput.h"
#include <vector>
#include <mutex>
#include <memory>
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
    
    int getNumPrograms() override { return 61; }  // 60 emotion presets + 1 default
    int getCurrentProgram() override { return currentProgram_; }
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int, const juce::String&) override {}
    
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;
    
    //==========================================================================
    // Kelly-specific API
    //==========================================================================
    
    /** Generate MIDI from wound description */
    GeneratedMidi generateFromWound(const std::string& description, float intensity);
    
    /** Generate MIDI from emotion description */
    GeneratedMidi generateFromEmotion(const std::string& emotionDescription, float intensity);
    
    /** Get the intent pipeline for UI access */
    IntentPipeline& getIntentPipeline() { return intentPipeline_; }
    const IntentPipeline& getIntentPipeline() const { return intentPipeline_; }
    
    //==========================================================================
    // v2.0 Features
    //==========================================================================
    
    /** Enable/disable voice synthesis */
    void setVoiceSynthesisEnabled(bool enabled);
    bool isVoiceSynthesisEnabled() const { return voiceSynthesisEnabled_; }
    
    /** Enable/disable biometric input */
    void setBiometricInputEnabled(bool enabled);
    bool isBiometricInputEnabled() const { return biometricInputEnabled_; }
    
    /** Process biometric data and update emotion parameters */
    void processBiometricData(const BiometricInput::BiometricData& data);
    
    /** Get cache directory for temporary MIDI files */
    juce::File getCacheDirectory() const;
    
    /** Clean up old cache files (called on plugin initialization) */
    void cleanupCache();
    
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
    
    /** Trigger immediate MIDI send (for UI) */
    void triggerImmediateSend() { sendImmediately_ = true; }
    
    /** Export current preset to JSON file */
    bool exportPreset(const juce::File& file) const;
    
    /** Import preset from JSON file */
    bool importPreset(const juce::File& file);
    
    /** Get preset directory for import/export */
    juce::File getPresetDirectory() const;
    
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
        int channel;  // MIDI channel (1=chords, 2=melody, 3=bass)
        bool noteOnSent = false;
        bool noteOffSent = false;
    };
    
    std::vector<ScheduledNote> scheduledNotes_;
    std::mutex notesMutex_;
    
    std::atomic<bool> isPlaying_{false};
    std::atomic<bool> sendImmediately_{false};  // Send notes immediately on next processBlock
    std::atomic<double> playbackPositionBeats_{0.0};
    std::atomic<double> currentBPM_{120.0};  // Current BPM from DAW
    double lastHostBeat_ = 0.0;
    bool wasHostPlaying_ = false;
    
    double currentSampleRate_ = 44100.0;
    int currentProgram_ = 0;
    
    // v2.0 Features
    bool voiceSynthesisEnabled_ = false;
    bool biometricInputEnabled_ = false;
    
    // Forward declarations for v2.0 features
    class VoiceSynthesizer;
    class BiometricInput;
    
    std::unique_ptr<VoiceSynthesizer> voiceSynthesizer_;
    std::unique_ptr<BiometricInput> biometricInput_;
    
    // 36 emotion presets
    struct EmotionPreset {
        int id;
        juce::String name;
        float valence;
        float arousal;
        float intensity;
    };
    static const std::vector<EmotionPreset>& getEmotionPresets();
    
    // Parameter IDs
    static constexpr const char* PARAM_INTENSITY = "intensity";
    static constexpr const char* PARAM_HUMANIZE = "humanize";
    static constexpr const char* PARAM_COMPLEXITY = "complexity";
    static constexpr const char* PARAM_FEEL = "feel";
    static constexpr const char* PARAM_DYNAMICS = "dynamics";
    static constexpr const char* PARAM_BARS = "bars";
    static constexpr const char* PARAM_TEMPO_LOCK = "tempoLock";
    static constexpr const char* PARAM_EMOTION_ID = "emotionId";
    static constexpr const char* PARAM_VALENCE = "valence";
    static constexpr const char* PARAM_AROUSAL = "arousal";
    static constexpr const char* PARAM_CATEGORY = "category";
    static constexpr const char* PARAM_STYLE = "style";
    
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    
    void scheduleNotesForPlayback(bool immediate = false);
    
    // Public access for editor
    friend class PluginEditor;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

} // namespace kelly
