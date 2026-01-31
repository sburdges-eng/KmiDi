#include "plugin/PluginProcessor.h"
#include "plugin/PluginEditor.h"
#include "midi/MidiGenerator.h"
#include "midi/MidiBuilder.h"
#include "voice/VoiceSynthesizer.h"
#include "biometric/BiometricInput.h"
#include <juce_core/juce_core.h>

namespace kelly {

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "KellyParams", createParameterLayout()),
      currentProgram_(0),
      voiceSynthesisEnabled_(false),
      biometricInputEnabled_(false)
{
    cleanupCache();
    
    // Initialize v2.0 features
    voiceSynthesizer_ = std::make_unique<VoiceSynthesizer>();
    biometricInput_ = std::make_unique<BiometricInput>();
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // Core generation parameters
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_INTENSITY, 1}, "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.7f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_COMPLEXITY, 1}, "Complexity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_HUMANIZE, 1}, "Humanize",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.4f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_FEEL, 1}, "Feel",
        juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f), 0.0f)); // Pull/Push
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_DYNAMICS, 1}, "Dynamics",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.75f));
    
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{PARAM_BARS, 1}, "Bars",
        4, 32, 8));
    
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_TEMPO_LOCK, 1}, "Lock to DAW Tempo", true));
    
    // Category and Style (for future expansion)
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{PARAM_CATEGORY, 1}, "Category",
        0, 10, 0));
    
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{PARAM_STYLE, 1}, "Style",
        0, 20, 0));
    
    // Emotion parameters
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{PARAM_EMOTION_ID, 1}, "Emotion ID",
        0, 1000, 0));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_VALENCE, 1}, "Valence",
        juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f), 0.0f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_AROUSAL, 1}, "Arousal",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    
    return {params.begin(), params.end()};
}

const std::vector<PluginProcessor::EmotionPreset>& PluginProcessor::getEmotionPresets() {
    static std::vector<EmotionPreset> presets = {
        // Joy cluster
        {1, "Joy", 0.8f, 0.7f, 0.6f},
        {2, "Happiness", 0.7f, 0.6f, 0.5f},
        {3, "Elation", 0.9f, 0.9f, 0.8f},
        {4, "Euphoria", 1.0f, 1.0f, 1.0f},
        {5, "Contentment", 0.6f, 0.3f, 0.4f},
        {6, "Gratitude", 0.8f, 0.4f, 0.5f},
        
        // Sadness cluster
        {7, "Sadness", -0.6f, 0.3f, 0.5f},
        {8, "Grief", -0.9f, 0.3f, 0.9f},
        {9, "Melancholy", -0.6f, 0.2f, 0.4f},
        {10, "Despair", -1.0f, 0.4f, 0.9f},
        {11, "Sorrow", -0.8f, 0.3f, 0.6f},
        {12, "Loneliness", -0.5f, 0.2f, 0.5f},
        
        // Anger cluster
        {13, "Anger", -0.8f, 0.9f, 0.7f},
        {14, "Rage", -0.9f, 1.0f, 1.0f},
        {15, "Fury", -0.9f, 0.95f, 0.9f},
        {16, "Frustration", -0.5f, 0.7f, 0.6f},
        {17, "Irritation", -0.3f, 0.5f, 0.4f},
        {18, "Resentment", -0.6f, 0.5f, 0.6f},
        
        // Fear cluster
        {19, "Fear", -0.7f, 0.8f, 0.7f},
        {20, "Terror", -0.9f, 1.0f, 1.0f},
        {21, "Anxiety", -0.6f, 0.7f, 0.7f},
        {22, "Dread", -0.8f, 0.6f, 0.8f},
        {23, "Worry", -0.4f, 0.5f, 0.5f},
        {24, "Panic", -0.7f, 1.0f, 0.9f},
        
        // Surprise cluster
        {25, "Surprise", 0.2f, 0.8f, 0.6f},
        {26, "Astonishment", 0.3f, 0.9f, 0.7f},
        {27, "Amazement", 0.5f, 0.7f, 0.6f},
        {28, "Shock", 0.0f, 0.9f, 0.8f},
        {29, "Wonder", 0.6f, 0.5f, 0.5f},
        {30, "Awe", 0.7f, 0.4f, 0.6f},
        
        // Complex/Neutral
        {31, "Peace", 0.5f, 0.1f, 0.3f},
        {32, "Calm", 0.4f, 0.1f, 0.2f},
        {33, "Serenity", 0.6f, 0.2f, 0.3f},
        {34, "Nostalgia", -0.2f, 0.3f, 0.4f},
        {35, "Longing", -0.4f, 0.5f, 0.6f},
        {36, "Hope", 0.5f, 0.4f, 0.5f},
        
        // Additional v1.1 presets - Complex emotions
        {37, "Bittersweet", -0.1f, 0.4f, 0.5f},
        {38, "Melancholy", -0.5f, 0.3f, 0.4f},
        {39, "Euphoria", 0.9f, 0.9f, 0.9f},
        {40, "Catharsis", 0.3f, 0.7f, 0.8f},
        {41, "Yearning", -0.3f, 0.6f, 0.7f},
        {42, "Triumph", 0.8f, 0.8f, 0.8f},
        {43, "Despair", -0.9f, 0.4f, 0.9f},
        {44, "Ecstasy", 0.95f, 1.0f, 1.0f},
        {45, "Resignation", -0.6f, 0.2f, 0.5f},
        {46, "Determination", 0.4f, 0.7f, 0.8f},
        {47, "Vulnerability", -0.2f, 0.5f, 0.6f},
        {48, "Empowerment", 0.7f, 0.7f, 0.7f},
        {49, "Isolation", -0.7f, 0.3f, 0.6f},
        {50, "Connection", 0.6f, 0.5f, 0.6f},
        {51, "Release", 0.2f, 0.6f, 0.7f},
        {52, "Tension", -0.1f, 0.8f, 0.8f},
        {53, "Relief", 0.5f, 0.3f, 0.4f},
        {54, "Anticipation", 0.3f, 0.7f, 0.6f},
        {55, "Fulfillment", 0.7f, 0.4f, 0.5f},
        {56, "Emptiness", -0.8f, 0.2f, 0.5f},
        {57, "Wholeness", 0.8f, 0.3f, 0.4f},
        {58, "Fragility", -0.3f, 0.4f, 0.5f},
        {59, "Resilience", 0.5f, 0.6f, 0.7f},
        {60, "Transcendence", 0.9f, 0.5f, 0.6f}
    };
    return presets;
}

void PluginProcessor::setCurrentProgram(int index) {
    if (index < 0 || index >= getNumPrograms()) return;
    
    currentProgram_ = index;
    
    if (index == 0) {
        // Default: neutral values
        *parameters.getRawParameterValue(PARAM_VALENCE) = 0.0f;
        *parameters.getRawParameterValue(PARAM_AROUSAL) = 0.5f;
        *parameters.getRawParameterValue(PARAM_INTENSITY) = 0.5f;
        *parameters.getRawParameterValue(PARAM_EMOTION_ID) = 0;
    } else {
        // Load emotion preset
        const auto& presets = getEmotionPresets();
        if (index <= static_cast<int>(presets.size())) {
            const auto& preset = presets[index - 1];
            *parameters.getRawParameterValue(PARAM_VALENCE) = preset.valence;
            *parameters.getRawParameterValue(PARAM_AROUSAL) = preset.arousal;
            *parameters.getRawParameterValue(PARAM_INTENSITY) = preset.intensity;
            *parameters.getRawParameterValue(PARAM_EMOTION_ID) = preset.id;
        }
    }
}

const juce::String PluginProcessor::getProgramName(int index) {
    if (index == 0) return "Default";
    const auto& presets = getEmotionPresets();
    if (index > 0 && index <= static_cast<int>(presets.size())) {
        return presets[index - 1].name;
    }
    return {};
}

juce::File PluginProcessor::getCacheDirectory() const {
    auto musicDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    auto cacheDir = musicDir.getChildFile("Kelly MIDI Companion");
    cacheDir.createDirectory();
    return cacheDir;
}

void PluginProcessor::cleanupCache() {
    auto cacheDir = getCacheDirectory();
    if (!cacheDir.isDirectory()) return;
    
    // Delete all .mid files in cache (they're temporary)
    auto files = cacheDir.findChildFiles(juce::File::findFiles, false, "*.mid");
    for (const auto& file : files) {
        file.deleteFile();
    }
}

void PluginProcessor::prepareToPlay(double sampleRate, int) {
    currentSampleRate_ = sampleRate;
    currentBPM_.store(120.0);  // Default BPM
}

void PluginProcessor::releaseResources() {
    stopPlayback();
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    buffer.clear();
    
    // Get current emotion parameters (with automation support)
    // JUCE's APVTS automatically handles automation, so we just read the values
    float valence = *parameters.getRawParameterValue(PARAM_VALENCE);
    float arousal = *parameters.getRawParameterValue(PARAM_AROUSAL);
    int emotionId = (int)*parameters.getRawParameterValue(PARAM_EMOTION_ID);
    float intensity = *parameters.getRawParameterValue(PARAM_INTENSITY);
    
    // Enhanced DAW transport sync - get more accurate timing
    auto* playhead = getPlayHead();
    if (playhead) {
        juce::AudioPlayHead::CurrentPositionInfo pos;
        if (playhead->getCurrentPosition(pos)) {
            // Use more accurate timing from playhead
            if (pos.bpm > 0.0) {
                currentBPM_.store(pos.bpm);
            }
            // Better beat position tracking
            if (pos.isPlaying) {
                playbackPositionBeats_.store(pos.ppqPosition);
            } else {
                playbackPositionBeats_.store(pos.ppqPosition);  // Still track position when stopped
            }
        }
    }
    
    // If emotion ID is set, use it; otherwise use valence/arousal
    if (emotionId > 0) {
        auto emotion = intentPipeline_.thesaurus().findById(emotionId);
        if (emotion) {
            valence = emotion->valence;
            arousal = emotion->arousal;
            intensity = emotion->intensity;
        }
    }
    
    // Auto-generate MIDI if parameters changed significantly (simple implementation)
    static float lastValence = -999.0f;
    static float lastArousal = -999.0f;
    static int lastEmotionId = -1;
    
    bool shouldRegenerate = false;
    if (std::abs(valence - lastValence) > 0.1f || 
        std::abs(arousal - lastArousal) > 0.1f || 
        emotionId != lastEmotionId) {
        shouldRegenerate = true;
        lastValence = valence;
        lastArousal = arousal;
        lastEmotionId = emotionId;
    }
    
    if (shouldRegenerate && emotionId > 0) {
        // Generate from emotion
        auto emotion = intentPipeline_.thesaurus().findById(emotionId);
        if (emotion) {
            Wound wound{"emotion_selected", intensity, "parameter"};
            IntentResult intent = intentPipeline_.process(wound);
            intent.emotion = *emotion;
            
            // Use full generator with current parameters
            float complexity = *parameters.getRawParameterValue(PARAM_COMPLEXITY);
            float humanize = *parameters.getRawParameterValue(PARAM_HUMANIZE);
            float feel = *parameters.getRawParameterValue(PARAM_FEEL);
            float dynamics = *parameters.getRawParameterValue(PARAM_DYNAMICS);
            int bars = (int)*parameters.getRawParameterValue(PARAM_BARS);
            
            MidiGenerator generator;
            lastGenerated_ = generator.generate(intent, bars, complexity, humanize, feel, dynamics);
            
            scheduleNotesForPlayback();
        }
    }
    
    // Playback scheduled notes
    if (scheduledNotes_.empty()) return;
    
    // Get host info
    double bpm = 120.0;
    double hostBeat = 0.0;
    bool hostIsPlaying = false;
    
    if (auto* playHead = getPlayHead()) {
        if (auto pos = playHead->getPosition()) {
            if (auto b = pos->getBpm()) bpm = *b;
            if (auto p = pos->getPpqPosition()) hostBeat = *p;
            hostIsPlaying = pos->getIsPlaying();
        }
    }
    
    double samplesPerBeat = (currentSampleRate_ * 60.0) / bpm;
    double beatsPerSample = 1.0 / samplesPerBeat;
    double bufferBeats = buffer.getNumSamples() * beatsPerSample;
    double blockEnd = hostBeat + bufferBeats;
    
    std::lock_guard<std::mutex> lock(notesMutex_);
    
    // If sendImmediately_ is set, send all notes starting from sample 0
    bool immediateMode = sendImmediately_.exchange(false);
    
    for (auto& n : scheduledNotes_) {
        // Use the channel from the scheduled note
        int channel = n.channel;
        
        if (immediateMode) {
            // Send immediately: respect relative timing but start from sample 0
            // n.startBeat is already relative (starts at 0) when immediate=true
            if (!n.noteOnSent) {
                int noteOnSample = std::clamp(int(n.startBeat * samplesPerBeat), 0, buffer.getNumSamples() - 1);
                midiMessages.addEvent(juce::MidiMessage::noteOn(channel, n.pitch, (juce::uint8)n.velocity), noteOnSample);
                n.noteOnSent = true;
            }
            // Schedule note-off based on endBeat (also relative)
            if (n.noteOnSent && !n.noteOffSent) {
                int noteOffSample = int(n.endBeat * samplesPerBeat);
                if (noteOffSample >= 0 && noteOffSample < buffer.getNumSamples()) {
                    midiMessages.addEvent(juce::MidiMessage::noteOff(channel, n.pitch), noteOffSample);
                    n.noteOffSent = true;
                }
                // If note-off is beyond this buffer, it will be sent in a later buffer
            }
        } else if (isPlaying_ || hostIsPlaying) {
            // Normal playback mode: sync to playhead
            // Note On
            if (!n.noteOnSent && n.startBeat >= hostBeat && n.startBeat < blockEnd) {
                int offset = std::clamp(int((n.startBeat - hostBeat) * samplesPerBeat), 0, buffer.getNumSamples() - 1);
                midiMessages.addEvent(juce::MidiMessage::noteOn(channel, n.pitch, (juce::uint8)n.velocity), offset);
                n.noteOnSent = true;
            }
            // Note Off
            if (n.noteOnSent && !n.noteOffSent && n.endBeat >= hostBeat && n.endBeat < blockEnd) {
                int offset = std::clamp(int((n.endBeat - hostBeat) * samplesPerBeat), 0, buffer.getNumSamples() - 1);
                midiMessages.addEvent(juce::MidiMessage::noteOff(channel, n.pitch), offset);
                n.noteOffSent = true;
            }
        }
    }
}

void PluginProcessor::scheduleNotesForPlayback(bool immediate) {
    std::lock_guard<std::mutex> lock(notesMutex_);
    scheduledNotes_.clear();
    
    double startBeat = 0.0;
    if (!immediate) {
        // For normal playback, start from current playhead position
        if (auto* playHead = getPlayHead()) {
            if (auto pos = playHead->getPosition()) {
                if (auto p = pos->getPpqPosition()) startBeat = *p;
            }
        }
    }
    // For immediate mode, startBeat stays at 0.0
    
    // Schedule chord notes (channel 1)
    for (const auto& chord : lastGenerated_.chords) {
        if (chord.pitches.empty()) continue;
        for (int pitch : chord.pitches) {
            scheduledNotes_.push_back({pitch, 80, startBeat + chord.startBeat, 
                startBeat + chord.startBeat + chord.duration * 0.9, 1, false, false});
        }
    }
    
    // Schedule melody notes (channel 2)
    for (const auto& note : lastGenerated_.melody) {
        scheduledNotes_.push_back({note.pitch, note.velocity, startBeat + note.startBeat,
            startBeat + note.startBeat + note.duration * 0.9, 2, false, false});
    }
    
    // Schedule bass notes (channel 3)
    for (const auto& note : lastGenerated_.bass) {
        scheduledNotes_.push_back({note.pitch, note.velocity, startBeat + note.startBeat,
            startBeat + note.startBeat + note.duration * 0.9, 3, false, false});
    }
}

void PluginProcessor::startPlayback() {
    scheduleNotesForPlayback(false);  // Normal playback from playhead
    isPlaying_ = true;
    sendImmediately_ = false;
}

void PluginProcessor::stopPlayback() { 
    isPlaying_ = false; 
}

GeneratedMidi PluginProcessor::generateFromWound(const std::string& description, float intensity) {
    Wound wound{description, intensity, "user_input"};
    IntentResult intent = intentPipeline_.process(wound);
    
    // Get fine-tuning parameters
    float complexity = *parameters.getRawParameterValue(PARAM_COMPLEXITY);
    float humanize = *parameters.getRawParameterValue(PARAM_HUMANIZE);
    int bars = (int)*parameters.getRawParameterValue(PARAM_BARS);
    
    // Use full MidiGenerator with all modules
    MidiGenerator generator;
    lastGenerated_ = generator.generate(intent, bars, complexity, humanize);
    
    scheduleNotesForPlayback(true);  // Schedule for immediate playback
    sendImmediately_ = true;  // Send immediately on next processBlock
    isPlaying_ = true;
    return lastGenerated_;
}

GeneratedMidi PluginProcessor::generateFromEmotion(const std::string& emotionDescription, float intensity) {
    Wound wound{emotionDescription, intensity, "ui"};
    IntentResult intent = intentPipeline_.process(wound);
    
    // Get fine-tuning parameters
    float complexity = *parameters.getRawParameterValue(PARAM_COMPLEXITY);
    float humanize = *parameters.getRawParameterValue(PARAM_HUMANIZE);
    float feel = *parameters.getRawParameterValue(PARAM_FEEL);
    float dynamics = *parameters.getRawParameterValue(PARAM_DYNAMICS);
    int bars = (int)*parameters.getRawParameterValue(PARAM_BARS);
    
    // Use full MidiGenerator with all modules
    MidiGenerator generator;
    lastGenerated_ = generator.generate(intent, bars, complexity, humanize, feel, dynamics);
    
    scheduleNotesForPlayback(true);  // Schedule for immediate playback
    sendImmediately_ = true;  // Send immediately on next processBlock
    isPlaying_ = true;
    return lastGenerated_;
}

bool PluginProcessor::exportMidiToFile(const juce::File& file) {
    if (lastGenerated_.chords.empty()) return false;
    
    MidiBuilder builder;
    auto midiFile = builder.buildMidiFile(lastGenerated_);
    
    juce::FileOutputStream stream(file);
    if (stream.openedOk()) {
        midiFile.writeTo(stream);
        return true;
    }
    return false;
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = parameters.copyState();
    
    // Add emotion-specific state
    state.setProperty("lastEmotionId", (int)*parameters.getRawParameterValue(PARAM_EMOTION_ID), nullptr);
    state.setProperty("lastValence", (double)*parameters.getRawParameterValue(PARAM_VALENCE), nullptr);
    state.setProperty("lastArousal", (double)*parameters.getRawParameterValue(PARAM_AROUSAL), nullptr);
    state.setProperty("lastIntensity", (double)*parameters.getRawParameterValue(PARAM_INTENSITY), nullptr);
    state.setProperty("currentProgram", currentProgram_, nullptr);
    
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

juce::File PluginProcessor::getPresetDirectory() const {
    auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
    auto presetDir = appDataDir.getChildFile("Kelly MIDI Companion").getChildFile("Presets");
    presetDir.createDirectory();
    return presetDir;
}

bool PluginProcessor::exportPreset(const juce::File& file) const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    
    // Export all current parameter values
    obj->setProperty("valence", *parameters.getRawParameterValue(PARAM_VALENCE));
    obj->setProperty("arousal", *parameters.getRawParameterValue(PARAM_AROUSAL));
    obj->setProperty("intensity", *parameters.getRawParameterValue(PARAM_INTENSITY));
    obj->setProperty("complexity", *parameters.getRawParameterValue(PARAM_COMPLEXITY));
    obj->setProperty("humanize", *parameters.getRawParameterValue(PARAM_HUMANIZE));
    obj->setProperty("feel", *parameters.getRawParameterValue(PARAM_FEEL));
    obj->setProperty("dynamics", *parameters.getRawParameterValue(PARAM_DYNAMICS));
    obj->setProperty("bars", *parameters.getRawParameterValue(PARAM_BARS));
    obj->setProperty("emotionId", (int)*parameters.getRawParameterValue(PARAM_EMOTION_ID));
    obj->setProperty("currentProgram", currentProgram_);
    obj->setProperty("version", "1.1.0");
    
    juce::var root(obj);
    juce::String json = juce::JSON::toString(root, true);
    
    juce::FileOutputStream stream(file);
    if (stream.openedOk()) {
        stream.writeText(json, false, false, "\n");
        return true;
    }
    return false;
}

bool PluginProcessor::importPreset(const juce::File& file) {
    if (!file.existsAsFile()) return false;
    
    juce::String json = file.loadFileAsString();
    juce::var root = juce::JSON::parse(json);
    
    if (!root.isObject()) return false;
    
    auto* obj = root.getDynamicObject();
    if (!obj) return false;
    
    // Import parameter values
    if (obj->hasProperty("valence"))
        *parameters.getRawParameterValue(PARAM_VALENCE) = (float)obj->getProperty("valence");
    if (obj->hasProperty("arousal"))
        *parameters.getRawParameterValue(PARAM_AROUSAL) = (float)obj->getProperty("arousal");
    if (obj->hasProperty("intensity"))
        *parameters.getRawParameterValue(PARAM_INTENSITY) = (float)obj->getProperty("intensity");
    if (obj->hasProperty("complexity"))
        *parameters.getRawParameterValue(PARAM_COMPLEXITY) = (float)obj->getProperty("complexity");
    if (obj->hasProperty("humanize"))
        *parameters.getRawParameterValue(PARAM_HUMANIZE) = (float)obj->getProperty("humanize");
    if (obj->hasProperty("feel"))
        *parameters.getRawParameterValue(PARAM_FEEL) = (float)obj->getProperty("feel");
    if (obj->hasProperty("dynamics"))
        *parameters.getRawParameterValue(PARAM_DYNAMICS) = (float)obj->getProperty("dynamics");
    if (obj->hasProperty("bars"))
        *parameters.getRawParameterValue(PARAM_BARS) = (float)obj->getProperty("bars");
    if (obj->hasProperty("emotionId"))
        *parameters.getRawParameterValue(PARAM_EMOTION_ID) = (float)(int)obj->getProperty("emotionId");
    if (obj->hasProperty("currentProgram"))
        setCurrentProgram((int)obj->getProperty("currentProgram"));
    
    return true;
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(parameters.state.getType())) {
        auto state = juce::ValueTree::fromXml(*xml);
        
        // Restore emotion-specific state (these are already in the parameter tree, but we save them explicitly for clarity)
        // The parameters will be restored automatically by replaceState, but we can add custom properties if needed
        
        parameters.replaceState(state);
    }
}

void PluginProcessor::setVoiceSynthesisEnabled(bool enabled) {
    voiceSynthesisEnabled_ = enabled;
    if (voiceSynthesizer_) {
        voiceSynthesizer_->setEnabled(enabled);
    }
}

void PluginProcessor::setBiometricInputEnabled(bool enabled) {
    biometricInputEnabled_ = enabled;
    if (biometricInput_) {
        biometricInput_->setEnabled(enabled);
    }
}

void PluginProcessor::processBiometricData(const BiometricInput::BiometricData& data) {
    if (!biometricInputEnabled_ || !biometricInput_) return;
    
    auto emotionParams = biometricInput_->processBiometricData(data);
    
    // Update emotion parameters based on biometric data
    *parameters.getRawParameterValue(PARAM_VALENCE) = emotionParams.valence;
    *parameters.getRawParameterValue(PARAM_AROUSAL) = emotionParams.arousal;
    *parameters.getRawParameterValue(PARAM_INTENSITY) = emotionParams.intensity;
}

} // namespace kelly

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
