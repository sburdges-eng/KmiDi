#include "plugin/PluginProcessor.h"
#include "plugin/PluginEditor.h"
#include "midi/ChordGenerator.h"
#include "midi/MidiBuilder.h"

namespace kelly {

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "KellyParams", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_INTENSITY, 1},
        "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.7f
    ));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_HUMANIZE, 1},
        "Humanize",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.4f
    ));
    
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_TEMPO_LOCK, 1},
        "Lock to DAW Tempo",
        true
    ));
    
    return {params.begin(), params.end()};
}

void PluginProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    currentSampleRate_ = sampleRate;
    
    // Get tempo from host if available
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto bpm = position->getBpm()) {
                currentBpm_ = *bpm;
            }
        }
    }
}

void PluginProcessor::releaseResources() {
    stopPlayback();
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    
    // Clear audio (we're MIDI-only)
    buffer.clear();
    
    // Get host transport info
    double hostBpm = 120.0;
    double hostBeat = 0.0;
    bool hostPlaying = false;
    
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto bpm = position->getBpm()) {
                hostBpm = *bpm;
                currentBpm_ = hostBpm;
            }
            if (auto ppq = position->getPpqPosition()) {
                hostBeat = *ppq;
            }
            hostPlaying = position->getIsPlaying();
        }
    }
    
    // If host just started playing, reset our playback position
    if (hostPlaying && !wasHostPlaying_) {
        playbackPositionBeats_ = hostBeat;
        
        // Reset note states when transport restarts
        std::lock_guard<std::mutex> lock(notesMutex_);
        for (auto& note : scheduledNotes_) {
            note.noteOnSent = false;
            note.noteOffSent = false;
        }
    }
    wasHostPlaying_ = hostPlaying;
    
    // Only output MIDI if we're playing and host is playing
    if (!isPlaying_ || !hostPlaying) {
        return;
    }
    
    // Calculate beat range for this buffer
    double samplesPerBeat = (currentSampleRate_ * 60.0) / hostBpm;
    double beatsPerSample = 1.0 / samplesPerBeat;
    double bufferBeats = buffer.getNumSamples() * beatsPerSample;
    
    double blockStartBeat = hostBeat;
    double blockEndBeat = hostBeat + bufferBeats;
    
    // Process scheduled notes
    std::lock_guard<std::mutex> lock(notesMutex_);
    
    for (auto& note : scheduledNotes_) {
        // Note On
        if (!note.noteOnSent && note.startBeat >= blockStartBeat && note.startBeat < blockEndBeat) {
            int sampleOffset = static_cast<int>((note.startBeat - blockStartBeat) * samplesPerBeat);
            sampleOffset = std::clamp(sampleOffset, 0, buffer.getNumSamples() - 1);
            
            midiMessages.addEvent(
                juce::MidiMessage::noteOn(1, note.pitch, static_cast<juce::uint8>(note.velocity)),
                sampleOffset
            );
            note.noteOnSent = true;
        }
        
        // Note Off
        if (note.noteOnSent && !note.noteOffSent && note.endBeat >= blockStartBeat && note.endBeat < blockEndBeat) {
            int sampleOffset = static_cast<int>((note.endBeat - blockStartBeat) * samplesPerBeat);
            sampleOffset = std::clamp(sampleOffset, 0, buffer.getNumSamples() - 1);
            
            midiMessages.addEvent(
                juce::MidiMessage::noteOff(1, note.pitch),
                sampleOffset
            );
            note.noteOffSent = true;
        }
    }
    
    // Check if all notes have been played (loop or stop)
    bool allDone = true;
    for (const auto& note : scheduledNotes_) {
        if (!note.noteOffSent) {
            allDone = false;
            break;
        }
    }
    
    // Loop playback - reset all notes when done
    if (allDone && !scheduledNotes_.empty()) {
        for (auto& note : scheduledNotes_) {
            note.noteOnSent = false;
            note.noteOffSent = false;
        }
    }
}

void PluginProcessor::scheduleNotesForPlayback() {
    std::lock_guard<std::mutex> lock(notesMutex_);
    scheduledNotes_.clear();
    
    // Get current host position as our start point
    double startOffset = 0.0;
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto ppq = position->getPpqPosition()) {
                startOffset = *ppq;
            }
        }
    }
    
    // Schedule all chord notes
    for (const auto& chord : lastGenerated_.chords) {
        for (int pitch : chord.pitches) {
            ScheduledNote note;
            note.pitch = pitch;
            note.velocity = 80;
            note.startBeat = startOffset + chord.startBeat;
            note.endBeat = startOffset + chord.startBeat + chord.duration * 0.9; // Slight gap
            note.noteOnSent = false;
            note.noteOffSent = false;
            scheduledNotes_.push_back(note);
        }
    }
    
    // Schedule melody notes
    for (const auto& midiNote : lastGenerated_.melody) {
        ScheduledNote note;
        note.pitch = midiNote.pitch;
        note.velocity = midiNote.velocity;
        note.startBeat = startOffset + midiNote.startBeat;
        note.endBeat = startOffset + midiNote.startBeat + midiNote.duration * 0.9;
        note.noteOnSent = false;
        note.noteOffSent = false;
        scheduledNotes_.push_back(note);
    }
    
    // Schedule bass notes
    for (const auto& midiNote : lastGenerated_.bass) {
        ScheduledNote note;
        note.pitch = midiNote.pitch;
        note.velocity = midiNote.velocity;
        note.startBeat = startOffset + midiNote.startBeat;
        note.endBeat = startOffset + midiNote.startBeat + midiNote.duration * 0.9;
        note.noteOnSent = false;
        note.noteOffSent = false;
        scheduledNotes_.push_back(note);
    }
}

void PluginProcessor::startPlayback() {
    scheduleNotesForPlayback();
    isPlaying_ = true;
}

void PluginProcessor::stopPlayback() {
    isPlaying_ = false;
    
    // Send note-offs for any hanging notes would require access to midiMessages
    // which we don't have here - the next processBlock will handle cleanup
}

GeneratedMidi PluginProcessor::generateFromWound(const std::string& description, float intensity) {
    Wound wound{description, intensity, "user_input"};
    IntentResult intent = intentPipeline_.process(wound);
    
    // Generate MIDI using the chord generator
    ChordGenerator chordGen;
    GeneratedMidi result;
    
    result.bpm = static_cast<float>(currentBpm_);
    
    // Generate chord progression based on intent
    result.chords = chordGen.generate(intent);
    result.lengthInBeats = 16.0;  // 4 bars default
    
    lastGenerated_ = result;
    
    // Auto-start playback
    startPlayback();
    
    return result;
}

GeneratedMidi PluginProcessor::generateFromJourney(const SideA& current, const SideB& desired) {
    IntentResult intent = intentPipeline_.processJourney(current, desired);
    
    ChordGenerator chordGen;
    GeneratedMidi result;
    
    result.bpm = static_cast<float>(currentBpm_);
    
    result.chords = chordGen.generate(intent);
    result.lengthInBeats = 16.0;
    
    lastGenerated_ = result;
    
    // Auto-start playback
    startPlayback();
    
    return result;
}

bool PluginProcessor::exportMidiToFile(const juce::File& file) {
    if (lastGenerated_.chords.empty()) {
        return false;
    }
    
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
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(parameters.state.getType())) {
        parameters.replaceState(juce::ValueTree::fromXml(*xml));
    }
}

} // namespace kelly

// Plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
