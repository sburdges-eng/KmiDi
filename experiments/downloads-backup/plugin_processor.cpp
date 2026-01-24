#include "plugin_processor.h"
#include "plugin_editor.h"

namespace kelly {

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)) {
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;
    playbackPosition = 0;
}

void PluginProcessor::releaseResources() {
    stopPlayback();
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    buffer.clear();
    
    if (isPlaying.load()) {
        processGeneratedMidi(midiMessages, buffer.getNumSamples());
    }
}

void PluginProcessor::processGeneratedMidi(juce::MidiBuffer& midiMessages, int numSamples) {
    if (generatedNotes.empty()) return;
    
    auto tempoVal = tempo.load();
    auto barsVal = bars.load();
    
    // Calculate total length in samples
    int ticksPerBar = TICKS_PER_BEAT * 4; // Assuming 4/4
    int totalTicks = ticksPerBar * barsVal;
    double samplesPerTick = (currentSampleRate * 60.0) / (tempoVal * TICKS_PER_BEAT);
    int totalSamples = static_cast<int>(totalTicks * samplesPerTick);
    
    // Process notes for this block
    int startSample = playbackPosition;
    int endSample = startSample + numSamples;
    
    for (const auto& note : generatedNotes) {
        int noteStartSample = static_cast<int>(note.startTick * samplesPerTick);
        int noteEndSample = static_cast<int>((note.startTick + note.durationTicks) * samplesPerTick);
        
        // Note on
        if (noteStartSample >= startSample && noteStartSample < endSample) {
            int sampleOffset = noteStartSample - startSample;
            midiMessages.addEvent(juce::MidiMessage::noteOn(note.channel + 1, note.pitch, (juce::uint8)note.velocity), sampleOffset);
        }
        
        // Note off
        if (noteEndSample >= startSample && noteEndSample < endSample) {
            int sampleOffset = noteEndSample - startSample;
            midiMessages.addEvent(juce::MidiMessage::noteOff(note.channel + 1, note.pitch), sampleOffset);
        }
    }
    
    // Update position
    playbackPosition += numSamples;
    playbackProgress.store(static_cast<float>(playbackPosition) / totalSamples);
    
    // Loop or stop
    if (playbackPosition >= totalSamples) {
        playbackPosition = 0;
        // For now, loop. Could add option to stop.
    }
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    juce::ValueTree state("KellyState");
    
    state.setProperty("sideA_description", sideA.description, nullptr);
    state.setProperty("sideA_intensity", sideA.intensity, nullptr);
    state.setProperty("sideB_description", sideB.description, nullptr);
    state.setProperty("sideB_intensity", sideB.intensity, nullptr);
    
    state.setProperty("tempo", tempo.load(), nullptr);
    state.setProperty("bars", bars.load(), nullptr);
    state.setProperty("keyRoot", keyRoot, nullptr);
    state.setProperty("mode", mode, nullptr);
    state.setProperty("genre", genre, nullptr);
    
    juce::MemoryOutputStream stream(destData, false);
    state.writeToStream(stream);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    auto state = juce::ValueTree::readFromData(data, static_cast<size_t>(sizeInBytes));
    
    if (state.isValid()) {
        sideA.description = state.getProperty("sideA_description", "").toString();
        sideA.intensity = state.getProperty("sideA_intensity", 0.7f);
        sideB.description = state.getProperty("sideB_description", "").toString();
        sideB.intensity = state.getProperty("sideB_intensity", 0.5f);
        
        tempo.store(state.getProperty("tempo", 120));
        bars.store(state.getProperty("bars", 8));
        keyRoot = state.getProperty("keyRoot", "C").toString();
        mode = state.getProperty("mode", "Minor").toString();
        genre = state.getProperty("genre", "Ambient").toString();
    }
}

void PluginProcessor::generateFromEmotion() {
    // TODO: Connect to Python emotion engine or C++ implementation
    // For now, generate test pattern
    
    generatedNotes.clear();
    
    auto barsVal = bars.load();
    auto tempoVal = tempo.load();
    
    // Simple test pattern - minor chord arpeggio
    std::vector<int> chordNotes = {60, 63, 67, 72}; // C minor
    
    int ticksPerBar = TICKS_PER_BEAT * 4;
    int noteIndex = 0;
    
    for (int bar = 0; bar < barsVal; ++bar) {
        for (int beat = 0; beat < 4; ++beat) {
            int tick = bar * ticksPerBar + beat * TICKS_PER_BEAT;
            int pitch = chordNotes[noteIndex % chordNotes.size()];
            
            MidiNote note;
            note.pitch = pitch;
            note.startTick = tick;
            note.durationTicks = TICKS_PER_BEAT - 20;
            note.velocity = 80 + (noteIndex % 4) * 10;
            note.channel = 0;
            
            generatedNotes.push_back(note);
            noteIndex++;
        }
    }
    
    DBG("Generated " << generatedNotes.size() << " notes");
}

void PluginProcessor::startPlayback() {
    playbackPosition = 0;
    playbackProgress.store(0.0f);
    isPlaying.store(true);
}

void PluginProcessor::stopPlayback() {
    isPlaying.store(false);
    playbackPosition = 0;
    playbackProgress.store(0.0f);
}

bool PluginProcessor::exportMidi(const juce::File& outputFile) {
    if (generatedNotes.empty()) return false;
    
    juce::MidiFile midiFile;
    juce::MidiMessageSequence sequence;
    
    auto tempoVal = tempo.load();
    
    // Add tempo
    sequence.addEvent(juce::MidiMessage::tempoMetaEvent(static_cast<int>(60000000.0 / tempoVal)));
    
    // Add notes
    for (const auto& note : generatedNotes) {
        sequence.addEvent(juce::MidiMessage::noteOn(note.channel + 1, note.pitch, (juce::uint8)note.velocity), note.startTick);
        sequence.addEvent(juce::MidiMessage::noteOff(note.channel + 1, note.pitch), note.startTick + note.durationTicks);
    }
    
    sequence.updateMatchedPairs();
    midiFile.addTrack(sequence);
    midiFile.setTicksPerQuarterNote(TICKS_PER_BEAT);
    
    juce::FileOutputStream stream(outputFile);
    if (stream.openedOk()) {
        midiFile.writeTo(stream);
        return true;
    }
    
    return false;
}

} // namespace kelly

//=============================================================================
// JUCE PLUGIN ENTRY POINT
//=============================================================================

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
