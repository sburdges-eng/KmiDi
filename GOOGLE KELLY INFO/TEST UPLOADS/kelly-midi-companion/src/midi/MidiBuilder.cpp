#include "midi/MidiBuilder.h"

namespace kelly {

juce::MidiFile MidiBuilder::buildMidiFile(const GeneratedMidi& midi) {
    juce::MidiFile file;
    file.setTicksPerQuarterNote(DEFAULT_TICKS_PER_BEAT);
    
    // Track 0: Tempo and time signature
    juce::MidiMessageSequence metaTrack;
    metaTrack.addEvent(juce::MidiMessage::timeSignatureMetaEvent(4, 4));
    metaTrack.addEvent(juce::MidiMessage::tempoMetaEvent(
        static_cast<int>(60000000.0 / midi.bpm)));  // Microseconds per beat
    file.addTrack(metaTrack);
    
    // Track 1: Chords
    juce::MidiMessageSequence chordTrack;
    for (const auto& chord : midi.chords) {
        addChordToSequence(chordTrack, chord, DEFAULT_TICKS_PER_BEAT);
    }
    chordTrack.updateMatchedPairs();
    file.addTrack(chordTrack);
    
    // Track 2: Melody (if present)
    if (!midi.melody.empty()) {
        juce::MidiMessageSequence melodyTrack;
        addNotesToSequence(melodyTrack, midi.melody, DEFAULT_TICKS_PER_BEAT);
        melodyTrack.updateMatchedPairs();
        file.addTrack(melodyTrack);
    }
    
    // Track 3: Bass (if present)
    if (!midi.bass.empty()) {
        juce::MidiMessageSequence bassTrack;
        addNotesToSequence(bassTrack, midi.bass, DEFAULT_TICKS_PER_BEAT);
        bassTrack.updateMatchedPairs();
        file.addTrack(bassTrack);
    }
    
    return file;
}

juce::MidiBuffer MidiBuilder::buildMidiBuffer(const GeneratedMidi& midi,
                                               double sampleRate,
                                               float bpm) {
    juce::MidiBuffer buffer;
    
    double samplesPerBeat = (sampleRate * 60.0) / bpm;
    
    for (const auto& chord : midi.chords) {
        int startSample = static_cast<int>(chord.startBeat * samplesPerBeat);
        int endSample = static_cast<int>((chord.startBeat + chord.duration) * samplesPerBeat);
        
        for (int pitch : chord.pitches) {
            buffer.addEvent(
                juce::MidiMessage::noteOn(1, pitch, static_cast<juce::uint8>(80)),
                startSample
            );
            buffer.addEvent(
                juce::MidiMessage::noteOff(1, pitch),
                endSample
            );
        }
    }
    
    for (const auto& note : midi.melody) {
        int startSample = static_cast<int>(note.startBeat * samplesPerBeat);
        int endSample = static_cast<int>((note.startBeat + note.duration) * samplesPerBeat);
        
        buffer.addEvent(
            juce::MidiMessage::noteOn(2, note.pitch, static_cast<juce::uint8>(note.velocity)),
            startSample
        );
        buffer.addEvent(
            juce::MidiMessage::noteOff(2, note.pitch),
            endSample
        );
    }
    
    return buffer;
}

void MidiBuilder::addChordToSequence(juce::MidiMessageSequence& seq,
                                      const Chord& chord,
                                      int ticksPerBeat,
                                      int velocity) {
    int startTick = static_cast<int>(chord.startBeat * ticksPerBeat);
    int endTick = static_cast<int>((chord.startBeat + chord.duration) * ticksPerBeat);
    
    for (int pitch : chord.pitches) {
        seq.addEvent(juce::MidiMessage::noteOn(1, pitch, static_cast<juce::uint8>(velocity)), 
                     startTick);
        seq.addEvent(juce::MidiMessage::noteOff(1, pitch), 
                     endTick);
    }
}

void MidiBuilder::addNotesToSequence(juce::MidiMessageSequence& seq,
                                      const std::vector<MidiNote>& notes,
                                      int ticksPerBeat) {
    for (const auto& note : notes) {
        int startTick = static_cast<int>(note.startBeat * ticksPerBeat);
        int endTick = static_cast<int>((note.startBeat + note.duration) * ticksPerBeat);
        
        seq.addEvent(
            juce::MidiMessage::noteOn(1, note.pitch, static_cast<juce::uint8>(note.velocity)),
            startTick
        );
        seq.addEvent(
            juce::MidiMessage::noteOff(1, note.pitch),
            endTick
        );
    }
}

} // namespace kelly
