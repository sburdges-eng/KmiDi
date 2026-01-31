#pragma once

#include "common/Types.h"
#include <juce_audio_basics/juce_audio_basics.h>

namespace kelly {

/**
 * Builds JUCE MIDI objects from Kelly's internal representations.
 */
class MidiBuilder {
public:
    /** Build a MidiFile for export/drag-drop */
    juce::MidiFile buildMidiFile(const GeneratedMidi& midi);
    
    /** Build a MidiBuffer for real-time output */
    juce::MidiBuffer buildMidiBuffer(const GeneratedMidi& midi, 
                                      double sampleRate, 
                                      float bpm);
    
    /** Convert a single chord to MIDI messages */
    void addChordToSequence(juce::MidiMessageSequence& seq,
                            const Chord& chord,
                            int ticksPerBeat,
                            int velocity = 80);
    
    /** Convert melody notes to MIDI messages */
    void addNotesToSequence(juce::MidiMessageSequence& seq,
                            const std::vector<MidiNote>& notes,
                            int ticksPerBeat);
    
private:
    static constexpr int DEFAULT_TICKS_PER_BEAT = 480;
};

} // namespace kelly
