#pragma once

#include "common/Types.h"
#include <juce_audio_formats/juce_audio_formats.h>

namespace kelly {

/**
 * Builds JUCE MidiFile objects from GeneratedMidi structures.
 */
class MidiBuilder {
public:
    MidiBuilder();
    ~MidiBuilder() = default;
    
    /** Build a JUCE MidiFile from a validated output plan. */
    juce::MidiFile buildMidiFile(const ValidatedOutputPlan& outputPlan);
    
private:
    /** Add chord to MIDI track */
    void addChordToTrack(juce::MidiMessageSequence& track, const Chord& chord, double startTime,
                         int channel, double secondsPerBeat, int velocity);
    
    /** Add note to MIDI track */
    void addNoteToTrack(juce::MidiMessageSequence& track, const MidiNote& note, double startTime,
                        int channel, double secondsPerBeat);
    
    /** Convert beats to MIDI ticks */
    int beatsToTicks(double beats, double ticksPerQuarterNote);
    
    static double sanitizeDurationBeats(double durationBeats);
};

} // namespace kelly
