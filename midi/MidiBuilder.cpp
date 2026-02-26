#include "midi/MidiBuilder.h"
#include <algorithm>
#include <cmath>

namespace kelly {

MidiBuilder::MidiBuilder() {
}

double MidiBuilder::sanitizeDurationBeats(double durationBeats) {
    if (!std::isfinite(durationBeats) || durationBeats <= 0.0) {
        return 1.0 / 480.0;
    }
    return durationBeats;
}

juce::MidiFile MidiBuilder::buildMidiFile(const ValidatedOutputPlan& outputPlan) {
    const auto& generated = outputPlan.midi();
    juce::MidiFile midiFile;
    double ticksPerQuarterNote = 480.0;
    midiFile.setTicksPerQuarterNote(static_cast<int>(ticksPerQuarterNote));
    const double bpm = (std::isfinite(generated.bpm) && generated.bpm > 0.0f)
        ? static_cast<double>(generated.bpm)
        : 120.0;
    double secondsPerBeat = 60.0 / bpm;
    double microsecondsPerQuarterNote = 60.0 * 1000000.0 / bpm;

    juce::MidiMessageSequence chordTrack;
    chordTrack.addEvent(juce::MidiMessage::timeSignatureMetaEvent(4, 4), 0);
    chordTrack.addEvent(juce::MidiMessage::tempoMetaEvent(static_cast<int>(microsecondsPerQuarterNote)), 0);
    for (size_t i = 0; i < generated.chords.size(); ++i) {
        const auto& chord = generated.chords[i];
        int velocity = 80;
        if (generated.chordVelocities.has_value() && i < generated.chordVelocities->size()) {
            velocity = std::clamp((*generated.chordVelocities)[i], 1, 127);
        }
        double startTime = chord.startBeat * secondsPerBeat;
        addChordToTrack(chordTrack, chord, startTime, 1, secondsPerBeat, velocity);
    }
    midiFile.addTrack(chordTrack);

    // LISTENING_GUARDRAIL: Absence must be explicit. Only add melody/bass when set (I-1).
    if (generated.melody.has_value()) {
        juce::MidiMessageSequence melodyTrack;
        melodyTrack.addEvent(juce::MidiMessage::timeSignatureMetaEvent(4, 4), 0);
        melodyTrack.addEvent(juce::MidiMessage::tempoMetaEvent(static_cast<int>(microsecondsPerQuarterNote)), 0);
        for (const auto& note : *generated.melody) {
            double startTime = note.startBeat * secondsPerBeat;
            addNoteToTrack(melodyTrack, note, startTime, 2, secondsPerBeat);
        }
        midiFile.addTrack(melodyTrack);
    }
    if (generated.bass.has_value()) {
        juce::MidiMessageSequence bassTrack;
        bassTrack.addEvent(juce::MidiMessage::timeSignatureMetaEvent(4, 4), 0);
        bassTrack.addEvent(juce::MidiMessage::tempoMetaEvent(static_cast<int>(microsecondsPerQuarterNote)), 0);
        for (const auto& note : *generated.bass) {
            double startTime = note.startBeat * secondsPerBeat;
            addNoteToTrack(bassTrack, note, startTime, 3, secondsPerBeat);
        }
        midiFile.addTrack(bassTrack);
    }

    return midiFile;
}

void MidiBuilder::addChordToTrack(juce::MidiMessageSequence& track, const Chord& chord, double startTime,
                                  int channel, double secondsPerBeat, int velocity) {
    const double normalizedDuration = sanitizeDurationBeats(chord.duration);
    double endTime = startTime + (normalizedDuration * secondsPerBeat);
    const int clampedVelocity = clampMidiVelocity(velocity);
    
    for (int pitch : chord.pitches) {
        const int clampedPitch = clampMidiPitch(pitch);
        // Note on
        juce::MidiMessage noteOn =
            juce::MidiMessage::noteOn(channel, clampedPitch, static_cast<juce::uint8>(clampedVelocity));
        track.addEvent(noteOn, startTime);
        
        // Note off
        juce::MidiMessage noteOff = juce::MidiMessage::noteOff(channel, clampedPitch);
        track.addEvent(noteOff, endTime);
    }
}

void MidiBuilder::addNoteToTrack(juce::MidiMessageSequence& track, const MidiNote& note, double startTime,
                                 int channel, double secondsPerBeat) {
    const double normalizedDuration = sanitizeDurationBeats(note.duration);
    double durationSeconds = normalizedDuration * secondsPerBeat;
    double endTime = startTime + durationSeconds;
    const int clampedPitch = clampMidiPitch(note.pitch);
    const int clampedVelocity = clampMidiVelocity(note.velocity);
    
    // Note on
    juce::MidiMessage noteOn =
        juce::MidiMessage::noteOn(channel, clampedPitch, static_cast<juce::uint8>(clampedVelocity));
    track.addEvent(noteOn, startTime);
    
    // Note off
    juce::MidiMessage noteOff = juce::MidiMessage::noteOff(channel, clampedPitch);
    track.addEvent(noteOff, endTime);
}

int MidiBuilder::beatsToTicks(double beats, double ticksPerQuarterNote) {
    return static_cast<int>(beats * ticksPerQuarterNote);
}

} // namespace kelly
