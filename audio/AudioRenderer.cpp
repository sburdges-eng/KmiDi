#include "audio/AudioRenderer.h"
#include <cmath>
#include <algorithm>

namespace kelly {

namespace {

constexpr double PI = 3.14159265358979323846;

double midiToHz(int midiNote) {
    return 440.0 * std::pow(2.0, (midiNote - 69) / 12.0);
}

void addNoteToBuffer(std::vector<float>& buffer,
                     double sampleRate,
                     size_t startSample,
                     size_t durationSamples,
                     int pitch,
                     float gain) {
    if (pitch < 0 || pitch > 127 || startSample >= buffer.size() || durationSamples == 0) {
        return;
    }
    const double freq = midiToHz(pitch);
    for (size_t i = 0; i < durationSamples && (startSample + i) < buffer.size(); ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        float env = 1.0f;
        if (i < static_cast<size_t>(0.01 * sampleRate)) {
            env = static_cast<float>(i) / (0.01f * static_cast<float>(sampleRate));
        } else if (i > durationSamples - static_cast<size_t>(0.05 * sampleRate)) {
            env = static_cast<float>(durationSamples - i) / (0.05f * static_cast<float>(sampleRate));
        }
        buffer[startSample + i] += gain * env * static_cast<float>(std::sin(2.0 * PI * freq * t));
    }
}

} // namespace

std::vector<float> renderMidiToPcm(const ValidatedOutputPlan& outputPlan, double sampleRate) {
    const auto& midi = outputPlan.midi();
    if (midi.chords.empty() || sampleRate <= 0)
        return {};

    // Clamp inputs to prevent unbounded allocation
    const double clampedLength = std::min(midi.lengthInBeats, kMaxLengthInBeats);
    const double clampedBpm = std::clamp(static_cast<double>(midi.bpm), 20.0, 300.0);
    const double beatsPerSecond = clampedBpm / 60.0;
    const double samplesPerBeat = sampleRate / beatsPerSecond;
    const size_t totalSamples = std::min(
        static_cast<size_t>(std::ceil(clampedLength * samplesPerBeat)),
        kMaxPcmBufferSamples);

    std::vector<float> buffer(totalSamples, 0.0f);

    for (size_t chordIdx = 0; chordIdx < midi.chords.size(); ++chordIdx) {
        const auto& chord = midi.chords[chordIdx];
        const size_t startSample = static_cast<size_t>(
            chord.startBeat * samplesPerBeat);
        const size_t durationSamples = static_cast<size_t>(
            chord.duration * samplesPerBeat * 0.9);
        int velocity = 80;
        if (midi.chordVelocities.has_value() && chordIdx < midi.chordVelocities->size()) {
            velocity = std::clamp((*midi.chordVelocities)[chordIdx], 1, 127);
        }
        const float velocityGain = static_cast<float>(velocity) / 127.0f;
        const float gain = (0.2f * velocityGain) / std::max(1, static_cast<int>(chord.pitches.size()));

        for (int pitch : chord.pitches) {
            addNoteToBuffer(buffer, sampleRate, startSample, durationSamples, pitch, gain);
        }
    }

    if (midi.melody.has_value()) {
        for (const auto& note : *midi.melody) {
            const size_t startSample = static_cast<size_t>(note.startBeat * samplesPerBeat);
            const size_t durationSamples = static_cast<size_t>(note.duration * samplesPerBeat * 0.95);
            const float gain = 0.12f * (static_cast<float>(std::clamp(note.velocity, 1, 127)) / 127.0f);
            addNoteToBuffer(buffer, sampleRate, startSample, durationSamples, note.pitch, gain);
        }
    }
    if (midi.bass.has_value()) {
        for (const auto& note : *midi.bass) {
            const size_t startSample = static_cast<size_t>(note.startBeat * samplesPerBeat);
            const size_t durationSamples = static_cast<size_t>(note.duration * samplesPerBeat * 0.95);
            const float gain = 0.16f * (static_cast<float>(std::clamp(note.velocity, 1, 127)) / 127.0f);
            addNoteToBuffer(buffer, sampleRate, startSample, durationSamples, note.pitch, gain);
        }
    }

    float peak = 0.0f;
    for (float s : buffer)
        peak = std::max(peak, std::abs(s));
    if (peak > 0.001f) {
        const float norm = 0.8f / peak;
        for (float& s : buffer)
            s *= norm;
    }

    return buffer;
}

} // namespace kelly
