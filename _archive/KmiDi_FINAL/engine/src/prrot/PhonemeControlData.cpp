#include "prrot/PhonemeControlData.h"
#include <algorithm>
#include <cmath>

namespace prrot {

// =============================================================================
// AutomationEnvelope Implementation
// =============================================================================

float AutomationEnvelope::getValueAt(float time_ms) const {
    if (points.empty()) {
        return 0.0f;
    }

    // Find the two points that bracket this time
    for (size_t i = 0; i < points.size() - 1; ++i) {
        if (time_ms >= points[i].time_ms && time_ms <= points[i + 1].time_ms) {
            const auto& p1 = points[i];
            const auto& p2 = points[i + 1];

            if (interpolation == Interpolation::Step) {
                return p1.value; // Hold first value until next point
            } else if (interpolation == Interpolation::Linear) {
                // Linear interpolation
                float t = (time_ms - p1.time_ms) / (p2.time_ms - p1.time_ms);
                return p1.value + t * (p2.value - p1.value);
            } else { // Smooth
                // Smooth interpolation (ease-in-out curve)
                float t = (time_ms - p1.time_ms) / (p2.time_ms - p1.time_ms);
                float t_smooth = t * t * (3.0f - 2.0f * t); // Smoothstep
                return p1.value + t_smooth * (p2.value - p1.value);
            }
        }
    }

    // Extrapolate: before first point or after last point
    if (time_ms < points.front().time_ms) {
        return points.front().value;
    } else {
        return points.back().value;
    }
}

void AutomationEnvelope::addPoint(float time_ms, float value) {
    AutomationPoint point;
    point.time_ms = time_ms;
    point.value = value;

    points.push_back(point);

    // Keep points sorted by time
    std::sort(points.begin(), points.end(),
              [](const AutomationPoint& a, const AutomationPoint& b) {
                  return a.time_ms < b.time_ms;
              });
}

void AutomationEnvelope::normalize() {
    if (points.empty()) {
        return;
    }

    // Find min and max values
    float min_val = points[0].value;
    float max_val = points[0].value;

    for (const auto& point : points) {
        min_val = std::min(min_val, point.value);
        max_val = std::max(max_val, point.value);
    }

    // Normalize to [0, 1] range
    float range = max_val - min_val;
    if (range > 0.001f) { // Avoid division by zero
        for (auto& point : points) {
            point.value = (point.value - min_val) / range;
        }
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

float midiNoteToFrequency(int midi_note, float cents_offset) {
    // MIDI note 69 = A4 = 440 Hz
    float semitones = static_cast<float>(midi_note - 69) + (cents_offset / 100.0f);
    return 440.0f * std::pow(2.0f, semitones / 12.0f);
}

std::pair<int, float> frequencyToMidiNote(float frequency_hz) {
    if (frequency_hz <= 0.0f) {
        return {60, 0.0f}; // Default to middle C
    }

    // MIDI note 69 = A4 = 440 Hz
    float semitones = 12.0f * std::log2(frequency_hz / 440.0f) + 69.0f;

    int midi_note = static_cast<int>(std::round(semitones));
    float cents = (semitones - midi_note) * 100.0f;

    // Clamp MIDI note to valid range
    midi_note = std::clamp(midi_note, 0, 127);
    cents = std::clamp(cents, -50.0f, 50.0f);

    return {midi_note, cents};
}

uint64_t msToSamples(float time_ms, float sample_rate_hz) {
    return static_cast<uint64_t>(std::round(time_ms * sample_rate_hz / 1000.0f));
}

float samplesToMs(uint64_t samples, float sample_rate_hz) {
    return static_cast<float>(samples) * 1000.0f / sample_rate_hz;
}

} // namespace prrot
