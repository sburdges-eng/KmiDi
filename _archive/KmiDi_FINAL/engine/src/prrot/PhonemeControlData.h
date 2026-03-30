#pragma once

/**
 * PhonemeControlData.h - Output Control Data Structures
 * =====================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * Defines data structures for phoneme control data output.
 * This is the primary output format of PRROT/PARROT, containing MIDI notes,
 * phoneme timing, pitch curves, and articulation control data suitable for
 * driving DAW vocal synthesizers, formant instruments, samplers, and autotune.
 *
 * RT-Safe: Structures designed for use in real-time audio contexts with
 * fixed-size arrays where possible.
 */

#include "prrot/VoiceProfile.h"
#include <array>
#include <cstdint>
#include <vector>

namespace prrot {

// =============================================================================
// Phoneme Timing Information
// =============================================================================

struct PhonemeTiming {
    PhonemeType phoneme = PhonemeType::UNKNOWN;
    float start_time_ms = 0.0f;      // Start time relative to phrase start
    float duration_ms = 0.0f;        // Duration of phoneme
    float onset_time_ms = 0.0f;      // Onset detection time (within phoneme)
    float offset_time_ms = 0.0f;     // Offset detection time (within phoneme)

    // Stress and emphasis markers
    bool is_stressed = false;        // Primary stress marker
    bool is_emphasized = false;      // Emphasis marker
    float stress_level = 0.0f;       // Stress level [0-1]
    float prominence = 0.0f;         // Prominence value [0-1]
};

// =============================================================================
// Pitch Target
// =============================================================================

struct PitchTarget {
    float time_ms = 0.0f;            // Time relative to phrase start
    int midi_note = 60;              // MIDI note number (0-127)
    float cents_offset = 0.0f;       // Fine pitch offset in cents (-50 to +50)
    float confidence = 1.0f;         // Confidence in pitch target [0-1]
};

// =============================================================================
// Vibrato Parameters
// =============================================================================

struct VibratoParameters {
    bool enabled = false;            // Whether vibrato is active
    float rate_hz = 5.5f;            // Vibrato rate (Hz)
    float depth_cents = 15.0f;       // Vibrato depth (cents)
    float delay_ms = 50.0f;          // Delay before vibrato starts
    float fade_in_ms = 100.0f;       // Fade-in time for vibrato
    float fade_out_ms = 100.0f;      // Fade-out time for vibrato
};

// =============================================================================
// Breath Marker
// =============================================================================

struct BreathMarker {
    float time_ms = 0.0f;            // Time relative to phrase start
    float intensity = 0.5f;          // Breath intensity [0-1]
    float duration_ms = 200.0f;      // Duration of breath marker
};

// =============================================================================
// MIDI Note
// =============================================================================

struct MidiNote {
    int note_number = 60;            // MIDI note number (0-127)
    int velocity = 100;              // MIDI velocity (0-127)
    float start_time_ms = 0.0f;      // Start time relative to phrase start
    float duration_ms = 500.0f;      // Duration in milliseconds
    int channel = 0;                 // MIDI channel (0-15)

    // Articulation parameters
    float attack_time_ms = 10.0f;    // Attack time (envelope)
    float release_time_ms = 50.0f;   // Release time (envelope)
};

// =============================================================================
// Automation Envelope Point
// =============================================================================

struct AutomationPoint {
    float time_ms = 0.0f;            // Time relative to phrase start
    float value = 0.0f;              // Automation value [0-1] or [-1, 1] depending on parameter
};

// =============================================================================
// Automation Envelope
// =============================================================================

enum class AutomationType : uint8_t {
    Dynamics = 0,       // Volume/dynamics envelope
    Expression,         // Expression CC (CC11)
    Modulation,         // Modulation depth (CC1)
    PitchBend,          // Pitch bend envelope
    VibratoDepth,       // Vibrato depth modulation
    FormantShift,       // Formant shifting
    BreathControl,      // Breath controller (CC2)
    Custom              // Custom automation
};

struct AutomationEnvelope {
    AutomationType type = AutomationType::Dynamics;
    std::vector<AutomationPoint> points;  // Time-ordered automation points

    // Interpolation method
    enum class Interpolation {
        Linear,
        Smooth,     // Smooth/curved interpolation
        Step        // Step/held values
    } interpolation = Interpolation::Linear;

    float getValueAt(float time_ms) const;
    void addPoint(float time_ms, float value);
    void normalize(); // Normalize points to [0, 1] or [-1, 1] range
};

// =============================================================================
// Articulation Envelope (per phoneme)
// =============================================================================

struct ArticulationEnvelope {
    PhonemeType phoneme = PhonemeType::UNKNOWN;

    // Attack phase
    float attack_time_ms = 10.0f;    // Time to reach peak
    float attack_curve = 0.5f;       // Attack curve shape [0=linear, 1=exponential]

    // Sustain phase
    float sustain_level = 1.0f;      // Sustain level [0-1]
    float sustain_time_ms = 100.0f;  // Sustain duration

    // Release phase
    float release_time_ms = 50.0f;   // Release time
    float release_curve = 0.5f;      // Release curve shape [0=linear, 1=exponential]

    // Phoneme-specific characteristics
    float consonant_strength = 1.0f; // For consonants: strength of articulation
    float vowel_sustain_mult = 1.0f; // For vowels: sustain multiplier
};

// =============================================================================
// Phoneme Control Data (Main Output Structure)
// =============================================================================

/**
 * PhonemeControlData - Complete control data output
 *
 * Contains all control data needed to drive DAW vocal synthesizers,
 * formant instruments, samplers, and autotune. This is the primary
 * output format of PRROT/PARROT.
 *
 * The structure contains:
 * - Phoneme sequence with timing
 * - Pitch targets and curves
 * - MIDI notes
 * - Automation envelopes
 * - Articulation control
 * - Vibrato and breath markers
 *
 * This data is designed to be consumed by DAW plugins and never
 * contains final rendered audio.
 */
struct PhonemeControlData {
    // Metadata
    std::string output_id;                  // Unique identifier for this output
    std::string voice_profile_id;          // ID of voice profile used
    uint64_t created_timestamp = 0;        // Creation timestamp
    float tempo_bpm = 120.0f;              // Tempo for timing calculations
    float sample_rate_hz = 44100.0f;       // Sample rate (for envelope calculations)

    // Phoneme sequence
    std::vector<PhonemeTiming> phoneme_sequence;

    // Pitch targets (time-ordered)
    std::vector<PitchTarget> pitch_targets;

    // Vibrato parameters (time-varying)
    std::vector<std::pair<float, VibratoParameters>> vibrato_events; // (time_ms, params)

    // Breath markers
    std::vector<BreathMarker> breath_markers;

    // MIDI notes
    std::vector<MidiNote> midi_notes;

    // Automation envelopes
    std::vector<AutomationEnvelope> automation_envelopes;

    // Articulation envelopes (one per phoneme in sequence)
    std::vector<ArticulationEnvelope> articulation_envelopes;

    // Total duration
    float total_duration_ms() const {
        if (phoneme_sequence.empty()) {
            return 0.0f;
        }
        const auto& last = phoneme_sequence.back();
        return last.start_time_ms + last.duration_ms;
    }

    // Get vibrato parameters at a given time
    VibratoParameters getVibratoAt(float time_ms) const {
        VibratoParameters result{};

        // Find the most recent vibrato event before or at this time
        for (const auto& event : vibrato_events) {
            if (event.first <= time_ms) {
                result = event.second;
            } else {
                break;
            }
        }

        return result;
    }

    // Get automation value at a given time
    float getAutomationValue(AutomationType type, float time_ms) const {
        for (const auto& envelope : automation_envelopes) {
            if (envelope.type == type) {
                return envelope.getValueAt(time_ms);
            }
        }
        return 0.0f; // Default value
    }

    // Validation
    bool isValid() const {
        return !output_id.empty() && !phoneme_sequence.empty() && tempo_bpm > 0.0f;
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

// Convert MIDI note + cents to frequency (Hz)
float midiNoteToFrequency(int midi_note, float cents_offset = 0.0f);

// Convert frequency (Hz) to MIDI note + cents
std::pair<int, float> frequencyToMidiNote(float frequency_hz);

// Convert time in milliseconds to samples
uint64_t msToSamples(float time_ms, float sample_rate_hz);

// Convert samples to time in milliseconds
float samplesToMs(uint64_t samples, float sample_rate_hz);

} // namespace prrot
