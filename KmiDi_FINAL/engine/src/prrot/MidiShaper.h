#pragma once

/**
 * MidiShaper.h - MIDI Probability Shaping
 * =======================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe MIDI probability shaping for generating MIDI notes
 * from phoneme sequences and pitch targets.
 */

#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeControlData.h"
#include <vector>

namespace prrot {

/**
 * MidiShaper - MIDI probability shaping
 */
class MidiShaper {
public:
    MidiShaper();
    ~MidiShaper() = default;

    // Shape MIDI notes from phoneme sequence and pitch targets
    std::vector<MidiNote> shapeMidiNotes(
        const std::vector<PhonemeTiming>& phoneme_sequence,
        const std::vector<PitchTarget>& pitch_targets,
        const VoiceProfile& voice_profile,
        float tempo_bpm
    ) const noexcept;

    // Compute MIDI velocity from phoneme characteristics
    int computeVelocity(
        PhonemeType phoneme,
        float stress_level,
        const VoiceProfile& voice_profile
    ) const noexcept;

    // Compute MIDI note probability distribution
    struct NoteProbability {
        int midi_note;
        float probability;
    };

    std::vector<NoteProbability> computeNoteProbabilities(
        const PitchTarget& target,
        const VoiceProfile& voice_profile
    ) const noexcept;

private:
    // Helper functions
    float getPhonemeDurationMultiplier(PhonemeType phoneme, const VoiceProfile& profile) const noexcept;
};

} // namespace prrot
