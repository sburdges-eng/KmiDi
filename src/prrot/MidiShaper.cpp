#include "prrot/MidiShaper.h"
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>
#include <climits>

namespace prrot {

MidiShaper::MidiShaper() {
}

std::vector<MidiNote> MidiShaper::shapeMidiNotes(
    const std::vector<PhonemeTiming>& phoneme_sequence,
    const std::vector<PitchTarget>& pitch_targets,
    const VoiceProfile& voice_profile,
    float tempo_bpm
) const  {
    std::vector<MidiNote> midi_notes;

    // Validate inputs
    if (phoneme_sequence.empty()) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "MidiShaper::shapeMidiNotes: Empty phoneme sequence");
        return midi_notes;
    }

    if (pitch_targets.empty()) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "MidiShaper::shapeMidiNotes: Empty pitch targets");
        return midi_notes;
    }

    if (tempo_bpm <= 0.0f || tempo_bpm > 300.0f) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            ("MidiShaper::shapeMidiNotes: Invalid tempo: " +
             std::to_string(tempo_bpm) + ", clamping").c_str());
        tempo_bpm = std::clamp(tempo_bpm, 1.0f, 300.0f);
    }

    if (!voice_profile.isValid()) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "MidiShaper::shapeMidiNotes: Invalid voice profile");
        return midi_notes;
    }

    // Map pitch targets to phonemes based on timing
    size_t target_idx = 0;

    for (size_t i = 0; i < phoneme_sequence.size(); ++i) {
        const auto& phoneme_timing = phoneme_sequence[i];

        // Find pitch target(s) that overlap with this phoneme
        std::vector<const PitchTarget*> overlapping_targets;
        for (const auto& target : pitch_targets) {
            if (target.time_ms >= phoneme_timing.start_time_ms &&
                target.time_ms < phoneme_timing.start_time_ms + phoneme_timing.duration_ms) {
                overlapping_targets.push_back(&target);
            }
        }

        // Create MIDI note(s) for this phoneme
        if (!overlapping_targets.empty()) {
            // Use first overlapping target (or could average multiple)
            const PitchTarget& target = *overlapping_targets[0];

            MidiNote note;
            // Validate and clamp MIDI note
            note.note_number = std::clamp(target.midi_note, 0, 127);
            if (target.midi_note != note.note_number) {
                penta::getLogger().logRT(penta::LogLevel::Warning,
                    ("MidiShaper::shapeMidiNotes: MIDI note out of range: " +
                     std::to_string(target.midi_note) + ", clamping").c_str());
            }

            note.velocity = computeVelocity(phoneme_timing.phoneme, phoneme_timing.stress_level, voice_profile);

            // Validate timing
            if (phoneme_timing.start_time_ms < 0.0f) {
                penta::getLogger().logRT(penta::LogLevel::Warning,
                    ("MidiShaper::shapeMidiNotes: Negative start time: " +
                     std::to_string(phoneme_timing.start_time_ms)).c_str());
            }
            note.start_time_ms = std::max(phoneme_timing.start_time_ms, 0.0f);

            if (phoneme_timing.duration_ms <= 0.0f) {
                penta::getLogger().logRT(penta::LogLevel::Warning,
                    ("MidiShaper::shapeMidiNotes: Invalid duration: " +
                     std::to_string(phoneme_timing.duration_ms)).c_str());
                note.duration_ms = 100.0f; // Default duration
            } else {
                note.duration_ms = phoneme_timing.duration_ms;
            }

            // Adjust duration based on phoneme characteristics
            float duration_mult = getPhonemeDurationMultiplier(phoneme_timing.phoneme, voice_profile);
            if (duration_mult <= 0.0f || duration_mult > 10.0f) {
                penta::getLogger().logRT(penta::LogLevel::Warning,
                    ("MidiShaper::shapeMidiNotes: Invalid duration multiplier: " +
                     std::to_string(duration_mult) + ", using 1.0").c_str());
                duration_mult = 1.0f;
            }
            note.duration_ms *= duration_mult;

            // Validate final duration
            if (note.duration_ms <= 0.0f || note.duration_ms > 60000.0f) {
                penta::getLogger().logRT(penta::LogLevel::Warning,
                    ("MidiShaper::shapeMidiNotes: Duration out of range: " +
                     std::to_string(note.duration_ms) + ", clamping").c_str());
                note.duration_ms = std::clamp(note.duration_ms, 10.0f, 60000.0f);
            }

            // Set articulation parameters based on phoneme type
            if (isConsonant(phoneme_timing.phoneme)) {
                auto consonant_profile = voice_profile.getConsonantProfile(phoneme_timing.phoneme);
                note.attack_time_ms = consonant_profile.attack_time_ms;
                note.release_time_ms = consonant_profile.release_time_ms;
            } else {
                // Vowels: smoother attack/release
                note.attack_time_ms = 20.0f;
                note.release_time_ms = 50.0f;
            }

            note.channel = 0; // Default channel

            midi_notes.push_back(note);
        }
    }

    return midi_notes;
}

int MidiShaper::computeVelocity(
    PhonemeType phoneme,
    float stress_level,
    const VoiceProfile& voice_profile
) const noexcept {
    // Base velocity: 80 (moderate)
    float base_velocity = 80.0f;

    // Adjust for stress level
    base_velocity += stress_level * 30.0f; // Up to +30 for stressed

    // Adjust for phoneme type (vowels typically louder)
    if (isVowel(phoneme)) {
        base_velocity += 10.0f;
    } else if (isConsonant(phoneme)) {
        auto consonant_profile = voice_profile.getConsonantProfile(phoneme);
        base_velocity *= consonant_profile.strength_scalar;
    }

    // Clamp to valid MIDI range
    return std::clamp(static_cast<int>(base_velocity), 1, 127);
}

std::vector<MidiShaper::NoteProbability> MidiShaper::computeNoteProbabilities(
    const PitchTarget& target,
    const VoiceProfile& voice_profile
) const  {
    std::vector<NoteProbability> probabilities;

    // Create probability distribution centered on target note
    // Probability decreases for notes further from target
    float cents_uncertainty = 50.0f; // ±50 cents uncertainty

    for (int note_offset = -2; note_offset <= 2; ++note_offset) {
        int midi_note = target.midi_note + note_offset;
        if (midi_note >= 0 && midi_note <= 127) {
            float cents_diff = static_cast<float>(note_offset) * 100.0f + target.cents_offset;
            float probability = std::exp(-(cents_diff * cents_diff) / (2.0f * cents_uncertainty * cents_uncertainty));

            NoteProbability prob;
            prob.midi_note = midi_note;
            prob.probability = probability;
            probabilities.push_back(prob);
        }
    }

    // Normalize probabilities
    float total_prob = 0.0f;
    for (auto& prob : probabilities) {
        total_prob += prob.probability;
    }
    if (total_prob > 1e-6f) {
        for (auto& prob : probabilities) {
            prob.probability /= total_prob;
        }
    }

    return probabilities;
}

float MidiShaper::getPhonemeDurationMultiplier(PhonemeType phoneme, const VoiceProfile& profile) const noexcept {
    auto duration_stats = profile.getPhonemeDuration(phoneme);

    if (duration_stats.sample_count > 0) {
        // Use mean duration from profile, normalized to a baseline
        float baseline_duration = 100.0f; // 100ms baseline
        return duration_stats.mean_ms / baseline_duration;
    }

    // Default multipliers
    if (isVowel(phoneme)) {
        return 1.2f; // Vowels longer
    } else if (isConsonant(phoneme)) {
        return 0.8f; // Consonants shorter
    }

    return 1.0f;
}

} // namespace prrot
