#pragma once

/**
 * VoiceLeadingAdapter.h
 *
 * Preserves kelly's int-based VoiceLeading API for ChordGenerator compatibility.
 * Delegates to penta::harmony::VoiceLeading internally.
 */

#include "penta/harmony/VoiceLeading.h"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace kelly {

enum class VoiceLeadingStyle {
    Smooth,
    Common,
    Contrary,
    Parallel,
    Oblique,
    Free
};

struct VoiceLeadingConfig {
    VoiceLeadingStyle style = VoiceLeadingStyle::Smooth;
    bool avoidParallelFifths = true;
    bool avoidParallelOctaves = true;
    int maxVoiceMovement = 4;  // semitones
    bool preferStepwiseMotion = true;
};

class VoiceLeadingEngine {
public:
    VoiceLeadingEngine() : penta_() {}

    void setConfig(const VoiceLeadingConfig& config) {
        config_ = config;

        penta::harmony::VoiceLeading::Config pc;
        pc.maxVoiceDistance = static_cast<float>(config.maxVoiceMovement);
        pc.parallelPenalty = (config.avoidParallelFifths || config.avoidParallelOctaves) ? 5.0f : 0.0f;
        pc.contraryBonus = (config.style == VoiceLeadingStyle::Contrary) ? 4.0f : 2.0f;
        pc.allowVoiceCrossing = (config.style == VoiceLeadingStyle::Free);

        penta_.updateConfig(pc);
    }

    std::vector<std::vector<int>> voiceProgression(
        const std::vector<std::vector<int>>& chordTones,
        int startingBass = 36
    ) {
        if (chordTones.empty()) return {};

        // Build penta Chords from int pitch vectors
        std::vector<penta::Chord> chords;
        chords.reserve(chordTones.size());
        for (const auto& tones : chordTones) {
            penta::Chord c;
            c.pitchClass.fill(false);
            for (int pitch : tones) {
                int pc = pitch % 12;
                if (pc < 0) pc += 12;
                c.pitchClass[pc] = true;
            }
            if (!tones.empty()) {
                c.root = static_cast<uint8_t>(tones[0] % 12);
            }
            chords.push_back(c);
        }

        // Build starting voices from the first chord's int pitches
        std::vector<penta::Note> startingVoices;
        if (!chordTones[0].empty()) {
            for (int pitch : chordTones[0]) {
                startingVoices.emplace_back(
                    static_cast<uint8_t>(std::clamp(pitch, 0, 127)),
                    uint8_t(80), uint8_t(0), uint64_t(0));
            }
        } else {
            startingVoices.emplace_back(
                static_cast<uint8_t>(std::clamp(startingBass, 0, 127)),
                uint8_t(80), uint8_t(0), uint64_t(0));
        }

        uint8_t targetOctave = static_cast<uint8_t>(
            std::clamp(startingBass / 12, 0, 9));

        // Delegate to penta
        auto voiced = penta_.voiceProgression(chords, startingVoices, targetOctave);

        // Convert back to vector<vector<int>>
        std::vector<std::vector<int>> result;
        result.reserve(voiced.size());
        for (const auto& noteVec : voiced) {
            std::vector<int> pitches;
            pitches.reserve(noteVec.size());
            for (const auto& note : noteVec) {
                pitches.push_back(static_cast<int>(note.pitch));
            }
            result.push_back(std::move(pitches));
        }

        return result;
    }

private:
    VoiceLeadingConfig config_;
    penta::harmony::VoiceLeading penta_;
};

} // namespace kelly
