#pragma once

#include "penta/common/Platform.h"

#include "penta/common/RTTypes.h"
#include <vector>

namespace penta::harmony {

/**
 * Voice leading optimizer using minimal motion principles
 * Implements smooth voice transitions between chords
 */
class VoiceLeading {
public:
    struct Config {
        float maxVoiceDistance;  // Max semitones to move
        float parallelPenalty;    // Penalty for parallel motion
        float contraryBonus;      // Bonus for contrary motion
        bool allowVoiceCrossing;
        
        Config()
            : maxVoiceDistance(12.0f)
            , parallelPenalty(5.0f)
            , contraryBonus(2.0f)
            , allowVoiceCrossing(false)
        {}
    };
    
    explicit VoiceLeading(const Config& config = Config{});
    ~VoiceLeading() = default;
    
    // RT-safe: Find optimal voice leading from current to target chord
    std::vector<Note> findOptimalVoicing(
        const Chord& targetChord,
        const std::vector<Note>& currentVoices,
        uint8_t targetOctave = 4
    ) const noexcept;
    
    // RT-safe: Calculate voice leading cost
    float calculateCost(
        const std::vector<Note>& from,
        const std::vector<Note>& to
    ) const noexcept;
    
    // Configuration
    void updateConfig(const Config& config) noexcept;

    // --- Ported from kelly::VoiceLeadingEngine ---

    struct VoiceMovement {
        uint8_t fromPitch;
        uint8_t toPitch;
        int interval;  // toPitch - fromPitch (positive=up, negative=down, 0=static)
    };

    struct VoiceLeadingResult {
        std::vector<Note> fromVoicing;
        std::vector<Note> toVoicing;
        std::vector<VoiceMovement> movements;
        float smoothnessScore;
        bool hasParallelFifths;
        bool hasParallelOctaves;
    };

    // Analyze movement between two voicings
    VoiceLeadingResult analyze(
        const std::vector<Note>& fromVoicing,
        const std::vector<Note>& toVoicing
    ) const noexcept;

    // Voice an entire chord progression
    std::vector<std::vector<Note>> voiceProgression(
        const std::vector<Chord>& chords,
        const std::vector<Note>& startingVoices,
        uint8_t targetOctave = 4
    ) const noexcept;

    // Smoothness metric (0.0 = poor, 1.0 = ideal)
    float calculateSmoothness(
        const std::vector<VoiceMovement>& movements
    ) const noexcept;

    // Chord inversion (rotate voices by N positions, octave-shift the rotated note)
    static std::vector<Note> invertVoicing(
        const std::vector<Note>& voicing,
        int inversion
    ) noexcept;

private:
    struct VoicingCandidate {
        std::vector<Note> voices;
        float cost;
        
        bool operator<(const VoicingCandidate& other) const {
            return cost < other.cost;
        }
    };
    
    void generateVoicingCandidates(
        const Chord& chord,
        uint8_t octave,
        std::vector<VoicingCandidate>& candidates
    ) const noexcept;
    
    float calculateMotionCost(
        uint8_t fromPitch,
        uint8_t toPitch
    ) const noexcept;
    
    Config config_;
};

} // namespace penta::harmony
