#include "penta/harmony/VoiceLeading.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace penta::harmony {

VoiceLeading::VoiceLeading(const Config& config)
    : config_(config)
{
}

std::vector<Note> VoiceLeading::findOptimalVoicing(
    const Chord& targetChord,
    const std::vector<Note>& currentVoices,
    uint8_t targetOctave
) const  {
    // If no current voices, generate a default voicing
    if (currentVoices.empty()) {
        std::vector<Note> result;
        
        // Extract chord tones from pitch class set
        for (int i = 0; i < 12; ++i) {
            if (targetChord.pitchClass[i]) {
                Note note;
                note.pitch = targetOctave * 12 + i;
                note.velocity = 80;
                note.timestamp = 0.0;
                result.push_back(note);
            }
        }
        
        return result;
    }
    
    // Generate candidate voicings
    std::vector<VoicingCandidate> candidates;
    generateVoicingCandidates(targetChord, targetOctave, candidates);
    
    if (candidates.empty()) {
        return currentVoices;  // Fallback to current voicing
    }
    
    // Find voicing with minimum cost from current voices
    float minCost = std::numeric_limits<float>::max();
    const VoicingCandidate* bestCandidate = nullptr;
    
    for (const auto& candidate : candidates) {
        float cost = calculateCost(currentVoices, candidate.voices);
        
        if (cost < minCost) {
            minCost = cost;
            bestCandidate = &candidate;
        }
    }
    
    return bestCandidate ? bestCandidate->voices : currentVoices;
}

float VoiceLeading::calculateCost(
    const std::vector<Note>& from,
    const std::vector<Note>& to
) const noexcept {
    if (from.size() != to.size()) {
        return std::numeric_limits<float>::max();
    }
    
    float totalCost = 0.0f;
    
    // Calculate individual voice motion costs
    for (size_t i = 0; i < from.size(); ++i) {
        totalCost += calculateMotionCost(from[i].pitch, to[i].pitch);
    }
    
    // Penalize parallel motion
    for (size_t i = 0; i < from.size(); ++i) {
        for (size_t j = i + 1; j < from.size(); ++j) {
            int interval1 = std::abs(static_cast<int>(from[i].pitch) - static_cast<int>(from[j].pitch));
            int interval2 = std::abs(static_cast<int>(to[i].pitch) - static_cast<int>(to[j].pitch));
            int motion1 = static_cast<int>(to[i].pitch) - static_cast<int>(from[i].pitch);
            int motion2 = static_cast<int>(to[j].pitch) - static_cast<int>(from[j].pitch);
            
            // Parallel fifths and octaves are heavily penalized
            if ((interval1 == 7 || interval1 == 12) && interval1 == interval2 && 
                motion1 == motion2 && motion1 != 0) {
                totalCost += config_.parallelPenalty;
            }
        }
    }
    
    // Reward contrary motion
    for (size_t i = 0; i < from.size(); ++i) {
        for (size_t j = i + 1; j < from.size(); ++j) {
            int motion1 = static_cast<int>(to[i].pitch) - static_cast<int>(from[i].pitch);
            int motion2 = static_cast<int>(to[j].pitch) - static_cast<int>(from[j].pitch);
            
            // Voices moving in opposite directions
            if ((motion1 > 0 && motion2 < 0) || (motion1 < 0 && motion2 > 0)) {
                totalCost -= config_.contraryBonus;
            }
        }
    }
    
    // Penalize voice crossing if not allowed
    if (!config_.allowVoiceCrossing) {
        for (size_t i = 0; i < to.size(); ++i) {
            for (size_t j = i + 1; j < to.size(); ++j) {
                bool crossing = 
                    (from[i].pitch < from[j].pitch && to[i].pitch > to[j].pitch) ||
                    (from[i].pitch > from[j].pitch && to[i].pitch < to[j].pitch);
                
                if (crossing) {
                    totalCost += config_.parallelPenalty * 2.0f;
                }
            }
        }
    }
    
    return totalCost;
}

void VoiceLeading::updateConfig(const Config& config) noexcept {
    config_ = config;
}

void VoiceLeading::generateVoicingCandidates(
    const Chord& chord,
    uint8_t octave,
    std::vector<VoicingCandidate>& candidates
) const  {
    // Extract chord tones
    std::vector<uint8_t> chordTones;
    for (int i = 0; i < 12; ++i) {
        if (chord.pitchClass[i]) {
            chordTones.push_back(i);
        }
    }
    
    if (chordTones.empty()) {
        return;
    }
    
    // Generate voicings in multiple octaves (for flexibility)
    for (int oct = octave - 1; oct <= octave + 1; ++oct) {
        if (oct < 0 || oct > 8) continue;
        
        VoicingCandidate candidate;
        
        for (uint8_t tone : chordTones) {
            Note note;
            note.pitch = oct * 12 + tone;
            note.velocity = 80;
            note.timestamp = 0.0;
            candidate.voices.push_back(note);
        }
        
        // Sort voices by pitch (low to high)
        std::sort(candidate.voices.begin(), candidate.voices.end(),
            [](const Note& a, const Note& b) { return a.pitch < b.pitch; });
        
        candidate.cost = 0.0f;  // Will be calculated later
        candidates.push_back(candidate);
    }
    
    // Generate inversions (different bass notes)
    for (size_t bassIndex = 1; bassIndex < chordTones.size(); ++bassIndex) {
        VoicingCandidate candidate;
        
        // Start with a different chord tone in the bass
        for (size_t i = 0; i < chordTones.size(); ++i) {
            size_t toneIndex = (bassIndex + i) % chordTones.size();
            uint8_t tone = chordTones[toneIndex];
            
            Note note;
            note.pitch = octave * 12 + tone;
            // Adjust octave for lower voices if needed
            if (i > 0 && note.pitch <= candidate.voices.back().pitch) {
                note.pitch += 12;
            }
            note.velocity = 80;
            note.timestamp = 0.0;
            candidate.voices.push_back(note);
        }
        
        candidate.cost = 0.0f;
        candidates.push_back(candidate);
    }
}

float VoiceLeading::calculateMotionCost(
    uint8_t fromPitch,
    uint8_t toPitch
) const noexcept {
    float distance = std::abs(static_cast<int>(toPitch) - static_cast<int>(fromPitch));
    
    // Penalize large leaps
    if (distance > config_.maxVoiceDistance) {
        return distance * 2.0f;  // Heavy penalty for exceeding max distance
    }
    
    // Linear cost for motion (prefer minimal motion)
    return distance;
}

// --- Ported from kelly::VoiceLeadingEngine ---

VoiceLeading::VoiceLeadingResult VoiceLeading::analyze(
    const std::vector<Note>& fromVoicing,
    const std::vector<Note>& toVoicing
) const {
    VoiceLeadingResult result;
    result.fromVoicing = fromVoicing;
    result.toVoicing = toVoicing;
    result.hasParallelFifths = false;
    result.hasParallelOctaves = false;

    size_t minSize = std::min(fromVoicing.size(), toVoicing.size());

    for (size_t i = 0; i < minSize; ++i) {
        VoiceMovement movement;
        movement.fromPitch = fromVoicing[i].pitch;
        movement.toPitch = toVoicing[i].pitch;
        movement.interval = static_cast<int>(toVoicing[i].pitch) -
                            static_cast<int>(fromVoicing[i].pitch);
        result.movements.push_back(movement);
    }

    // Check for parallel fifths and octaves
    for (size_t i = 0; i < minSize; ++i) {
        for (size_t j = i + 1; j < minSize; ++j) {
            int fromInterval = std::abs(static_cast<int>(fromVoicing[i].pitch) -
                                        static_cast<int>(fromVoicing[j].pitch)) % 12;
            int toInterval = std::abs(static_cast<int>(toVoicing[i].pitch) -
                                      static_cast<int>(toVoicing[j].pitch)) % 12;

            int dir_i = result.movements[i].interval;
            int dir_j = result.movements[j].interval;

            if (fromInterval == 7 && toInterval == 7 &&
                ((dir_i > 0 && dir_j > 0) || (dir_i < 0 && dir_j < 0))) {
                result.hasParallelFifths = true;
            }
            if (fromInterval == 0 && toInterval == 0 &&
                ((dir_i > 0 && dir_j > 0) || (dir_i < 0 && dir_j < 0))) {
                result.hasParallelOctaves = true;
            }
        }
    }

    result.smoothnessScore = calculateSmoothness(result.movements);
    return result;
}

std::vector<std::vector<Note>> VoiceLeading::voiceProgression(
    const std::vector<Chord>& chords,
    const std::vector<Note>& startingVoices,
    uint8_t targetOctave
) const  {
    std::vector<std::vector<Note>> result;
    if (chords.empty()) return result;

    std::vector<Note> currentVoices = startingVoices;

    for (const auto& chord : chords) {
        auto voiced = findOptimalVoicing(chord, currentVoices, targetOctave);
        result.push_back(voiced);
        currentVoices = voiced;
    }

    return result;
}

float VoiceLeading::calculateSmoothness(
    const std::vector<VoiceMovement>& movements
) const noexcept {
    if (movements.empty()) return 1.0f;

    float totalMovement = 0;
    int stepwiseCount = 0;

    for (const auto& m : movements) {
        totalMovement += std::abs(m.interval);
        if (std::abs(m.interval) <= 2) stepwiseCount++;
    }

    float avgMovement = totalMovement / static_cast<float>(movements.size());
    float stepwiseRatio = static_cast<float>(stepwiseCount) /
                          static_cast<float>(movements.size());

    // Lower average movement and higher stepwise ratio = smoother
    float smoothness = (1.0f - avgMovement / 12.0f) * 0.5f + stepwiseRatio * 0.5f;
    return std::clamp(smoothness, 0.0f, 1.0f);
}

std::vector<Note> VoiceLeading::invertVoicing(
    const std::vector<Note>& voicing,
    int inversion
) {
    if (voicing.empty()) return voicing;

    std::vector<Note> result = voicing;
    for (int i = 0; i < inversion && !result.empty(); ++i) {
        result[0].pitch += 12;
        std::rotate(result.begin(), result.begin() + 1, result.end());
    }
    return result;
}

} // namespace penta::harmony
