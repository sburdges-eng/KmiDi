#include "midi/GrooveEngine.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace kelly {

// Constants from Python groove engine
constexpr float MAX_BEAT_DRIFT = 0.036f;  // ~35 ticks at 960 PPQ = ~0.036 beats
constexpr float HUMAN_LATENCY_BIAS = 0.005f;  // Slight behind-the-beat bias
constexpr int VELOCITY_MIN = 20;
constexpr int VELOCITY_MAX = 120;
constexpr float MAX_DROPOUT_PROB = 0.2f;  // 20% max dropout at complexity=1.0
constexpr float GHOST_NOTE_PROBABILITY = 0.15f;
constexpr float GHOST_NOTE_VELOCITY_MULT = 0.4f;

GrooveEngine::GrooveEngine() : rng_(std::random_device{}()) {
}

std::vector<MidiNote> GrooveEngine::applyGroove(
    const std::vector<MidiNote>& notes,
    GrooveType grooveType,
    float humanizationLevel)
{
    if (humanizationLevel < 0.01f) return notes;
    
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    std::uniform_real_distribution<float> timingDist(-MAX_BEAT_DRIFT, MAX_BEAT_DRIFT);
    std::uniform_real_distribution<float> velocityDist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dropoutDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> ghostDist(0.0f, 1.0f);
    
    for (const auto& note : notes) {
        // Dropout probability based on humanization level
        float dropoutProb = humanizationLevel * MAX_DROPOUT_PROB;
        if (dropoutDist(rng_) < dropoutProb) {
            continue;  // Drop this note
        }
        
        MidiNote processed = note;
        
        // Apply timing drift (with slight behind-the-beat bias)
        float timingDrift = timingDist(rng_) * humanizationLevel;
        timingDrift += HUMAN_LATENCY_BIAS * humanizationLevel;  // Slight lag
        processed.startBeat += timingDrift;
        
        // Apply groove type adjustments
        switch (grooveType) {
            case GrooveType::Swing:
                // Swing: delay off-beats by ~33%
                if (std::fmod(processed.startBeat * 2.0, 1.0) > 0.5) {
                    processed.startBeat += 0.033f * humanizationLevel;
                }
                break;
            case GrooveType::Syncopated:
                // Syncopation: push some notes ahead
                if (dropoutDist(rng_) < 0.3f) {
                    processed.startBeat -= 0.02f * humanizationLevel;
                }
                break;
            case GrooveType::Shuffle:
                // Shuffle: delay every other 8th note
                if (std::fmod(processed.startBeat * 2.0, 1.0) > 0.5) {
                    processed.startBeat += 0.05f * humanizationLevel;
                }
                break;
            case GrooveType::Halftime:
                // Halftime: stretch timing
                processed.startBeat *= 1.0f + (0.1f * humanizationLevel);
                break;
            default:  // Straight
                break;
        }
        
        // Apply velocity humanization
        int velocityChange = static_cast<int>(velocityDist(rng_) * humanizationLevel);
        processed.velocity = std::clamp(note.velocity + velocityChange, VELOCITY_MIN, VELOCITY_MAX);
        
        result.push_back(processed);
        
        // Add ghost notes occasionally
        if (ghostDist(rng_) < GHOST_NOTE_PROBABILITY * humanizationLevel) {
            MidiNote ghost = processed;
            ghost.startBeat += timingDist(rng_) * 0.5f;  // Near the original
            ghost.velocity = std::clamp(
                static_cast<int>(processed.velocity * GHOST_NOTE_VELOCITY_MULT),
                VELOCITY_MIN / 2,
                VELOCITY_MAX
            );
            ghost.duration *= 0.5f;  // Shorter ghost notes
            result.push_back(ghost);
        }
    }
    
    // Sort by startBeat to maintain order
    std::sort(result.begin(), result.end(),
        [](const MidiNote& a, const MidiNote& b) {
            return a.startBeat < b.startBeat;
        });
    
    return result;
}

std::vector<MidiNote> GrooveEngine::applyEmotionTiming(
    const std::vector<MidiNote>& notes,
    const EmotionNode& emotion)
{
    std::vector<MidiNote> result;
    result.reserve(notes.size());
    
    // Emotion-based timing adjustments
    // Sad emotions drag behind, angry emotions rush ahead
    float timingBias = 0.0f;
    
    if (emotion.valence < -0.5f) {
        // Negative emotions (sad, angry) - timing adjustments
        if (emotion.name.find("sad") != std::string::npos ||
            emotion.name.find("grief") != std::string::npos ||
            emotion.name.find("lonely") != std::string::npos) {
            timingBias = -0.01f * emotion.intensity;  // Drag behind
        } else if (emotion.name.find("angry") != std::string::npos ||
                   emotion.name.find("rage") != std::string::npos ||
                   emotion.name.find("fury") != std::string::npos) {
            timingBias = 0.01f * emotion.intensity;  // Rush ahead
        }
    }
    
    // Arousal affects timing tightness
    float timingVariance = (1.0f - emotion.arousal) * 0.02f;  // Lower arousal = more variance
    std::uniform_real_distribution<float> timingDist(-timingVariance, timingVariance);
    
    for (const auto& note : notes) {
        MidiNote processed = note;
        processed.startBeat += timingBias + timingDist(rng_);
        result.push_back(processed);
    }
    
    return result;
}

} // namespace kelly
