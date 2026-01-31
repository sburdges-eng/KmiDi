#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <cstdint>

namespace kelly {

//==============================================================================
// Emotion Types
//==============================================================================

enum class EmotionCategory {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation
};

struct EmotionNode {
    int id;
    std::string name;
    EmotionCategory category;
    float intensity;    // 0.0 to 1.0
    float valence;      // -1.0 (negative) to 1.0 (positive)
    float arousal;      // 0.0 (calm) to 1.0 (excited)
    std::vector<int> relatedEmotions;
};

//==============================================================================
// Intent Types
//==============================================================================

struct Wound {
    std::string description;
    float intensity;
    std::string source;
};

enum class RuleBreakType {
    Harmony,
    Rhythm,
    Dynamics,
    Melody,
    Form
};

struct RuleBreak {
    RuleBreakType type;
    float severity;         // 0.0 to 1.0
    std::string description;
    std::string reason;     // Why this rule is being broken (emotional justification)
};

struct IntentResult {
    Wound wound;
    EmotionNode emotion;
    std::vector<RuleBreak> ruleBreaks;
    
    // Musical parameters derived from emotion + rule breaks
    std::string mode;           // "major", "minor", "dorian", etc.
    float tempo;                // BPM modifier (0.5 to 2.0)
    float dynamicRange;         // 0.0 to 1.0
    bool allowDissonance;
    float syncopationLevel;     // 0.0 to 1.0
    float humanization;         // 0.0 to 1.0 (timing/velocity variance)
};

//==============================================================================
// MIDI Types
//==============================================================================

struct MidiNote {
    int pitch;          // 0-127
    int velocity;       // 0-127
    double startBeat;   // Position in beats
    double duration;    // Length in beats
};

struct Chord {
    std::vector<int> pitches;
    std::string name;           // e.g., "Am7", "Cmaj9"
    double startBeat;
    double duration;
};

enum class GrooveType {
    Straight,
    Swing,
    Syncopated,
    Halftime,
    Shuffle
};

struct GrooveTemplate {
    GrooveType type;
    std::string name;
    int numerator;
    int denominator;
    float swingAmount;          // 0.0 to 1.0
    std::vector<std::pair<float, int>> pattern;  // (beat position, velocity)
};

struct GeneratedMidi {
    std::vector<Chord> chords;
    std::vector<MidiNote> melody;
    std::vector<MidiNote> bass;
    double lengthInBeats;
    float bpm;
};

//==============================================================================
// Side A / Side B (Cassette Metaphor)
//==============================================================================

struct SideA {
    std::string description;    // "Where you are" - current emotional state
    float intensity;
    std::optional<int> emotionId;  // If user selected from wheel
};

struct SideB {
    std::string description;    // "Where you want to go" - desired state
    float intensity;
    std::optional<int> emotionId;
};

struct CassetteState {
    SideA sideA;
    SideB sideB;
    bool isFlipped;             // Which side is currently "active"
};

} // namespace kelly
