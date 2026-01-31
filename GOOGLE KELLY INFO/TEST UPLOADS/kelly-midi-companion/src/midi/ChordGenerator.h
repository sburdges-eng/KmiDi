#pragma once

#include "common/Types.h"
#include <vector>
#include <random>

namespace kelly {

/**
 * Generates chord progressions based on emotional intent.
 * 
 * Uses the rule-break system to intentionally violate 
 * conventional harmony when emotions demand it.
 */
class ChordGenerator {
public:
    ChordGenerator();
    
    /** Generate a chord progression from intent */
    std::vector<Chord> generate(const IntentResult& intent, int bars = 4);
    
    /** Get a specific progression pattern */
    std::vector<Chord> generateProgression(
        const std::string& mode,
        int rootNote,
        int bars,
        bool allowDissonance,
        float intensity
    );
    
private:
    std::mt19937 rng_;
    
    // Progression templates by emotional character
    struct ProgressionTemplate {
        std::string name;
        std::vector<int> degrees;  // Scale degrees (1-7)
        float valenceRange[2];     // Min/max valence this works for
        float energyRange[2];      // Min/max arousal this works for
    };
    
    std::vector<ProgressionTemplate> templates_;
    
    void initializeTemplates();
    
    // Build a chord from scale degree
    Chord buildChord(int degree, const std::string& mode, int rootNote, 
                     double startBeat, double duration, bool addExtension);
    
    // Get scale intervals for a mode
    std::vector<int> getScaleIntervals(const std::string& mode) const;
    
    // Apply dissonance based on rule-breaks
    void applyDissonance(std::vector<Chord>& chords, float severity);
    
    // Add chromatic passing chords
    void addChromaticism(std::vector<Chord>& chords, float severity);
};

} // namespace kelly
