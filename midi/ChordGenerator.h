#pragma once

#include "common/Types.h"
#include <vector>

namespace kelly {

/**
 * Generates chord progressions based on musical intent.
 */
class ChordGenerator {
public:
    ChordGenerator();
    ~ChordGenerator() = default;
    
    /** Generate chord progression from validated intent (compile-time guardrail enforced). */
    std::vector<Chord> generate(const ValidatedIntent& validated);
    /** Backward-compatible wrapper used by legacy runtime tests. */
    std::vector<Chord> generate(const IntentResult& intent);

    /** C2: Return 2–3 voicing/inversion variations. No density increase. Opt-in only (caller enforces). */
    std::vector<std::vector<Chord>> suggestVariations(const std::vector<Chord>& chords);
    
private:
    /** Generate chords for a specific mode. escalationTokenPresent: if false, triads only (I-2). */
    std::vector<Chord> generateForMode(const std::string& mode, int numChords, double beatsPerChord,
                                       bool escalationTokenPresent = false, float complexity = 0.0f);
    
    /** Get scale degrees for a mode */
    std::vector<int> getScaleDegrees(const std::string& mode, int root = 60);
    
    /** Build chord from root scale degree, scale, and chord degree indices */
    std::vector<int> buildChord(int rootScaleDegree, const std::vector<int>& scale, const std::vector<int>& chordDegrees);
};

} // namespace kelly
