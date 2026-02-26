#include "midi/ChordGenerator.h"
#include "common/GuardrailValidator.h"
#include "common/DebugLog.h"
#include <juce_core/juce_core.h>
#include <algorithm>
#include <random>

namespace kelly {

ChordGenerator::ChordGenerator() {
}

std::vector<Chord> ChordGenerator::generate(const ValidatedIntent& validated) {
    const auto& intent = validated.intent();
    if (!canProduceOutput(intent)) {
        return {};
    }
    // #region agent log
    debugLogB("ChordGenerator.cpp:generate", "generate entry", "escalationTokenPresent", intent.escalationTokenPresent, "C");
    // #endregion

    // LISTENING_GUARDRAIL: Minimal harmonic realization (I-2, M-1).
    int numChords = 4;
    double beatsPerChord = 4.0;
    float complexity = 0.0f;
    return generateForMode(intent.mode, numChords, beatsPerChord, intent.escalationTokenPresent, complexity);
}

std::vector<Chord> ChordGenerator::generate(const IntentResult& intent) {
    const bool hasAnyDelta = !(intent.status == IntentResultStatus::ABSTAIN
        && intent.abstainReason == AbstainReason::EMPTY_INPUT);
    const bool hasTextDelta = !(intent.status == IntentResultStatus::ABSTAIN
        && (intent.abstainReason == AbstainReason::EMPTY_INPUT
            || intent.abstainReason == AbstainReason::UNDERSPECIFIED));
    GuardrailInputContext context(
        hasAnyDelta,
        hasTextDelta,
        false,
        false,
        intent.requestedGroupsMask,
        "");
    auto validation = GuardrailValidator::validate(IntentResult(intent), context);
    if (!validation.validated.has_value()) {
        return {};
    }
    return generate(*validation.validated);
}

std::vector<Chord> ChordGenerator::generateForMode(const std::string& mode, int numChords, double beatsPerChord,
                                                   bool escalationTokenPresent, float complexity) {
    std::vector<Chord> chords;
    
    // Get scale for the mode
    std::vector<int> scale = getScaleDegrees(mode, 60); // C as root
    
    // Common progressions for different modes
    std::vector<std::vector<int>> progression;
    
    if (mode == "major") {
        // I-V-vi-IV progression (C-G-Am-F)
        progression = {{0}, {4}, {5}, {3}};
    } else if (mode == "minor") {
        // i-iv-v-i progression (Cm-Fm-G-Cm)
        progression = {{0}, {3}, {4}, {0}};
    } else if (mode == "dorian") {
        // i-IV-i-v progression
        progression = {{0}, {3}, {0}, {4}};
    } else if (mode == "lydian") {
        // I-II-I-V progression
        progression = {{0}, {1}, {0}, {4}};
    } else if (mode == "locrian") {
        // i-bii-i progression (unstable)
        progression = {{0}, {1}, {0}, {6}};
    } else {
        // Default: simple progression
        progression = {{0}, {4}, {0}, {4}};
    }
    
        // Generate chords
    for (int i = 0; i < numChords; ++i) {
        Chord chord;
        int progressionIndex = i % progression.size();
        int rootDegree = progression[progressionIndex][0];
        
        // LISTENING_GUARDRAIL: Minimal harmonic realization (M-1). Triads only unless escalation token.
        std::vector<int> scaleDegrees = {0, 2, 4}; // Root, third, fifth (triad max)
        if (escalationTokenPresent && mode != "locrian") {
            // Use complexity axis to graduate extensions: >0.3 = sevenths, >0.7 = ninths
            if (complexity > 0.3f || i % 2 == 0) {
                scaleDegrees.push_back(6); // Seventh when escalated (E-1)
            }
            if (complexity > 0.7f && i % 2 == 0) {
                scaleDegrees.push_back(8); // Ninth at high complexity
            }
        }
        // #region agent log
        if (i == 0) {
            debugLogB("ChordGenerator.cpp:generateForMode", "chord degrees", "escalationTokenPresent", escalationTokenPresent, "C");
            debugLogI("ChordGenerator.cpp:generateForMode", "chord degrees", "scaleDegrees_size", (int)scaleDegrees.size(), "C");
        }
        // #endregion
        jassert(escalationTokenPresent || scaleDegrees.size() <= 3); // No unrequested extensions
        
        chord.pitches = buildChord(rootDegree, scale, scaleDegrees);
        chord.startBeat = static_cast<double>(i) * beatsPerChord;
        chord.duration = beatsPerChord * 0.95; // Slight gap between chords
        
        chords.push_back(chord);
    }
    
    return chords;
}

std::vector<int> ChordGenerator::getScaleDegrees(const std::string& mode, int root) {
    // Returns MIDI note numbers for a scale starting at root
    // Intervals in semitones
    std::vector<int> intervals;
    
    if (mode == "major") {
        intervals = {0, 2, 4, 5, 7, 9, 11}; // Major scale
    } else if (mode == "minor") {
        intervals = {0, 2, 3, 5, 7, 8, 10}; // Natural minor
    } else if (mode == "dorian") {
        intervals = {0, 2, 3, 5, 7, 9, 10}; // Dorian mode
    } else if (mode == "lydian") {
        intervals = {0, 2, 4, 6, 7, 9, 11}; // Lydian mode
    } else if (mode == "locrian") {
        intervals = {0, 1, 3, 5, 6, 8, 10}; // Locrian mode
    } else {
        // Default to major
        intervals = {0, 2, 4, 5, 7, 9, 11};
    }
    
    std::vector<int> scale;
    for (int interval : intervals) {
        scale.push_back(root + interval);
    }
    
    return scale;
}

std::vector<int> ChordGenerator::buildChord(int rootScaleDegree, const std::vector<int>& scale, const std::vector<int>& chordDegrees) {
    std::vector<int> pitches;
    
    // Build chord from scale degrees relative to root
    // rootScaleDegree is the index in the scale (e.g., 0 = C, 4 = G)
    // chordDegrees are relative to the root (0 = root, 2 = third, 4 = fifth)
    int rootIndex = rootScaleDegree % scale.size();
    int rootOctave = rootScaleDegree / scale.size();
    int rootNote = scale[rootIndex] + (rootOctave * 12);
    
    for (int degree : chordDegrees) {
        // Calculate absolute scale index
        int absoluteIndex = (rootIndex + degree) % scale.size();
        int absoluteOctave = rootOctave + ((rootIndex + degree) / scale.size());
        
        // Get the note from the scale
        int chordNote = scale[absoluteIndex] + (absoluteOctave * 12);
        
        // Clamp to valid MIDI range
        chordNote = std::clamp(chordNote, 0, 127);
        pitches.push_back(chordNote);
    }
    
    // Sort pitches for cleaner voicing
    std::sort(pitches.begin(), pitches.end());
    
    return pitches;
}

static std::vector<int> invertChord(const std::vector<int>& pitches, int inversion) {
    if (pitches.empty() || inversion == 0)
        return pitches;
    const size_t n = pitches.size();
    const int inv = inversion % static_cast<int>(n);
    if (inv == 0)
        return pitches;
    std::vector<int> sorted(pitches);
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> result;
    for (size_t i = 0; i < n; ++i)
        result.push_back((i < static_cast<size_t>(inv)) ? sorted[i] + 12 : sorted[i]);
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::vector<Chord>> ChordGenerator::suggestVariations(const std::vector<Chord>& chords) {
    std::vector<std::vector<Chord>> variations;
    if (chords.empty())
        return variations;
    // Determine max non-identity inversion: for n-note chords, inv % n == 0 is identity
    int minVoices = 127;
    for (const auto& c : chords) {
        if (!c.pitches.empty() && static_cast<int>(c.pitches.size()) < minVoices)
            minVoices = static_cast<int>(c.pitches.size());
    }
    const int maxInv = std::min(3, std::max(1, minVoices - 1));

    // Start at inversion 1 — inversion 0 is identity (same as baseline)
    for (int inv = 1; inv <= maxInv; ++inv) {
        std::vector<Chord> var;
        for (const auto& c : chords) {
            Chord nc;
            nc.pitches = invertChord(c.pitches, inv);
            nc.startBeat = c.startBeat;
            nc.duration = c.duration;
            var.push_back(nc);
        }
        variations.push_back(std::move(var));
    }
    return variations;
}

} // namespace kelly
