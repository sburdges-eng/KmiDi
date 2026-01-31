#include "midi/ChordGenerator.h"
#include <algorithm>

namespace kelly {

ChordGenerator::ChordGenerator() 
    : rng_(std::random_device{}()) 
{
    initializeTemplates();
}

void ChordGenerator::initializeTemplates() {
    // Sad/Melancholic progressions (negative valence, low arousal)
    templates_.push_back({"Sad i-VI-III-VII", {1, 6, 3, 7}, {-1.0f, -0.3f}, {0.0f, 0.4f}});
    templates_.push_back({"Grief i-iv-i-V", {1, 4, 1, 5}, {-1.0f, -0.5f}, {0.0f, 0.3f}});
    templates_.push_back({"Melancholy i-VII-VI-VII", {1, 7, 6, 7}, {-0.7f, -0.2f}, {0.1f, 0.5f}});
    
    // Angry/Intense progressions (negative valence, high arousal)
    templates_.push_back({"Rage i-bII-i-V", {1, 2, 1, 5}, {-1.0f, -0.5f}, {0.7f, 1.0f}});
    templates_.push_back({"Tension i-iv-bVI-V", {1, 4, 6, 5}, {-0.8f, -0.3f}, {0.5f, 0.9f}});
    
    // Anxious progressions (negative valence, mid-high arousal)
    templates_.push_back({"Anxiety ii-V-i-vi", {2, 5, 1, 6}, {-0.6f, -0.2f}, {0.5f, 0.8f}});
    templates_.push_back({"Dread i-bVI-bVII-i", {1, 6, 7, 1}, {-0.8f, -0.4f}, {0.4f, 0.7f}});
    
    // Hopeful progressions (positive valence, mid arousal)
    templates_.push_back({"Hope I-V-vi-IV", {1, 5, 6, 4}, {0.2f, 0.7f}, {0.3f, 0.7f}});
    templates_.push_back({"Rising I-IV-vi-V", {1, 4, 6, 5}, {0.3f, 0.8f}, {0.4f, 0.8f}});
    
    // Joyful progressions (positive valence, high arousal)
    templates_.push_back({"Joy I-IV-V-I", {1, 4, 5, 1}, {0.5f, 1.0f}, {0.6f, 1.0f}});
    templates_.push_back({"Triumph I-V-vi-iii-IV", {1, 5, 6, 3, 4}, {0.6f, 1.0f}, {0.7f, 1.0f}});
    
    // Peaceful progressions (positive valence, low arousal)
    templates_.push_back({"Peace I-vi-IV-V", {1, 6, 4, 5}, {0.3f, 0.7f}, {0.0f, 0.3f}});
    templates_.push_back({"Serenity I-iii-vi-IV", {1, 3, 6, 4}, {0.4f, 0.8f}, {0.0f, 0.4f}});
    
    // Bittersweet/Complex progressions (near-zero valence)
    templates_.push_back({"Bittersweet i-III-VII-IV", {1, 3, 7, 4}, {-0.3f, 0.3f}, {0.3f, 0.6f}});
    templates_.push_back({"Nostalgia I-vi-ii-V", {1, 6, 2, 5}, {-0.2f, 0.4f}, {0.2f, 0.5f}});
    templates_.push_back({"Ambivalent i-IV-i-V", {1, 4, 1, 5}, {-0.4f, 0.2f}, {0.3f, 0.6f}});
}

std::vector<Chord> ChordGenerator::generate(const IntentResult& intent, int bars) {
    // Find best matching template
    const ProgressionTemplate* best = nullptr;
    float bestScore = -999.0f;
    
    float valence = intent.emotion.valence;
    float arousal = intent.emotion.arousal;
    
    for (const auto& tmpl : templates_) {
        // Check if emotion fits this template's range
        if (valence >= tmpl.valenceRange[0] && valence <= tmpl.valenceRange[1] &&
            arousal >= tmpl.energyRange[0] && arousal <= tmpl.energyRange[1]) {
            
            // Score by how centered we are in the range
            float vCenter = (tmpl.valenceRange[0] + tmpl.valenceRange[1]) / 2.0f;
            float aCenter = (tmpl.energyRange[0] + tmpl.energyRange[1]) / 2.0f;
            float score = 1.0f - (std::abs(valence - vCenter) + std::abs(arousal - aCenter));
            
            // Add some randomness
            std::uniform_real_distribution<float> noise(-0.2f, 0.2f);
            score += noise(rng_);
            
            if (score > bestScore) {
                bestScore = score;
                best = &tmpl;
            }
        }
    }
    
    // Default to melancholy if nothing matches
    if (!best) {
        best = &templates_[1];  // Grief template
    }
    
    // Root note: C3 = 48, but adjust based on valence
    int rootNote = 48;  // C3
    if (valence < -0.5f) rootNote = 45;  // A2 - darker
    else if (valence > 0.5f) rootNote = 52;  // E3 - brighter
    
    // Generate chords from template
    std::vector<Chord> chords;
    double beatsPerChord = (bars * 4.0) / best->degrees.size();
    double currentBeat = 0.0;
    
    bool addExtensions = intent.emotion.intensity > 0.6f;
    
    for (int degree : best->degrees) {
        Chord chord = buildChord(degree, intent.mode, rootNote, 
                                  currentBeat, beatsPerChord, addExtensions);
        chords.push_back(chord);
        currentBeat += beatsPerChord;
    }
    
    // Apply rule-break modifications
    for (const auto& rb : intent.ruleBreaks) {
        if (rb.type == RuleBreakType::Harmony) {
            if (rb.severity > 0.5f) {
                applyDissonance(chords, rb.severity);
            }
            if (rb.severity > 0.7f) {
                addChromaticism(chords, rb.severity);
            }
        }
    }
    
    return chords;
}

std::vector<int> ChordGenerator::getScaleIntervals(const std::string& mode) const {
    // Intervals from root (in semitones)
    if (mode == "major" || mode == "ionian") return {0, 2, 4, 5, 7, 9, 11};
    if (mode == "minor" || mode == "aeolian") return {0, 2, 3, 5, 7, 8, 10};
    if (mode == "dorian") return {0, 2, 3, 5, 7, 9, 10};
    if (mode == "phrygian") return {0, 1, 3, 5, 7, 8, 10};
    if (mode == "lydian") return {0, 2, 4, 6, 7, 9, 11};
    if (mode == "mixolydian") return {0, 2, 4, 5, 7, 9, 10};
    if (mode == "locrian") return {0, 1, 3, 5, 6, 8, 10};
    
    return {0, 2, 3, 5, 7, 8, 10};  // Default to natural minor
}

Chord ChordGenerator::buildChord(int degree, const std::string& mode, int rootNote,
                                  double startBeat, double duration, bool addExtension) {
    auto scale = getScaleIntervals(mode);
    
    // Degree is 1-indexed, scale is 0-indexed
    int degreeIndex = (degree - 1) % 7;
    int chordRoot = rootNote + scale[degreeIndex];
    
    // Build triad: root, third, fifth
    int third = chordRoot + scale[(degreeIndex + 2) % 7] - scale[degreeIndex];
    int fifth = chordRoot + scale[(degreeIndex + 4) % 7] - scale[degreeIndex];
    
    // Handle octave wrapping
    if (third < chordRoot) third += 12;
    if (fifth < third) fifth += 12;
    
    Chord chord;
    chord.pitches = {chordRoot, third, fifth};
    chord.startBeat = startBeat;
    chord.duration = duration;
    
    // Add 7th for extensions
    if (addExtension) {
        int seventh = chordRoot + scale[(degreeIndex + 6) % 7] - scale[degreeIndex];
        if (seventh < fifth) seventh += 12;
        chord.pitches.push_back(seventh);
    }
    
    // Generate chord name (simplified)
    static const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    chord.name = noteNames[chordRoot % 12];
    
    int thirdInterval = third - chordRoot;
    if (thirdInterval == 3) chord.name += "m";  // Minor third
    if (addExtension) chord.name += "7";
    
    return chord;
}

void ChordGenerator::applyDissonance(std::vector<Chord>& chords, float severity) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    
    for (auto& chord : chords) {
        if (chance(rng_) < severity * 0.5f) {
            // Add a dissonant note (minor 2nd or tritone)
            int root = chord.pitches[0];
            
            if (chance(rng_) < 0.5f) {
                // Minor 2nd above root
                chord.pitches.push_back(root + 1);
            } else {
                // Tritone
                chord.pitches.push_back(root + 6);
            }
            
            chord.name += "(add b9)";
        }
    }
}

void ChordGenerator::addChromaticism(std::vector<Chord>& chords, float severity) {
    if (chords.size() < 2) return;
    
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::vector<Chord> newChords;
    
    for (size_t i = 0; i < chords.size(); ++i) {
        newChords.push_back(chords[i]);
        
        // Maybe insert a chromatic passing chord
        if (i < chords.size() - 1 && chance(rng_) < severity * 0.3f) {
            Chord passing = chords[i];
            
            // Shift all notes up or down by half step toward next chord
            int nextRoot = chords[i + 1].pitches[0];
            int currentRoot = chords[i].pitches[0];
            int direction = (nextRoot > currentRoot) ? 1 : -1;
            
            for (auto& pitch : passing.pitches) {
                pitch += direction;
            }
            
            // Shorten the original and add passing chord
            newChords.back().duration *= 0.75;
            passing.startBeat = newChords.back().startBeat + newChords.back().duration;
            passing.duration = chords[i].duration * 0.25;
            passing.name = "pass";
            
            newChords.push_back(passing);
        }
    }
    
    chords = newChords;
}

std::vector<Chord> ChordGenerator::generateProgression(
    const std::string& mode,
    int rootNote,
    int bars,
    bool allowDissonance,
    float intensity
) {
    // Create a synthetic intent for direct generation
    IntentResult intent;
    intent.mode = mode;
    intent.allowDissonance = allowDissonance;
    intent.emotion.intensity = intensity;
    intent.emotion.valence = (mode == "major" || mode == "lydian") ? 0.5f : -0.3f;
    intent.emotion.arousal = intensity;
    
    if (allowDissonance) {
        intent.ruleBreaks.push_back({
            RuleBreakType::Harmony,
            intensity,
            "User requested dissonance",
            "Direct parameter"
        });
    }
    
    return generate(intent, bars);
}

} // namespace kelly
