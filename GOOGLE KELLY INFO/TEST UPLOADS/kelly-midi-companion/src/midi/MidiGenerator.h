#pragma once

#include "common/Types.h"
#include "engine/IntentPipeline.h"
#include "midi/ChordGenerator.h"
#include "midi/GrooveEngine.h"
#include <vector>

namespace kelly {

/**
 * Complete MIDI Generator - Uses all modules to generate full arrangements.
 * 
 * Generates:
 * - Chord progressions (using ChordGenerator)
 * - Melody lines (emotion-driven)
 * - Bass lines (root-based with variations)
 * - Applies groove and humanization (using GrooveEngine)
 * - Applies rule-breaks for emotional authenticity
 */
class MidiGenerator {
public:
    MidiGenerator();
    
    /**
     * Generate complete MIDI arrangement from emotional intent.
     * @param intent The processed emotional intent
     * @param bars Number of bars to generate (default 8, can be 4-32)
     * @param complexity 0.0 (simple) to 1.0 (complex) - affects chord extensions, melody density
     * @param humanize 0.0 (quantized) to 1.0 (loose) - affects timing and velocity
     * @param feel -1.0 (pull) to 1.0 (push) - affects timing feel
     * @param dynamics 0.0 to 1.0 - affects velocity range
     * @return Complete MIDI with chords, melody, and bass
     */
    GeneratedMidi generate(
        const IntentResult& intent,
        int bars = 8,
        float complexity = 0.5f,
        float humanize = 0.4f,
        float feel = 0.0f,
        float dynamics = 0.75f
    );
    
    /**
     * Generate melody line based on emotion and chord progression.
     */
    std::vector<MidiNote> generateMelody(
        const std::vector<Chord>& chords,
        const EmotionNode& emotion,
        float complexity,
        float dynamics
    );
    
    /**
     * Generate bass line following chord roots with rhythmic variations.
     */
    std::vector<MidiNote> generateBass(
        const std::vector<Chord>& chords,
        const EmotionNode& emotion,
        float complexity,
        float dynamics
    );
    
private:
    ChordGenerator chordGen_;
    GrooveEngine grooveEngine_;
    std::mt19937 rng_;
    
    
    // Apply groove and humanization
    void applyGrooveAndHumanize(GeneratedMidi& midi, float humanize, const EmotionNode& emotion, float feel);
};

} // namespace kelly
