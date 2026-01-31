#include "midi/MidiGenerator.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kelly {

MidiGenerator::MidiGenerator() : rng_(std::random_device{}()) {
}

GeneratedMidi MidiGenerator::generate(
    const IntentResult& intent,
    int bars,
    float complexity,
    float humanize,
    float feel,
    float dynamics
) {
    GeneratedMidi result;
    result.bpm = 120.0f;
    result.lengthInBeats = bars * 4.0;
    
    // Generate chord progression (longer and more varied)
    result.chords = chordGen_.generate(intent, bars);
    
    // Generate melody
    result.melody = generateMelody(result.chords, intent.emotion, complexity, dynamics);
    
    // Generate bass
    result.bass = generateBass(result.chords, intent.emotion, complexity, dynamics);
    
    // Apply groove and humanization
    applyGrooveAndHumanize(result, humanize, intent.emotion, feel);
    
    return result;
}

std::vector<MidiNote> MidiGenerator::generateMelody(
    const std::vector<Chord>& chords,
    const EmotionNode& emotion,
    float complexity,
    float dynamics
) {
    std::vector<MidiNote> melody;
    
    // Scale intervals for different modes
    std::vector<int> scaleIntervals;
    if (emotion.valence > 0.3f) {
        // Major modes
        scaleIntervals = {0, 2, 4, 5, 7, 9, 11};  // Major
    } else if (emotion.valence < -0.3f) {
        // Minor modes
        scaleIntervals = {0, 2, 3, 5, 7, 8, 10};  // Natural minor
    } else {
        // Dorian (bittersweet)
        scaleIntervals = {0, 2, 3, 5, 7, 9, 10};
    }
    
    int rootNote = 60; // C4
    if (emotion.valence < -0.5f) rootNote = 57; // A3 - darker
    else if (emotion.valence > 0.5f) rootNote = 64; // E4 - brighter
    
    double currentBeat = 0.0;
    int lastPitch = rootNote + 7; // Start on 5th
    
    for (const auto& chord : chords) {
        // Safety check
        if (chord.pitches.empty()) {
            currentBeat += chord.duration;
            continue;
        }
        
        // Number of melody notes per chord depends on complexity and arousal
        int notesPerChord = 1;
        if (complexity > 0.3f) notesPerChord = 2;
        if (complexity > 0.6f && emotion.arousal > 0.5f) notesPerChord = 4;
        if (complexity > 0.8f) notesPerChord = 6;
        
        double noteDuration = notesPerChord > 0 ? chord.duration / notesPerChord : chord.duration;
        
        for (int i = 0; i < notesPerChord; ++i) {
            double noteStart = currentBeat + (i * noteDuration);
            
            // Choose note from chord tones or scale
            int pitch;
            if (complexity < 0.5f || i == 0) {
                // Use chord tones (simpler)
                int chordTone = i % static_cast<int>(chord.pitches.size());
                pitch = chord.pitches[chordTone];
                // Move to melody range
                while (pitch < 60) pitch += 12;
                while (pitch > 84) pitch -= 12;
            } else {
                // Use scale tones (more complex)
                int scaleDegree = (i + lastPitch - rootNote) % scaleIntervals.size();
                pitch = rootNote + scaleIntervals[scaleDegree];
                
                // Add some chromaticism for high complexity
                if (complexity > 0.7f) {
                    std::uniform_int_distribution<int> chrom(-1, 1);
                    pitch += chrom(rng_);
                }
            }
            
            // Smooth transitions
            if (i > 0) {
                int diff = pitch - lastPitch;
                if (std::abs(diff) > 7) {
                    pitch = lastPitch + (diff > 0 ? 7 : -7);
                }
            }
            
            // Velocity based on emotion and dynamics parameter
            int baseVelocity = 80;
            if (emotion.arousal > 0.7f) baseVelocity = 100;
            else if (emotion.arousal < 0.3f) baseVelocity = 60;
            
            // Apply dynamics scaling (0.0 = quiet, 1.0 = full range)
            int velocityRange = static_cast<int>((127 - 40) * dynamics);
            int velocityCenter = 40 + (velocityRange / 2);
            int velocity = velocityCenter + static_cast<int>((baseVelocity - 80) * (dynamics * 0.5f));
            
            // Add velocity variation
            std::uniform_int_distribution<int> velVar(-10, 10);
            velocity = std::clamp(velocity + velVar(rng_), 40, 127);
            
            melody.push_back({pitch, velocity, noteStart, noteDuration * 0.9});
            lastPitch = pitch;
        }
        
        currentBeat += chord.duration;
    }
    
    return melody;
}

std::vector<MidiNote> MidiGenerator::generateBass(
    const std::vector<Chord>& chords,
    const EmotionNode& emotion,
    float complexity,
    float dynamics
) {
    std::vector<MidiNote> bass;
    
    double currentBeat = 0.0;
    
    for (const auto& chord : chords) {
        // Safety check
        if (chord.pitches.empty()) continue;
        
        // Bass plays root note
        int rootPitch = chord.pitches[0];
        // Move to bass range (C2 = 36)
        while (rootPitch > 48) rootPitch -= 12;
        while (rootPitch < 24) rootPitch += 12;
        
        // Rhythm depends on arousal and complexity
        int notesPerChord = 1;
        if (emotion.arousal > 0.5f) notesPerChord = 2;
        if (complexity > 0.5f && emotion.arousal > 0.6f) notesPerChord = 4;
        
        double noteDuration = notesPerChord > 0 ? chord.duration / notesPerChord : chord.duration;
        
        for (int i = 0; i < notesPerChord; ++i) {
            double noteStart = currentBeat + (i * noteDuration);
            
            int pitch = rootPitch;
            
            // Add octave jumps for complexity
            if (complexity > 0.6f && i > 0 && (i % 2 == 0)) {
                pitch += 12; // Octave up
            }
            
            // Add passing tones for high complexity (lead to next chord)
            if (complexity > 0.7f && i == notesPerChord - 1) {
                // Find next chord in sequence
                auto nextChordIt = std::find_if(chords.begin(), chords.end(),
                    [&chord](const Chord& c) { return c.startBeat > chord.startBeat; });
                if (nextChordIt != chords.end() && !nextChordIt->pitches.empty()) {
                    int nextRoot = nextChordIt->pitches[0];
                    // Move to bass range
                    while (nextRoot > 48) nextRoot -= 12;
                    while (nextRoot < 24) nextRoot += 12;
                    // Create passing tone between current and next root
                    pitch = (rootPitch + nextRoot) / 2;
                }
            }
            
            // Bass velocity based on emotion and dynamics
            int baseVelocity = 90;
            if (emotion.arousal > 0.7f) baseVelocity = 110;
            else if (emotion.arousal < 0.3f) baseVelocity = 70;
            
            // Apply dynamics scaling
            int velocityRange = static_cast<int>((127 - 60) * dynamics);
            int velocityCenter = 60 + (velocityRange / 2);
            int velocity = velocityCenter + static_cast<int>((baseVelocity - 90) * (dynamics * 0.5f));
            velocity = std::clamp(velocity, 60, 127);
            
            bass.push_back({pitch, velocity, noteStart, noteDuration * 0.95});
        }
        
        currentBeat += chord.duration;
    }
    
    return bass;
}

void MidiGenerator::applyGrooveAndHumanize(
    GeneratedMidi& midi,
    float humanize,
    const EmotionNode& emotion,
    float feel
) {
    if (humanize < 0.01f && std::abs(feel) < 0.01f) return; // Skip if no humanization or feel
    
    // Apply emotion-based timing
    midi.melody = grooveEngine_.applyEmotionTiming(midi.melody, emotion);
    midi.bass = grooveEngine_.applyEmotionTiming(midi.bass, emotion);
    
    // Apply groove based on arousal
    GrooveType grooveType = GrooveType::Straight;
    if (emotion.arousal > 0.7f) grooveType = GrooveType::Syncopated;
    else if (emotion.arousal < 0.3f) grooveType = GrooveType::Shuffle;
    else if (emotion.valence > 0.5f) grooveType = GrooveType::Swing;
    
    midi.melody = grooveEngine_.applyGroove(midi.melody, grooveType, humanize);
    midi.bass = grooveEngine_.applyGroove(midi.bass, grooveType, humanize * 0.7f); // Less on bass
    
    // Apply feel (pull/push) - adjust timing slightly
    if (std::abs(feel) > 0.01f) {
        float feelOffset = feel * 0.05f; // Small timing adjustment (5% max)
        for (auto& note : midi.melody) {
            note.startBeat += feelOffset;
        }
        for (auto& note : midi.bass) {
            note.startBeat += feelOffset * 0.7f; // Less on bass
        }
    }
}


} // namespace kelly
