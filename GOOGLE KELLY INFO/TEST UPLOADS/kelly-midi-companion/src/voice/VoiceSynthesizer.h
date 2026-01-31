#pragma once

#include "common/Types.h"
#include <vector>
#include <string>
#include <optional>

namespace kelly {

// Forward declarations
struct GeneratedMidi;
struct EmotionNode;
struct Wound;

/**
 * Voice Synthesizer - Generates vocal melodies and lyrics based on emotion.
 * 
 * v2.0 feature: Synthesizes voice parts to accompany generated MIDI.
 * This is a framework for future voice synthesis integration.
 */
class VoiceSynthesizer {
public:
    VoiceSynthesizer();
    ~VoiceSynthesizer() = default;
    
    /**
     * Generate vocal melody based on emotion and MIDI context.
     * @param emotion The emotion to express
     * @param midiContext The generated MIDI to accompany
     * @return Vector of vocal notes (pitch, timing, lyrics)
     */
    struct VocalNote {
        int pitch;          // MIDI pitch
        double startBeat;   // Start position in beats
        double duration;    // Duration in beats
        std::string lyric;  // Optional lyric syllable
        float vibrato;      // Vibrato amount (0.0 to 1.0)
    };
    
    std::vector<VocalNote> generateVocalMelody(
        const EmotionNode& emotion,
        const GeneratedMidi& midiContext
    );
    
    /**
     * Generate lyrics based on emotion and wound description.
     * @param emotion The emotion to express
     * @param wound The original wound description
     * @return Vector of lyric lines
     */
    std::vector<std::string> generateLyrics(
        const EmotionNode& emotion,
        const Wound& wound
    );
    
    /**
     * Synthesize audio from vocal notes.
     * @param notes The vocal notes to synthesize
     * @param sampleRate Target sample rate
     * @return Audio buffer with synthesized voice
     */
    std::vector<float> synthesizeAudio(
        const std::vector<VocalNote>& notes,
        double sampleRate = 44100.0
    );
    
    /** Enable/disable voice synthesis */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
private:
    bool enabled_ = false;
    
    /** Generate melody contour based on emotion */
    std::vector<int> generateMelodyContour(const EmotionNode& emotion, int numNotes);
    
    /** Map emotion to vocal characteristics */
    struct VocalCharacteristics {
        float brightness;    // 0.0 (dark) to 1.0 (bright)
        float breathiness;   // 0.0 (clear) to 1.0 (breathy)
        float vibratoRate;   // Vibrato speed in Hz
        float vibratoDepth;  // Vibrato depth (0.0 to 1.0)
    };
    
    VocalCharacteristics getVocalCharacteristics(const EmotionNode& emotion) const;
};

} // namespace kelly
