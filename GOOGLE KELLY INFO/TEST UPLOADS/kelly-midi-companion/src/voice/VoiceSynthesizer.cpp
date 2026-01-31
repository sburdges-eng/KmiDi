#include "voice/VoiceSynthesizer.h"
#include "common/Types.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {

VoiceSynthesizer::VoiceSynthesizer() : enabled_(false) {
}

std::vector<VoiceSynthesizer::VocalNote> VoiceSynthesizer::generateVocalMelody(
    const EmotionNode& emotion,
    const GeneratedMidi& midiContext)
{
    if (!enabled_) return {};
    
    std::vector<VocalNote> vocalNotes;
    
    // Generate melody contour based on emotion
    int numNotes = static_cast<int>(midiContext.lengthInBeats / 0.5);  // ~2 notes per beat
    auto contour = generateMelodyContour(emotion, numNotes);
    
    // Map contour to actual pitches in a singable range (C4 to C6)
    int basePitch = 60;  // C4
    if (emotion.valence < -0.5f) basePitch = 57;  // Lower for negative emotions
    else if (emotion.valence > 0.5f) basePitch = 64;  // Higher for positive
    
    double beatPosition = 0.0;
    double noteDuration = 0.5;  // Half-beat notes
    
    for (size_t i = 0; i < contour.size() && beatPosition < midiContext.lengthInBeats; ++i) {
        VocalNote note;
        note.pitch = basePitch + contour[i];
        note.startBeat = beatPosition;
        note.duration = noteDuration;
        note.vibrato = emotion.intensity * 0.3f;  // More vibrato for intense emotions
        
        // Clamp to singable range
        note.pitch = std::clamp(note.pitch, 48, 84);  // C3 to C6
        
        vocalNotes.push_back(note);
        beatPosition += noteDuration;
    }
    
    return vocalNotes;
}

std::vector<int> VoiceSynthesizer::generateMelodyContour(const EmotionNode& emotion, int numNotes) {
    std::vector<int> contour;
    contour.reserve(numNotes);
    
    // Simple contour generation based on emotion
    // Negative emotions: descending patterns
    // Positive emotions: ascending patterns
    // High arousal: more variation
    
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> intervalDist(-3, 3);
    
    int currentOffset = 0;
    for (int i = 0; i < numNotes; ++i) {
        // Bias based on valence
        int bias = emotion.valence > 0 ? 1 : -1;
        int interval = intervalDist(rng) + bias;
        
        // High arousal = more variation
        if (emotion.arousal > 0.7f) {
            interval += (intervalDist(rng) % 2);
        }
        
        currentOffset += interval;
        currentOffset = std::clamp(currentOffset, -12, 12);  // Keep within octave
        contour.push_back(currentOffset);
    }
    
    return contour;
}

std::vector<std::string> VoiceSynthesizer::generateLyrics(
    const EmotionNode& emotion,
    const Wound& wound)
{
    if (!enabled_) return {};
    
    std::vector<std::string> lyrics;
    
    // Simple lyric generation based on emotion keywords
    // In a full implementation, this would use NLP or template-based generation
    
    if (emotion.name.find("sad") != std::string::npos ||
        emotion.name.find("lonely") != std::string::npos) {
        lyrics.push_back("Alone in the silence");
        lyrics.push_back("Echoes of what was");
        lyrics.push_back("Searching for meaning");
    } else if (emotion.name.find("joy") != std::string::npos ||
               emotion.name.find("happy") != std::string::npos) {
        lyrics.push_back("Light fills the spaces");
        lyrics.push_back("Moments of wonder");
        lyrics.push_back("Dancing in freedom");
    } else if (emotion.name.find("anger") != std::string::npos ||
               emotion.name.find("rage") != std::string::npos) {
        lyrics.push_back("Fire in the darkness");
        lyrics.push_back("Breaking the silence");
        lyrics.push_back("Voice of the wounded");
    } else {
        // Generic emotional expression
        lyrics.push_back("Feel what I feel");
        lyrics.push_back("Words cannot capture");
        lyrics.push_back("Music tells the truth");
    }
    
    return lyrics;
}

std::vector<float> VoiceSynthesizer::synthesizeAudio(
    const std::vector<VocalNote>& notes,
    double sampleRate)
{
    if (!enabled_ || notes.empty()) return {};
    
    // Placeholder: In a full implementation, this would use a vocoder or formant synthesizer
    // For now, return empty buffer (framework for future implementation)
    
    // Calculate total duration
    double totalDuration = 0.0;
    for (const auto& note : notes) {
        totalDuration = std::max(totalDuration, note.startBeat + note.duration);
    }
    
    // Convert beats to samples (assuming 120 BPM)
    int totalSamples = static_cast<int>(totalDuration * (sampleRate * 60.0 / 120.0));
    return std::vector<float>(totalSamples, 0.0f);
}

VoiceSynthesizer::VocalCharacteristics VoiceSynthesizer::getVocalCharacteristics(const EmotionNode& emotion) const {
    VocalCharacteristics chars;
    
    // Brightness: positive emotions = brighter
    chars.brightness = (emotion.valence + 1.0f) * 0.5f;
    
    // Breathiness: high intensity = more breathy
    chars.breathiness = emotion.intensity * 0.6f;
    
    // Vibrato: more for intense emotions
    chars.vibratoRate = 4.0f + (emotion.arousal * 2.0f);  // 4-6 Hz
    chars.vibratoDepth = emotion.intensity * 0.4f;
    
    return chars;
}

} // namespace kelly
