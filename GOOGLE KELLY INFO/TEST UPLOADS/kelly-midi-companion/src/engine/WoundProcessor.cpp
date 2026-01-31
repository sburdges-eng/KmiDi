#include "engine/WoundProcessor.h"

namespace kelly {

std::vector<WoundProcessor::EmotionClue> WoundProcessor::buildEmotionClues() {
    return {
        // Grief/Loss cluster
        {{"loss", "lost", "death", "died", "gone", "missing", "grief", "mourn"}, 1, 0.9f},
        {{"miss", "absence", "empty", "void", "hollow"}, 8, 0.7f},
        
        // Anger cluster  
        {{"angry", "anger", "rage", "furious", "mad", "pissed"}, 20, 0.9f},
        {{"frustrated", "frustration", "annoyed", "irritated"}, 22, 0.8f},
        {{"resentment", "bitter", "betrayed"}, 25, 0.85f},
        
        // Fear/Anxiety cluster
        {{"afraid", "fear", "scared", "terrified", "terror"}, 40, 0.9f},
        {{"anxious", "anxiety", "worried", "nervous", "panic"}, 42, 0.85f},
        {{"dread", "dreading"}, 45, 0.8f},
        
        // Sadness cluster
        {{"sad", "sadness", "unhappy", "depressed", "down"}, 4, 0.8f},
        {{"melancholy", "melancholic", "blue", "gloomy"}, 2, 0.85f},
        {{"lonely", "loneliness", "alone", "isolated"}, 8, 0.8f},
        {{"longing", "yearn", "yearning", "want", "wish"}, 6, 0.75f},
        
        // Joy cluster
        {{"happy", "happiness", "joy", "joyful", "glad", "pleased"}, 10, 0.9f},
        {{"excited", "excitement", "thrilled", "elated"}, 12, 0.85f},
        {{"grateful", "gratitude", "thankful", "appreciative"}, 14, 0.8f},
        {{"love", "loving", "adore", "cherish"}, 16, 0.9f},
        
        // Surprise cluster
        {{"surprised", "surprise", "shocked", "astonished", "amazed"}, 50, 0.85f},
        {{"wonder", "wondering", "awe", "awe-struck"}, 52, 0.8f},
        
        // Complex emotions
        {{"bittersweet", "nostalgic", "nostalgia", "wistful"}, 34, 0.8f},
        {{"confused", "confusion", "uncertain", "unsure"}, 0, 0.7f},
        {{"peaceful", "peace", "calm", "serene", "tranquil"}, 31, 0.85f},
        {{"hopeful", "hope", "optimistic", "positive"}, 36, 0.8f}
    };
}

EmotionNode WoundProcessor::processWound(const Wound& wound) {
    return findEmotionByKeywords(wound.description, wound.intensity);
}

EmotionNode WoundProcessor::findEmotionByKeywords(const std::string& text, float intensity) {
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    
    auto clues = buildEmotionClues();
    float bestConfidence = 0.0f;
    int bestEmotionId = 0;
    
    // Find the best matching clue
    for (const auto& clue : clues) {
        for (const auto& keyword : clue.keywords) {
            if (lowerText.find(keyword) != std::string::npos) {
                if (clue.confidence > bestConfidence) {
                    bestConfidence = clue.confidence;
                    bestEmotionId = clue.primaryEmotionId;
                }
                break;
            }
        }
    }
    
    // If we found a match, use it
    if (bestEmotionId > 0) {
        auto emotion = thesaurus_.findById(bestEmotionId);
        if (emotion) {
            auto result = *emotion;
            result.intensity = intensity;  // Override with wound intensity
            return result;
        }
    }
    
    // Fallback: find nearest emotion by analyzing valence/arousal from keywords
    float estimatedValence = 0.0f;
    float estimatedArousal = 0.5f;
    
    // Simple heuristic: negative words -> negative valence, intensity words -> high arousal
    if (lowerText.find("sad") != std::string::npos || 
        lowerText.find("angry") != std::string::npos ||
        lowerText.find("fear") != std::string::npos) {
        estimatedValence = -0.5f;
    }
    if (lowerText.find("happy") != std::string::npos ||
        lowerText.find("joy") != std::string::npos ||
        lowerText.find("love") != std::string::npos) {
        estimatedValence = 0.5f;
    }
    if (lowerText.find("intense") != std::string::npos ||
        lowerText.find("overwhelming") != std::string::npos ||
        lowerText.find("extreme") != std::string::npos) {
        estimatedArousal = 0.8f;
    }
    
    return thesaurus_.findNearest(estimatedValence, estimatedArousal, intensity);
}

} // namespace kelly
