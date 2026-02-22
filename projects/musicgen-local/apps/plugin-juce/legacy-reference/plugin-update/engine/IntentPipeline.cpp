#include "engine/IntentPipeline.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace kelly {

IntentPipeline::IntentPipeline() {
}

IntentResult IntentPipeline::process(const Wound& wound) {
    Emotion emotion = analyzeEmotion(wound.description, wound.intensity);
    std::string mode = determineMode(emotion);
    float tempo = calculateTempo(emotion);
    std::vector<RuleBreak> ruleBreaks = checkRuleBreaks(emotion, mode);
    
    return IntentResult{emotion, mode, tempo, ruleBreaks};
}

IntentResult IntentPipeline::processJourney(const SideA& current, const SideB& desired) {
    // Analyze both states
    Emotion currentEmotion = analyzeEmotion(current.description, current.intensity);
    Emotion desiredEmotion = analyzeEmotion(desired.description, desired.intensity);
    
    // Create a journey emotion (interpolation with direction toward desired)
    Emotion journeyEmotion;
    journeyEmotion.name = "journey";
    journeyEmotion.valence = (currentEmotion.valence * 0.3f + desiredEmotion.valence * 0.7f);
    journeyEmotion.arousal = (currentEmotion.arousal * 0.3f + desiredEmotion.arousal * 0.7f);
    
    std::string mode = determineMode(journeyEmotion);
    float tempo = calculateTempo(journeyEmotion);
    std::vector<RuleBreak> ruleBreaks = checkRuleBreaks(journeyEmotion, mode);
    
    return IntentResult{journeyEmotion, mode, tempo, ruleBreaks};
}

Emotion IntentPipeline::analyzeEmotion(const std::string& text, float intensity) {
    Emotion emotion;
    
    // Simple keyword-based analysis
    emotion.valence = detectEmotionalValence(text);
    emotion.arousal = detectEmotionalArousal(text);
    
    // Scale by intensity
    emotion.valence *= intensity;
    emotion.arousal *= intensity;
    
    // Clamp values
    emotion.valence = std::clamp(emotion.valence, -1.0f, 1.0f);
    emotion.arousal = std::clamp(emotion.arousal, 0.0f, 1.0f);
    
    // Generate name based on quadrant
    if (emotion.valence > 0.3f && emotion.arousal > 0.5f) {
        emotion.name = "joyful";
    } else if (emotion.valence > 0.3f && emotion.arousal <= 0.5f) {
        emotion.name = "content";
    } else if (emotion.valence < -0.3f && emotion.arousal > 0.5f) {
        emotion.name = "anxious";
    } else if (emotion.valence < -0.3f && emotion.arousal <= 0.5f) {
        emotion.name = "sad";
    } else {
        emotion.name = "neutral";
    }
    
    return emotion;
}

std::string IntentPipeline::determineMode(const Emotion& emotion) {
    // Map emotion to musical mode
    if (emotion.valence > 0.2f) {
        if (emotion.arousal > 0.6f) {
            return "lydian";  // Bright, uplifting
        } else {
            return "major";   // Happy, stable
        }
    } else if (emotion.valence < -0.2f) {
        if (emotion.arousal > 0.6f) {
            return "locrian"; // Tense, unstable
        } else {
            return "minor";   // Sad, melancholic
        }
    } else {
        return "dorian";      // Neutral, balanced
    }
}

float IntentPipeline::calculateTempo(const Emotion& emotion) {
    // Higher arousal = faster tempo
    // Base tempo multiplier: 0.8 to 1.4
    return 0.8f + (emotion.arousal * 0.6f);
}

std::vector<RuleBreak> IntentPipeline::checkRuleBreaks(const Emotion& emotion, const std::string& mode) {
    std::vector<RuleBreak> breaks;
    
    // Example rule: very negative emotions shouldn't use major mode
    if (emotion.valence < -0.5f && mode == "major") {
        breaks.push_back({"mode_emotion_mismatch", 
                         "Negative emotion with major mode - using modal mixture"});
    }
    
    // Example rule: extreme arousal might need special handling
    if (emotion.arousal > 0.9f) {
        breaks.push_back({"high_arousal", 
                         "Very high arousal detected - may need dynamic processing"});
    }
    
    return breaks;
}

float IntentPipeline::detectEmotionalValence(const std::string& text) {
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    
    // Positive keywords
    int positive = 0;
    std::vector<std::string> positiveWords = {
        "happy", "joy", "good", "great", "wonderful", "love", "peaceful",
        "calm", "content", "hopeful", "excited", "grateful", "blessed"
    };
    for (const auto& word : positiveWords) {
        if (lowerText.find(word) != std::string::npos) positive++;
    }
    
    // Negative keywords
    int negative = 0;
    std::vector<std::string> negativeWords = {
        "sad", "angry", "fear", "anxious", "worried", "hurt", "pain",
        "lonely", "depressed", "frustrated", "overwhelmed", "scared", "terrified"
    };
    for (const auto& word : negativeWords) {
        if (lowerText.find(word) != std::string::npos) negative++;
    }
    
    // Calculate valence: -1.0 (very negative) to 1.0 (very positive)
    int total = positive + negative;
    if (total == 0) return 0.0f;
    
    return static_cast<float>(positive - negative) / static_cast<float>(total);
}

float IntentPipeline::detectEmotionalArousal(const std::string& text) {
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    
    // High arousal keywords
    int highArousal = 0;
    std::vector<std::string> highArousalWords = {
        "excited", "anxious", "fear", "terrified", "panicked", "ecstatic",
        "energetic", "overwhelmed", "intense", "frantic", "urgent"
    };
    for (const auto& word : highArousalWords) {
        if (lowerText.find(word) != std::string::npos) highArousal++;
    }
    
    // Low arousal keywords
    int lowArousal = 0;
    std::vector<std::string> lowArousalWords = {
        "calm", "peaceful", "relaxed", "sleepy", "tired", "exhausted",
        "serene", "tranquil", "lethargic", "drowsy"
    };
    for (const auto& word : lowArousalWords) {
        if (lowerText.find(word) != std::string::npos) lowArousal++;
    }
    
    // Calculate arousal: 0.0 (low) to 1.0 (high)
    if (highArousal == 0 && lowArousal == 0) return 0.5f; // Neutral
    
    int total = highArousal + lowArousal;
    return static_cast<float>(highArousal) / static_cast<float>(total);
}

} // namespace kelly

