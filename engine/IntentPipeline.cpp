#include "engine/IntentPipeline.h"
#include "engine/CoreBridge.h"
#include "engine/AdapterRegistry.h"
#include "common/DebugLog.h"
#include <juce_core/juce_core.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace kelly {

IntentPipeline::IntentPipeline() {
}

IntentResult IntentPipeline::process(const Wound& wound) {
    // #region agent log
    debugLog("IntentPipeline.cpp:process", "process entry", "description", wound.description, "E");
    // #endregion
    // LISTENING_GUARDRAIL: Empty input must ABSTAIN (I-3, R-1)
    std::string trimmed = wound.description;
    const size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        // #region agent log
        debugLog("IntentPipeline.cpp:process", "ABSTAIN", "reason", "find_first_not_of_npos", "E");
        // #endregion
        return IntentResult::abstain();
    }
    const size_t end = trimmed.find_last_not_of(" \t\n\r");
    trimmed = trimmed.substr(start, end - start + 1);
    if (trimmed.empty()) {
        // #region agent log
        debugLog("IntentPipeline.cpp:process", "ABSTAIN", "reason", "trimmed_empty", "E");
        // #endregion
        return IntentResult::abstain();
    }

    // LISTENING_GUARDRAIL: I-3 — intent never inferred when input empty
    jassert(!trimmed.empty());

    // LISTENING_GUARDRAIL: R-2, I-3 — Underspecified intent must ABSTAIN (original contract)
    // Only infer when text matches a known intent type (LISTENING_SPEC M-1)
    {
        std::string lower = trimmed;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        bool hasKnownIntent = (lower.find("chord change") != std::string::npos ||
                              lower.find("darker") != std::string::npos ||
                              lower.find("more energy") != std::string::npos ||
                              lower.find("smoother") != std::string::npos);
        if (!hasKnownIntent) {
            // #region agent log
            debugLog("IntentPipeline.cpp:process", "ABSTAIN", "reason", "no_known_intent", "E");
            // #endregion
            return IntentResult::abstain();
        }
    }

    IntentResult result;
    result.status = IntentResultStatus::VALID;
    result.emotion = analyzeEmotion(wound.description, wound.intensity);

    UserInput adapterInput;
    adapterInput.text = trimmed;
    AdapterRegistry::instance().applyIntentEnrichers(adapterInput, result);

    std::string mode = determineMode(result.emotion);
    float tempo = calculateTempo(result.emotion);
    std::vector<RuleBreak> ruleBreaks = checkRuleBreaks(result.emotion, mode);

    // LISTENING_GUARDRAIL: Constraint violations are fatal (I-4)
    if (!ruleBreaks.empty()) {
        return IntentResult::invalid();
    }

    result.mode = mode;
    result.tempo = tempo;
    result.ruleBreaks = std::move(ruleBreaks);
    result.escalationTokenPresent = detectEscalationToken(wound.description);
    return CoreBridge::validateViaCore(result);
}

IntentResult IntentPipeline::processJourney(const SideA& current, const SideB& desired) {
    // #region agent log
    debugLog("IntentPipeline.cpp:processJourney", "processJourney entry", "current", current.description, "D");
    debugLog("IntentPipeline.cpp:processJourney", "processJourney entry", "desired", desired.description, "D");
    // #endregion
    // LISTENING_GUARDRAIL: Absence is signal (I-3, R-1). Empty Side A or Side B must ABSTAIN.
    auto trim = [](std::string s) {
        s.erase(0, s.find_first_not_of(" \t\n\r"));
        if (s.empty()) return s;
        s.erase(s.find_last_not_of(" \t\n\r") + 1);
        return s;
    };
    std::string currT = trim(current.description);
    std::string desT = trim(desired.description);
    if (currT.empty() || desT.empty()) {
        // #region agent log
        debugLog("IntentPipeline.cpp:processJourney", "ABSTAIN", "reason", currT.empty() ? "sideA_empty" : "sideB_empty", "D");
        // #endregion
        return IntentResult::abstain();
    }

    // LISTENING_GUARDRAIL: I-3 — intent never inferred when input empty
    jassert(!currT.empty() && !desT.empty());

    // LISTENING_GUARDRAIL: R-2, I-3 — Underspecified intent must ABSTAIN (original contract)
    auto hasKnownIntent = [](const std::string& s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower.find("chord change") != std::string::npos ||
               lower.find("darker") != std::string::npos ||
               lower.find("more energy") != std::string::npos ||
               lower.find("smoother") != std::string::npos;
    };
    bool curHas = hasKnownIntent(current.description);
    bool desHas = hasKnownIntent(desired.description);
    if (!curHas || !desHas) {
        // #region agent log
        debugLogB("IntentPipeline.cpp:processJourney", "ABSTAIN", "curHasKnownIntent", curHas, "D");
        debugLogB("IntentPipeline.cpp:processJourney", "ABSTAIN", "desHasKnownIntent", desHas, "D");
        // #endregion
        return IntentResult::abstain();
    }

    Emotion currentEmotion = analyzeEmotion(current.description, current.intensity);
    Emotion desiredEmotion = analyzeEmotion(desired.description, desired.intensity);

    Emotion journeyEmotion;
    journeyEmotion.name = "journey";
    journeyEmotion.valence = (currentEmotion.valence * 0.3f + desiredEmotion.valence * 0.7f);
    journeyEmotion.arousal = (currentEmotion.arousal * 0.3f + desiredEmotion.arousal * 0.7f);

    IntentResult result;
    result.status = IntentResultStatus::VALID;
    result.emotion = journeyEmotion;

    UserInput adapterInput;
    adapterInput.text = currT + " " + desT;
    AdapterRegistry::instance().applyIntentEnrichers(adapterInput, result);

    std::string mode = determineMode(result.emotion);
    float tempo = calculateTempo(result.emotion);
    std::vector<RuleBreak> ruleBreaks = checkRuleBreaks(result.emotion, mode);

    // LISTENING_GUARDRAIL: Constraint violations are fatal (I-4)
    if (!ruleBreaks.empty()) {
        return IntentResult::invalid();
    }

    result.mode = mode;
    result.tempo = tempo;
    result.ruleBreaks = std::move(ruleBreaks);
    result.escalationTokenPresent = detectEscalationToken(current.description) || detectEscalationToken(desired.description);
    return CoreBridge::validateViaCore(result);
}

Emotion IntentPipeline::analyzeEmotion(const std::string& text, float intensity) {
    Emotion emotion;
    
    // [LISTENING] Treats under-specification as permission: empty text → neutral, arousal 0.5.
    // No "I don't know" / abstain path. See LISTENING_CONTRACT_AUDIT.md
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
    // [LISTENING] Always produces a mode. No "unspecified—don't assume" path.
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
    
    // [LISTENING] RuleBreaks are logged but not honored as vetoes. System proceeds anyway.
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

bool IntentPipeline::detectEscalationToken(const std::string& text) {
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    static const std::vector<std::string> tokens = {
        "more", "richer", "push it", "give me options", "go further"
    };
    for (const auto& token : tokens) {
        if (lowerText.find(token) != std::string::npos)
            return true;
    }
    return false;
}

} // namespace kelly
