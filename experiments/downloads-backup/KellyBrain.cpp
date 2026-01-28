#include "KellyBrain.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {

// =============================================================================
// EmotionThesaurus Implementation
// =============================================================================

EmotionThesaurus::EmotionThesaurus() {
    initializeDefault();
}

void EmotionThesaurus::initializeDefault() {
    // Initialize 6 base emotions with sub-emotions
    // Full 216-node system: 6 base × 6 sub × 6 sub-sub
    
    struct BaseEmotion {
        EmotionCategory category;
        std::string name;
        float valence;
        float arousal;
        std::vector<std::string> subEmotions;
    };
    
    std::vector<BaseEmotion> bases = {
        {EmotionCategory::Joy, "joy", 0.8f, 0.6f, 
         {"happiness", "contentment", "elation", "pride", "optimism", "relief"}},
        {EmotionCategory::Sadness, "sadness", -0.7f, 0.3f,
         {"grief", "melancholy", "despair", "loneliness", "disappointment", "regret"}},
        {EmotionCategory::Anger, "anger", -0.5f, 0.8f,
         {"rage", "frustration", "irritation", "resentment", "defiance", "bitterness"}},
        {EmotionCategory::Fear, "fear", -0.6f, 0.7f,
         {"anxiety", "terror", "dread", "worry", "nervousness", "panic"}},
        {EmotionCategory::Surprise, "surprise", 0.1f, 0.8f,
         {"astonishment", "amazement", "shock", "wonder", "confusion", "disbelief"}},
        {EmotionCategory::Disgust, "disgust", -0.6f, 0.4f,
         {"revulsion", "contempt", "loathing", "aversion", "disapproval", "distaste"}}
    };
    
    int id = 0;
    for (size_t layer = 0; layer < bases.size(); ++layer) {
        const auto& base = bases[layer];
        
        for (size_t sub = 0; sub < base.subEmotions.size(); ++sub) {
            // Create sub-emotion node
            EmotionNode node;
            node.id = id++;
            node.name = base.subEmotions[sub];
            node.category = base.name;
            node.categoryEnum = base.category;
            node.layerIndex = static_cast<int>(layer);
            node.subIndex = static_cast<int>(sub);
            node.subSubIndex = 0;
            
            // Vary valence/arousal slightly from base
            float subVariance = (static_cast<float>(sub) - 2.5f) / 10.0f;
            node.valence = std::clamp(base.valence + subVariance, -1.0f, 1.0f);
            node.arousal = std::clamp(base.arousal + subVariance * 0.5f, 0.0f, 1.0f);
            node.dominance = 0.5f + subVariance;
            node.intensity = 0.5f + std::abs(subVariance);
            
            // Set musical attributes based on emotion
            node.musicalAttributes.tempoModifier = 0.8f + node.arousal * 0.4f;
            node.musicalAttributes.mode = (node.valence < 0) ? "minor" : "major";
            node.musicalAttributes.dynamics = 0.3f + node.arousal * 0.5f;
            node.musicalAttributes.articulation = 0.5f - node.arousal * 0.3f;
            node.musicalAttributes.dissonance = std::max(0.0f, -node.valence * 0.5f);
            
            // Suggest rule breaks
            if (node.name == "grief" || node.name == "melancholy") {
                node.musicalAttributes.suggestedRuleBreaks.push_back(RuleBreakType::ModalMixture);
            }
            if (node.name == "defiance" || node.name == "rage") {
                node.musicalAttributes.suggestedRuleBreaks.push_back(RuleBreakType::ParallelMotion);
            }
            if (node.name == "anxiety" || node.name == "dread") {
                node.musicalAttributes.suggestedRuleBreaks.push_back(RuleBreakType::UnresolvedTension);
            }
            
            nodes_[node.id] = node;
            nameIndex_[node.name] = node.id;
        }
    }
    
    // Add common synonyms
    synonymIndex_["sad"] = nameIndex_["sadness"];
    synonymIndex_["happy"] = nameIndex_["happiness"];
    synonymIndex_["angry"] = nameIndex_["anger"];
    synonymIndex_["scared"] = nameIndex_["fear"];
    synonymIndex_["worried"] = nameIndex_["anxiety"];
    synonymIndex_["hopeless"] = nameIndex_["despair"];
    synonymIndex_["lonely"] = nameIndex_["loneliness"];
    synonymIndex_["mad"] = nameIndex_["frustration"];
    synonymIndex_["upset"] = nameIndex_["disappointment"];
    synonymIndex_["devastated"] = nameIndex_["grief"];
    synonymIndex_["heartbroken"] = nameIndex_["grief"];
    synonymIndex_["lost"] = nameIndex_["loneliness"];
}

const EmotionNode* EmotionThesaurus::findByName(const std::string& name) const {
    auto it = nameIndex_.find(name);
    if (it != nameIndex_.end()) {
        auto nodeIt = nodes_.find(it->second);
        if (nodeIt != nodes_.end()) {
            return &nodeIt->second;
        }
    }
    return nullptr;
}

const EmotionNode* EmotionThesaurus::findById(int id) const {
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? &it->second : nullptr;
}

const EmotionNode* EmotionThesaurus::findByPosition(int layer, int sub, int subSub) const {
    for (const auto& [id, node] : nodes_) {
        if (node.layerIndex == layer && node.subIndex == sub && node.subSubIndex == subSub) {
            return &node;
        }
    }
    return nullptr;
}

const EmotionNode* EmotionThesaurus::resolveVernacular(const std::string& vernacular) const {
    // Try direct name first
    if (auto node = findByName(vernacular)) return node;
    
    // Try synonyms
    auto it = synonymIndex_.find(vernacular);
    if (it != synonymIndex_.end()) {
        return findById(it->second);
    }
    
    // Lowercase search
    std::string lower = vernacular;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (auto node = findByName(lower)) return node;
    
    it = synonymIndex_.find(lower);
    if (it != synonymIndex_.end()) {
        return findById(it->second);
    }
    
    return nullptr;
}

std::vector<const EmotionNode*> EmotionThesaurus::findByValence(float minVal, float maxVal) const {
    std::vector<const EmotionNode*> result;
    for (const auto& [id, node] : nodes_) {
        if (node.valence >= minVal && node.valence <= maxVal) {
            result.push_back(&node);
        }
    }
    return result;
}

std::vector<const EmotionNode*> EmotionThesaurus::findByArousal(float minArousal, float maxArousal) const {
    std::vector<const EmotionNode*> result;
    for (const auto& [id, node] : nodes_) {
        if (node.arousal >= minArousal && node.arousal <= maxArousal) {
            result.push_back(&node);
        }
    }
    return result;
}

float EmotionThesaurus::distance(const EmotionNode& a, const EmotionNode& b) const {
    float dv = a.valence - b.valence;
    float da = a.arousal - b.arousal;
    float dd = a.dominance - b.dominance;
    return std::sqrt(dv*dv + da*da + dd*dd);
}

std::vector<const EmotionNode*> EmotionThesaurus::findNearby(const EmotionNode& node, float threshold) const {
    std::vector<const EmotionNode*> result;
    for (const auto& [id, other] : nodes_) {
        if (id != node.id && distance(node, other) <= threshold) {
            result.push_back(&other);
        }
    }
    return result;
}

std::vector<const EmotionNode*> EmotionThesaurus::getCategory(EmotionCategory category) const {
    std::vector<const EmotionNode*> result;
    for (const auto& [id, node] : nodes_) {
        if (node.categoryEnum == category) {
            result.push_back(&node);
        }
    }
    return result;
}

bool EmotionThesaurus::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return loadFromJson(buffer.str());
}

bool EmotionThesaurus::loadFromJson(const std::string& json) {
    // TODO: Implement JSON parsing
    // For now, use default initialization
    return true;
}

// =============================================================================
// IntentPipeline Implementation
// =============================================================================

IntentPipeline::IntentPipeline() 
    : thesaurus_(std::make_shared<EmotionThesaurus>()) {
    enabledRuleBreaks_.fill(true);
    ruleBreakIntensities_.fill(0.5f);
}

IntentPipeline::IntentPipeline(std::shared_ptr<EmotionThesaurus> thesaurus)
    : thesaurus_(std::move(thesaurus)) {
    enabledRuleBreaks_.fill(true);
    ruleBreakIntensities_.fill(0.5f);
}

EmotionNode IntentPipeline::resolveEmotion(const Wound& wound) {
    // Try to find emotion from wound description
    if (auto node = thesaurus_->resolveVernacular(wound.expression)) {
        return *node;
    }
    
    // Parse description for emotion keywords
    std::string desc = wound.description;
    std::transform(desc.begin(), desc.end(), desc.begin(), ::tolower);
    
    std::vector<std::pair<std::string, const EmotionNode*>> matches;
    
    // Check for emotion keywords
    for (const auto& keyword : {"grief", "loss", "sad", "angry", "fear", "joy", "love"}) {
        if (desc.find(keyword) != std::string::npos) {
            if (auto node = thesaurus_->resolveVernacular(keyword)) {
                matches.push_back({keyword, node});
            }
        }
    }
    
    if (!matches.empty()) {
        return *matches[0].second;
    }
    
    // Default to melancholy for unknown wounds
    if (auto node = thesaurus_->findByName("melancholy")) {
        return *node;
    }
    
    // Fallback
    EmotionNode fallback;
    fallback.name = "melancholy";
    fallback.valence = -0.5f;
    fallback.arousal = 0.3f;
    return fallback;
}

std::vector<RuleBreak> IntentPipeline::determineRuleBreaks(
    const EmotionNode& emotion, const Wound& wound) {
    
    std::vector<RuleBreak> breaks;
    
    // Add suggested rule breaks from emotion
    for (auto type : emotion.musicalAttributes.suggestedRuleBreaks) {
        size_t idx = static_cast<size_t>(type);
        if (enabledRuleBreaks_[idx]) {
            RuleBreak rb;
            rb.type = type;
            rb.intensity = ruleBreakIntensities_[idx] * emotion.intensity;
            rb.justification = "Emotional authenticity for " + emotion.name;
            
            switch (type) {
                case RuleBreakType::ModalMixture:
                    rb.description = "Borrow chords from parallel minor/major";
                    break;
                case RuleBreakType::ParallelMotion:
                    rb.description = "Move voices in parallel (defy counterpoint)";
                    break;
                case RuleBreakType::UnresolvedTension:
                    rb.description = "Leave dissonance unresolved";
                    break;
                default:
                    rb.description = ruleBreakToString(type);
            }
            
            breaks.push_back(rb);
        }
    }
    
    // Check wound urgency for additional breaks
    if (wound.urgency > 0.7f && enabledRuleBreaks_[static_cast<size_t>(RuleBreakType::DynamicContrast)]) {
        RuleBreak rb;
        rb.type = RuleBreakType::DynamicContrast;
        rb.intensity = wound.urgency;
        rb.description = "Extreme dynamic shifts";
        rb.justification = "High emotional urgency";
        breaks.push_back(rb);
    }
    
    return breaks;
}

std::vector<std::string> IntentPipeline::generateProgression(
    const EmotionNode& emotion, const std::vector<RuleBreak>& breaks) {
    
    std::vector<std::string> progression;
    bool useBorrowed = false;
    
    for (const auto& rb : breaks) {
        if (rb.type == RuleBreakType::ModalMixture) {
            useBorrowed = true;
            break;
        }
    }
    
    // Generate based on valence
    if (emotion.valence > 0.3f) {
        // Positive: I-V-vi-IV
        progression = {"I", "V", "vi", "IV"};
    } else if (emotion.valence > -0.3f) {
        // Neutral/bittersweet: I-V-vi-iv (borrowed iv)
        progression = useBorrowed ? 
            std::vector<std::string>{"I", "V", "vi", "iv"} :
            std::vector<std::string>{"I", "IV", "vi", "V"};
    } else {
        // Negative: vi-IV-I-V or i-bVI-bIII-bVII
        progression = useBorrowed ?
            std::vector<std::string>{"i", "bVI", "bIII", "bVII"} :
            std::vector<std::string>{"vi", "IV", "I", "V"};
    }
    
    return progression;
}

MusicalAttributes IntentPipeline::deriveAttributes(const EmotionNode& emotion) {
    return emotion.musicalAttributes;
}

IntentResult IntentPipeline::processWound(const Wound& wound) {
    EmotionNode emotion = resolveEmotion(wound);
    emotion = wound.primaryEmotion.id != 0 ? wound.primaryEmotion : emotion;
    
    auto ruleBreaks = determineRuleBreaks(emotion, wound);
    auto progression = generateProgression(emotion, ruleBreaks);
    auto attributes = deriveAttributes(emotion);
    
    IntentResult result;
    result.key = defaultKey_;
    result.mode = attributes.mode;
    result.tempoBpm = static_cast<int>(defaultTempo_ * attributes.tempoModifier);
    result.chordProgression = progression;
    result.ruleBreaks = ruleBreaks;
    
    result.melodicRange = 0.5f + emotion.arousal * 0.3f;
    result.leapProbability = emotion.arousal * 0.4f;
    result.allowChromaticism = attributes.dissonance > 0.3f;
    
    result.syncopationLevel = emotion.arousal * 0.5f;
    result.humanization = 0.1f + (1.0f - emotion.arousal) * 0.1f;
    
    result.baseVelocity = attributes.dynamics;
    result.dynamicRange = emotion.intensity * 0.5f;
    
    result.sourceWound = wound;
    result.confidence = 0.8f;
    
    // Production notes
    if (emotion.valence < -0.3f) {
        result.productionNotes.push_back("Consider sparse arrangement");
        result.productionNotes.push_back("Room for breath between phrases");
    }
    if (emotion.arousal > 0.6f) {
        result.productionNotes.push_back("Build energy through arrangement");
    }
    
    return result;
}

IntentResult IntentPipeline::processEmotion(const EmotionNode& emotion, float intensity) {
    Wound wound;
    wound.description = "Feeling " + emotion.name;
    wound.urgency = intensity;
    wound.primaryEmotion = emotion;
    wound.primaryEmotion.intensity = intensity;
    
    return processWound(wound);
}

IntentResult IntentPipeline::processText(const std::string& description) {
    Wound wound;
    wound.description = description;
    wound.urgency = 0.5f;
    
    return processWound(wound);
}

void IntentPipeline::enableRuleBreak(RuleBreakType type, bool enabled) {
    enabledRuleBreaks_[static_cast<size_t>(type)] = enabled;
}

void IntentPipeline::setRuleBreakIntensity(RuleBreakType type, float intensity) {
    ruleBreakIntensities_[static_cast<size_t>(type)] = std::clamp(intensity, 0.0f, 1.0f);
}

// =============================================================================
// KellyBrain Implementation
// =============================================================================

KellyBrain::KellyBrain() 
    : pipeline_(std::make_unique<IntentPipeline>()) {}

bool KellyBrain::initialize(const std::string& dataPath) {
    // Load emotion data if available
    std::string emotionPath = dataPath + "/emotions.json";
    pipeline_->thesaurus().loadFromFile(emotionPath);
    
    initialized_ = true;
    return true;
}

IntentResult KellyBrain::fromWound(const Wound& wound) {
    return pipeline_->processWound(wound);
}

IntentResult KellyBrain::fromText(const std::string& description) {
    return pipeline_->processText(description);
}

IntentResult KellyBrain::fromEmotion(const std::string& emotionName, float intensity) {
    if (auto node = pipeline_->thesaurus().resolveVernacular(emotionName)) {
        return pipeline_->processEmotion(*node, intensity);
    }
    
    // Fallback
    EmotionNode fallback;
    fallback.name = emotionName;
    fallback.valence = 0.0f;
    fallback.arousal = 0.5f;
    fallback.intensity = intensity;
    return pipeline_->processEmotion(fallback, intensity);
}

GeneratedMidi KellyBrain::generateMidi(const IntentResult& intent, int bars) {
    GeneratedMidi result;
    result.tempoBpm = intent.tempoBpm;
    result.bars = bars;
    result.key = intent.key;
    result.mode = intent.mode;
    
    // Convert roman numerals to chords
    for (const auto& roman : intent.chordProgression) {
        Chord chord;
        chord.romanNumeral = roman;
        chord.symbol = roman;  // TODO: Convert properly
        result.chords.push_back(chord);
    }
    
    // Generate basic notes
    std::random_device rd;
    std::mt19937 gen(rd());
    
    int ticksPerBar = TICKS_PER_BEAT * 4;
    int baseNote = noteNameToMidi(intent.key + "4");
    
    for (int bar = 0; bar < bars; ++bar) {
        // Root note for each bar
        MidiNote note;
        note.pitch = baseNote;
        note.startTick = bar * ticksPerBar;
        note.durationTicks = ticksPerBar;
        note.velocity = static_cast<int>(intent.baseVelocity * 127);
        result.notes.push_back(note);
    }
    
    return result;
}

std::string KellyBrain::woundToDescription(const Wound& wound) {
    return wound.description + " - " + wound.expression;
}

Wound KellyBrain::descriptionToWound(const std::string& description) {
    Wound wound;
    wound.description = description;
    wound.urgency = 0.5f;
    return wound;
}

} // namespace kelly
