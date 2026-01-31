#include "engine/EmotionThesaurusLoader.h"
#include <algorithm>
#include <cmath>

namespace kelly {

// Static counter for unique IDs across all files
static int g_nextEmotionId = 1;

int EmotionThesaurusLoader::loadFromJsonFiles(const juce::File& dataDirectory, EmotionThesaurus& thesaurus) {
    if (!dataDirectory.isDirectory()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Data directory does not exist: " + dataDirectory.getFullPathName());
        return 0;
    }
    
    // Reset global ID counter
    g_nextEmotionId = 1;
    int totalLoaded = 0;
    
    // Expected emotion JSON files
    std::vector<std::string> emotionFiles = {
        "anger.json", "joy.json", "sad.json", "fear.json",
        "disgust.json", "surprise.json"
    };
    
    for (const auto& filename : emotionFiles) {
        juce::File jsonFile = dataDirectory.getChildFile(filename);
        if (jsonFile.existsAsFile()) {
            int loaded = loadFromJsonFile(jsonFile, thesaurus);
            totalLoaded += loaded;
            juce::Logger::writeToLog("Loaded " + juce::String(loaded) + " emotions from " + filename);
        } else {
            juce::Logger::writeToLog("EmotionThesaurusLoader: File not found: " + jsonFile.getFullPathName());
        }
    }
    
    return totalLoaded;
}

int EmotionThesaurusLoader::loadFromJsonFile(const juce::File& jsonFile, EmotionThesaurus& thesaurus) {
    if (!jsonFile.existsAsFile()) {
        return 0;
    }
    
    juce::String jsonText = jsonFile.loadFileAsString();
    if (jsonText.isEmpty()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Empty or unreadable file: " + jsonFile.getFullPathName());
        return 0;
    }
    
    juce::var parsedJson = juce::JSON::parse(jsonText);
    if (!parsedJson.isObject()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Invalid JSON in file: " + jsonFile.getFullPathName());
        return 0;
    }
    
    auto* root = parsedJson.getDynamicObject();
    if (!root) return 0;
    
    std::string categoryName = root->getProperty("category").toString().toStdString();
    auto* subEmotions = root->getProperty("sub_emotions").getDynamicObject();
    
    if (!subEmotions) return 0;
    
    int loaded = 0;
    
    // Iterate through sub_emotions
    auto properties = subEmotions->getProperties();
    for (auto& prop : properties) {
        if (prop.value.isObject()) {
            int beforeCount = g_nextEmotionId;
            processSubEmotion(prop.value, categoryName, prop.name.toString().toStdString(), thesaurus, g_nextEmotionId);
            loaded += (g_nextEmotionId - beforeCount);
        }
    }
    
    return loaded;
}


EmotionCategory EmotionThesaurusLoader::categoryFromString(const std::string& categoryStr) {
    if (categoryStr == "joy" || categoryStr == "happiness") return EmotionCategory::Joy;
    if (categoryStr == "sad" || categoryStr == "sadness") return EmotionCategory::Sadness;
    if (categoryStr == "anger") return EmotionCategory::Anger;
    if (categoryStr == "fear") return EmotionCategory::Fear;
    if (categoryStr == "surprise") return EmotionCategory::Surprise;
    if (categoryStr == "disgust") return EmotionCategory::Disgust;
    if (categoryStr == "trust" || categoryStr == "love") return EmotionCategory::Trust;
    if (categoryStr == "anticipation") return EmotionCategory::Anticipation;
    return EmotionCategory::Sadness; // Default
}

float EmotionThesaurusLoader::valenceFromString(const std::string& valenceStr) {
    if (valenceStr == "positive") return 0.7f;
    if (valenceStr == "negative") return -0.7f;
    return 0.0f; // Neutral
}

float EmotionThesaurusLoader::intensityFromTier(const std::string& tierStr) {
    if (tierStr.find("1_subtle") != std::string::npos) return 0.1f;
    if (tierStr.find("2_mild") != std::string::npos) return 0.3f;
    if (tierStr.find("3_moderate") != std::string::npos) return 0.5f;
    if (tierStr.find("4_intense") != std::string::npos) return 0.7f;
    if (tierStr.find("5_overwhelming") != std::string::npos) return 0.9f;
    return 0.5f; // Default moderate
}

float EmotionThesaurusLoader::arousalFromIntensity(float intensity, EmotionCategory category) {
    // Base arousal on category and intensity
    float baseArousal = 0.5f;
    
    // High arousal categories
    if (category == EmotionCategory::Anger || category == EmotionCategory::Fear) {
        baseArousal = 0.7f + (intensity * 0.3f);
    }
    // Low arousal categories
    else if (category == EmotionCategory::Sadness) {
        baseArousal = 0.2f + (intensity * 0.3f);
    }
    // Moderate arousal
    else {
        baseArousal = 0.4f + (intensity * 0.4f);
    }
    
    return std::clamp(baseArousal, 0.0f, 1.0f);
}

void EmotionThesaurusLoader::processSubEmotion(
    const juce::var& subData,
    const std::string& categoryName,
    const std::string& subEmotionName,
    EmotionThesaurus& thesaurus,
    int& nextId)
{
    if (!subData.isObject()) return;
    
    auto* subObj = subData.getDynamicObject();
    if (!subObj) return;
    
    auto* subSubEmotions = subObj->getProperty("sub_sub_emotions").getDynamicObject();
    if (!subSubEmotions) return;
    
    auto properties = subSubEmotions->getProperties();
    for (auto& prop : properties) {
        if (prop.value.isObject()) {
            processSubSubEmotion(prop.value, categoryName, subEmotionName, 
                                prop.name.toString().toStdString(), thesaurus, nextId);
        }
    }
}

void EmotionThesaurusLoader::processSubSubEmotion(
    const juce::var& subSubData,
    const std::string& categoryName,
    const std::string& subEmotionName,
    const std::string& subSubEmotionName,
    EmotionThesaurus& thesaurus,
    int& nextId)
{
    if (!subSubData.isObject()) return;
    
    auto* subSubObj = subSubData.getDynamicObject();
    if (!subSubObj) return;
    
    auto* intensityTiers = subSubObj->getProperty("intensity_tiers").getDynamicObject();
    if (!intensityTiers) return;
    
    auto properties = intensityTiers->getProperties();
    for (auto& prop : properties) {
        if (prop.value.isArray()) {
            processIntensityTier(prop.value, categoryName, subEmotionName, 
                                subSubEmotionName, prop.name.toString().toStdString(), 
                                thesaurus, nextId);
        }
    }
}

void EmotionThesaurusLoader::processIntensityTier(
    const juce::var& tierData,
    const std::string& categoryName,
    const std::string& /* subEmotionName */,
    const std::string& /* subSubEmotionName */,
    const std::string& tierName,
    EmotionThesaurus& thesaurus,
    int& nextId)
{
    if (!tierData.isArray()) return;
    
    auto* arr = tierData.getArray();
    if (!arr) return;
    
    EmotionCategory category = categoryFromString(categoryName);
    float valence = valenceFromString(categoryName == "joy" ? "positive" : 
                                      (categoryName == "sad" || categoryName == "anger" || categoryName == "fear" ? "negative" : "neutral"));
    float intensity = intensityFromTier(tierName);
    float arousal = arousalFromIntensity(intensity, category);
    
    // Build related emotions list (simplified - would need full graph in production)
    std::vector<int> related;
    
    for (int i = 0; i < arr->size(); ++i) {
        std::string emotionWord = arr->getReference(i).toString().toStdString();
        
        EmotionNode node;
        node.id = nextId++;
        node.name = emotionWord;
        node.category = category;
        node.intensity = intensity;
        node.valence = valence;
        node.arousal = arousal;
        node.relatedEmotions = related;
        
        // Add to thesaurus
        thesaurus.addNode(node);
    }
}

} // namespace kelly

