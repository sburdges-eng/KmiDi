#include "engine/KellyBrain.h"
// KellyBrain.h includes KellyTypes.h, so Wound, EmotionNode, etc. are
// KellyTypes versions Now we create aliases for the KellyTypes versions before
// Types.h redefines them
namespace kelly {
// Alias KellyTypes versions before Types.h redefines them
using KellyTypesWound = Wound;
using KellyTypesEmotionNode = EmotionNode;
using KellyTypesIntentResult = IntentResult;
using KellyTypesRuleBreak = RuleBreak;
using KellyTypesRuleBreakType = RuleBreakType;
} // namespace kelly

// Now include IntentPipeline.h - this brings in Types.h which redefines the
// types. Must include before using IntentPipeline as complete type.
#include "common/Types.h" // Explicit include - this redefines Wound, EmotionNode, etc.
#include "common/IntentIRAdapter.h"  // IntentFrame support
#include "engine/IntentPipeline.h" // Full definition needed for std::unique_ptr<IntentPipeline>
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace kelly {

// Helper function to convert EmotionCategory enum to string
static std::string categoryEnumToString(EmotionCategory cat) {
  const char *catNames[] = {"Joy",      "Sadness", "Anger", "Fear",
                            "Surprise", "Disgust", "Trust", "Anticipation"};
  int catIdx = static_cast<int>(cat);
  if (catIdx >= 0 && catIdx < 8) {
    return std::string(catNames[catIdx]);
  }
  return "Joy";
}

// Conversion helpers between KellyTypes.h and Types.h structures
// These work by manually copying fields between compatible structures
namespace {
// Convert KellyTypes::Wound to Types::Wound
Wound convertToLegacyWound(const KellyTypesWound &unified) {
  Wound legacy; // This is Types::Wound now
  legacy.description = unified.description;
  legacy.intensity = unified.intensity; // Use compatibility field
  legacy.source = unified.source;       // Use compatibility field
  return legacy;
}

// Convert Types::IntentResult to KellyTypes::IntentResult
KellyTypesIntentResult
convertFromLegacyIntentResult(const IntentResult &legacy) {
  KellyTypesIntentResult unified; // This is KellyTypes::IntentResult

  // Map wound to sourceWound
  unified.sourceWound.description = legacy.sourceWound.description;
  unified.sourceWound.intensity = legacy.sourceWound.intensity;
  unified.sourceWound.urgency =
      legacy.sourceWound.intensity; // urgency = intensity
  unified.sourceWound.source = legacy.sourceWound.source;
  unified.sourceWound.desire = legacy.sourceWound.source;

  // Map emotion to sourceWound.primaryEmotion and also set emotion
  // compatibility field
  unified.sourceWound.primaryEmotion.id = legacy.emotion.id;
  unified.sourceWound.primaryEmotion.name = legacy.emotion.name;
  unified.sourceWound.primaryEmotion.categoryEnum = legacy.emotion.categoryEnum;
  unified.sourceWound.primaryEmotion.category =
      categoryEnumToString(legacy.emotion.categoryEnum);
  unified.sourceWound.primaryEmotion.valence = legacy.emotion.valence;
  unified.sourceWound.primaryEmotion.arousal = legacy.emotion.arousal;
  unified.sourceWound.primaryEmotion.dominance = legacy.emotion.dominance;
  unified.sourceWound.primaryEmotion.intensity = legacy.emotion.intensity;

  // Also set emotion compatibility field (should match primaryEmotion)
  unified.emotion.id = legacy.emotion.id;
  unified.emotion.name = legacy.emotion.name;
  unified.emotion.categoryEnum = legacy.emotion.categoryEnum;
  unified.emotion.category = categoryEnumToString(legacy.emotion.categoryEnum);
  unified.emotion.valence = legacy.emotion.valence;
  unified.emotion.arousal = legacy.emotion.arousal;
  unified.emotion.dominance = legacy.emotion.dominance;
  unified.emotion.intensity = legacy.emotion.intensity;

  // Set tempo from tempoBpm (convert BPM to modifier)
  unified.tempo = static_cast<float>(unified.tempoBpm) / 120.0f;

  // Map musical parameters
  unified.mode = legacy.mode;
  unified.tempoBpm =
      static_cast<int>(120 * legacy.tempo); // tempo is a multiplier
  unified.syncopationLevel = legacy.syncopationLevel;
  unified.humanization = legacy.humanization;
  unified.dynamicRange = legacy.dynamicRange;
  unified.allowChromaticism = legacy.allowDissonance;

  // Convert rule breaks
  unified.ruleBreaks.clear();
  for (const auto &rb : legacy.ruleBreaks) {
    KellyTypesRuleBreak unifiedRb; // KellyTypes::RuleBreak
    // Map RuleBreakType enum values
    // legacy.ruleBreaks uses Types.h RuleBreakType (Harmony, Rhythm, etc.)
    // unified uses KellyTypes.h RuleBreakType (ModalMixture, CrossRhythm, etc.)
    // At this point, RuleBreakType refers to Types.h version (included last)
    // So we use integer values to map to KellyTypes version
    switch (rb.type) {
    case RuleBreakType::ModalMixture: // Types.h version
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(1); // ModalMixture
      break;
    case RuleBreakType::CrossRhythm:
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(4); // CrossRhythm
      break;
    case RuleBreakType::DynamicContrast:
      unifiedRb.type =
          static_cast<KellyTypesRuleBreakType>(6); // DynamicContrast
      break;
    case RuleBreakType::RegisterShift:
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(5); // RegisterShift
      break;
    case RuleBreakType::HarmonicAmbiguity:
      unifiedRb.type =
          static_cast<KellyTypesRuleBreakType>(7); // HarmonicAmbiguity
      break;
    default:
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(0); // None
    }
    unifiedRb.description = rb.description;
    unifiedRb.justification = rb.justification;
    unifiedRb.intensity = rb.intensity;
    unified.ruleBreaks.push_back(unifiedRb);
  }

  // Set defaults for unified-only fields
  unified.key = "C";
  unified.timeSignature = {4, 4};
  unified.chordProgression.clear();
  unified.melodicRange = std::clamp(legacy.melodicRange, 0.0f, 1.0f);
  unified.leapProbability = std::clamp(legacy.leapProbability, 0.0f, 1.0f);
  unified.baseVelocity = std::clamp(legacy.baseVelocity, 0.0f, 1.0f);
  unified.productionNotes.clear();
  unified.confidence = std::clamp(legacy.confidence, 0.0f, 1.0f);

  // Validate tempo
  if (unified.tempoBpm < 1 || unified.tempoBpm > 300) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::convertFromLegacyIntentResult: Tempo out of range: " +
         std::to_string(unified.tempoBpm) + ", clamping").c_str());
    unified.tempoBpm = std::clamp(unified.tempoBpm, 1, 300);
  }

  // Validate emotion values
  if (unified.sourceWound.primaryEmotion.valence < -1.0f ||
      unified.sourceWound.primaryEmotion.valence > 1.0f) {
    unified.sourceWound.primaryEmotion.valence =
        std::clamp(unified.sourceWound.primaryEmotion.valence, -1.0f, 1.0f);
  }

  if (unified.sourceWound.primaryEmotion.arousal < 0.0f ||
      unified.sourceWound.primaryEmotion.arousal > 1.0f) {
    unified.sourceWound.primaryEmotion.arousal =
        std::clamp(unified.sourceWound.primaryEmotion.arousal, 0.0f, 1.0f);
  }

  return unified;
}
} // namespace

KellyBrain::KellyBrain()
    : pipeline_(std::make_unique<IntentPipeline>())
    , midiGenerator_(std::make_unique<MidiGenerator>()) {
  // IntentPipeline and MidiGenerator are initialized
}

bool KellyBrain::initialize(const std::string &dataPath) {
  // The existing IntentPipeline already initializes EmotionThesaurus
  // This could load additional data if needed
  initialized_ = true;
  return true;
}

KellyTypesIntentResult KellyBrain::fromWound(const KellyTypesWound &wound) {
  // Validate wound input
  if (wound.description.empty()) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        "KellyBrain::fromWound: Empty wound description");
  }

  if (wound.intensity < 0.0f || wound.intensity > 1.0f) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::fromWound: Intensity out of range: " +
         std::to_string(wound.intensity) + ", clamping").c_str());
  }

  if (wound.urgency < 0.0f || wound.urgency > 1.0f) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::fromWound: Urgency out of range: " +
         std::to_string(wound.urgency) + ", clamping").c_str());
  }

  // Validate emotion if present
  if (wound.primaryEmotion.id < 0) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        "KellyBrain::fromWound: Invalid emotion ID");
  }

  // Wound parameter is KellyTypes::Wound (from header via alias)
  // Convert to Types::Wound for IntentPipeline
  Wound legacyWound = convertToLegacyWound(wound); // Wound here is Types::Wound

  // Call IntentPipeline with legacy types
  IntentResult legacyResult =
      pipeline_->process(legacyWound); // IntentResult is Types::IntentResult

  // Convert result back to unified types (KellyTypes::IntentResult)
  auto result = convertFromLegacyIntentResult(legacyResult);

  // Validate result
  if (result.tempoBpm < 1 || result.tempoBpm > 300) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::fromWound: Tempo out of range: " +
         std::to_string(result.tempoBpm) + ", clamping").c_str());
    result.tempoBpm = std::clamp(result.tempoBpm, 1, 300);
  }

  return result;
}

KellyTypesIntentResult KellyBrain::fromJourney(const SideA &current,
                                               const SideB &desired) {
  // Validate inputs
  if (current.intensity < 0.0f || current.intensity > 1.0f) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::fromJourney: Current intensity out of range: " +
         std::to_string(current.intensity) + ", clamping").c_str());
  }

  if (desired.intensity < 0.0f || desired.intensity > 1.0f) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::fromJourney: Desired intensity out of range: " +
         std::to_string(desired.intensity) + ", clamping").c_str());
  }

  // SideA/SideB parameters are KellyTypes versions (from header)
  // Types.h has SideA and SideB with same structure, so we can use them
  // directly Both have: description, intensity, emotionId
  SideA legacyCurrent; // Types::SideA
  legacyCurrent.description = current.description;
  legacyCurrent.intensity = std::clamp(current.intensity, 0.0f, 1.0f);
  legacyCurrent.emotionId = current.emotionId;

  SideB legacyDesired; // Types::SideB
  legacyDesired.description = desired.description;
  legacyDesired.intensity = std::clamp(desired.intensity, 0.0f, 1.0f);
  legacyDesired.emotionId = desired.emotionId;

  // Call IntentPipeline
  IntentResult legacyResult = pipeline_->processJourney(
      legacyCurrent, legacyDesired); // Types::IntentResult

  // Convert result back to unified types
  auto result = convertFromLegacyIntentResult(legacyResult);

  // Validate result
  if (result.tempoBpm < 1 || result.tempoBpm > 300) {
    result.tempoBpm = std::clamp(result.tempoBpm, 1, 300);
  }

  return result;
}

KellyTypesIntentResult KellyBrain::fromText(const std::string &description) {
  // Create a wound from text description
  Wound wound = descriptionToWound(description);
  return fromWound(wound);
}

KellyTypesIntentResult KellyBrain::fromEmotion(const std::string &emotionName,
                                               float intensity) {
  // Validate inputs
  if (emotionName.empty()) {
    penta::getLogger().logRT(penta::LogLevel::Error,
        "KellyBrain::fromEmotion: Empty emotion name");
    // Return default result
    return fromText("Feeling unknown");
  }

  if (intensity < 0.0f || intensity > 1.0f) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::fromEmotion: Intensity out of range: " +
         std::to_string(intensity) + ", clamping").c_str());
    intensity = std::clamp(intensity, 0.0f, 1.0f);
  }

  // Look up emotion in thesaurus (returns Types.h EmotionNode)
  auto emotionOpt = pipeline_->thesaurus().findByName(emotionName);
  if (emotionOpt) {
    // Create wound from emotion
    KellyTypesWound wound;
    wound.description = "Feeling " + emotionName;
    wound.urgency = intensity;
    wound.intensity = intensity;
    wound.source = "emotion_selection";
    wound.expression = "Emotion: " + emotionName;

    // Set primary emotion from thesaurus result
    wound.primaryEmotion.id = emotionOpt->id;
    wound.primaryEmotion.name = emotionOpt->name;
    wound.primaryEmotion.categoryEnum =
        emotionOpt->categoryEnum; // Use categoryEnum, not category
    wound.primaryEmotion.category =
        categoryEnumToString(emotionOpt->categoryEnum);
    wound.primaryEmotion.valence = emotionOpt->valence;
    wound.primaryEmotion.arousal = emotionOpt->arousal;
    wound.primaryEmotion.dominance = emotionOpt->dominance;
    wound.primaryEmotion.intensity = emotionOpt->intensity;

    return fromWound(wound);
  }

  // Fallback: create basic wound
  return fromText("Feeling " + emotionName);
}

GeneratedMidi KellyBrain::generateMidi(const KellyTypesIntentResult &intent,
                                       int bars) {
  // Validate inputs
  if (bars < 1 || bars > 1000) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::generateMidi: Bars out of range: " +
         std::to_string(bars) + ", clamping").c_str());
    bars = std::clamp(bars, 1, 1000);
  }

  if (intent.tempoBpm < 1 || intent.tempoBpm > 300) {
    penta::getLogger().logRT(penta::LogLevel::Warning,
        ("KellyBrain::generateMidi: Tempo out of range: " +
         std::to_string(intent.tempoBpm) + ", clamping").c_str());
  }

  // Fallback if generator is not available
  if (!midiGenerator_) {
    penta::getLogger().logRT(penta::LogLevel::Error,
        "KellyBrain::generateMidi: MidiGenerator not available, returning fallback");
    GeneratedMidi fallback;
    fallback.tempoBpm = std::clamp(intent.tempoBpm, 1, 300);
    fallback.bars = bars;
    fallback.key = intent.key;
    fallback.mode = intent.mode;
    fallback.lengthInBeats = static_cast<double>(bars) * 4.0;
    fallback.bpm = static_cast<float>(fallback.tempoBpm);
    return fallback;
  }

  // Prepare intent for MidiGenerator (sync emotion and tempo modifier)
  IntentResult intentForGenerator = intent;
  intentForGenerator.emotion = intent.sourceWound.primaryEmotion;
  intentForGenerator.tempo =
      static_cast<float>(intent.tempoBpm) / 120.0f; // Normalize around 120 BPM

  // Derive complexity from intent parameters
  // Complexity combines: melodic range, leap probability, rule breaks, harmonic complexity
  float melodic_complexity = (intent.melodicRange + intent.leapProbability) / 2.0f;
  float rule_break_complexity = std::min(static_cast<float>(intent.ruleBreaks.size()) / 5.0f, 1.0f);
  float harmonic_complexity = intent.allowChromaticism ? 0.7f : 0.3f;
  const float complexity = (melodic_complexity * 0.4f + rule_break_complexity * 0.3f + harmonic_complexity * 0.3f);

  const float humanize = intent.humanization;

  // Derive feel from syncopation and swing
  // Feel represents the "groove" or rhythmic character
  // Combines syncopation (off-beat emphasis) and swing (triplet feel)
  const float feel = std::clamp((intent.syncopationLevel * 0.6f + intent.swingAmount * 0.4f), 0.0f, 1.0f);

  const float dynamics = intent.dynamicRange;

  GeneratedMidi result = midiGenerator_->generate(
      intentForGenerator, bars, complexity, humanize, feel, dynamics);

  // Ensure metadata is populated
  result.tempoBpm = intent.tempoBpm;
  result.bars = bars;
  result.key = intent.key;
  result.mode = intent.mode;
  result.lengthInBeats = static_cast<double>(bars) * 4.0;
  result.bpm = static_cast<float>(intent.tempoBpm);

  return result;
}

GeneratedMidi KellyBrain::generateMidiFromWound(const KellyTypesWound &wound,
                                                int bars) {
  KellyTypesIntentResult result = fromWound(wound);
  return generateMidi(result, bars);
}

KellyTypesEmotionNode
KellyBrain::resolveEmotionByName(const std::string &emotionName) {
  // Try to find emotion in thesaurus (returns Types.h EmotionNode)
  auto emotionOpt = pipeline_->thesaurus().findByName(emotionName);
  if (emotionOpt) {
    // Convert Types.h EmotionNode to KellyTypes.h EmotionNode
    KellyTypesEmotionNode unified; // KellyTypes::EmotionNode
    unified.id = emotionOpt->id;
    unified.name = emotionOpt->name;
    unified.categoryEnum =
        emotionOpt->categoryEnum; // Use categoryEnum, not category
    unified.category = categoryEnumToString(emotionOpt->categoryEnum);
    unified.valence = emotionOpt->valence;
    unified.arousal = emotionOpt->arousal;
    unified.dominance = emotionOpt->dominance;
    unified.intensity = emotionOpt->intensity;
    unified.relatedEmotions = emotionOpt->relatedEmotions;
    // Set defaults for unified-only fields
    unified.synonyms.clear();
    unified.layerIndex = 0;
    unified.subIndex = 0;
    unified.subSubIndex = 0;
    return unified;
  }

  // Fallback: create a basic emotion node (KellyTypes::EmotionNode)
  KellyTypesEmotionNode fallback;
  fallback.name = emotionName;
  fallback.intensity = 0.5f;
  fallback.valence = 0.0f;
  fallback.arousal = 0.5f;
  fallback.dominance = 0.5f;
  fallback.categoryEnum = static_cast<EmotionCategory>(0); // Joy = 0
  fallback.category = "Joy";

  return fallback;
}

std::string KellyBrain::woundToDescription(const KellyTypesWound &wound) {
  if (!wound.expression.empty()) {
    return wound.description + " - " + wound.expression;
  }
  return wound.description;
}

KellyTypesWound KellyBrain::descriptionToWound(const std::string &description,
                                               float intensity) {
  KellyTypesWound wound;
  wound.description = description;
  wound.urgency = intensity;
  wound.intensity = intensity;
  wound.source = "text_input";
  wound.expression = description;
  return wound;
}

// Implement accessor methods that require IntentPipeline definition
IntentPipeline &KellyBrain::pipeline() { return *pipeline_; }

const IntentPipeline &KellyBrain::pipeline() const { return *pipeline_; }

IntentPipeline &KellyBrain::getIntentPipeline() { return *pipeline_; }

const IntentPipeline &KellyBrain::getIntentPipeline() const {
  return *pipeline_;
}

EmotionThesaurus &KellyBrain::thesaurus() { return pipeline_->thesaurus(); }

const EmotionThesaurus &KellyBrain::thesaurus() const {
  return pipeline_->thesaurus();
}

// NEW: IntentFrame-based methods
IntentFrame KellyBrain::fromWoundToIntentFrame(const Wound &wound) {
  // Convert KellyTypes::Wound to Types::Wound
  Wound legacyWound = convertToLegacyWound(wound);

  // Use IntentPipeline's new IntentFrame method
  return pipeline_->processToIntentFrame(legacyWound, 0);  // sessionId = 0 for now
}

IntentFrame KellyBrain::fromJourneyToIntentFrame(const SideA &current, const SideB &desired) {
  // Convert KellyTypes::SideA/SideB to Types::SideA/SideB
  SideA legacyCurrent;
  legacyCurrent.description = current.description;
  legacyCurrent.intensity = current.intensity;
  legacyCurrent.emotionId = current.emotionId;

  SideB legacyDesired;
  legacyDesired.description = desired.description;
  legacyDesired.intensity = desired.intensity;
  legacyDesired.emotionId = desired.emotionId;

  // Use IntentPipeline's new IntentFrame method
  return pipeline_->processJourneyToIntentFrame(legacyCurrent, legacyDesired, 0);
}

IntentFrame KellyBrain::fromTextToIntentFrame(const std::string &description) {
  // descriptionToWound returns KellyTypes::Wound, which is what fromWoundToIntentFrame expects
  KellyTypesWound wound = descriptionToWound(description);
  return fromWoundToIntentFrame(wound);
}

IntentFrame KellyBrain::fromEmotionToIntentFrame(const std::string &emotionName, float intensity) {
  // Look up emotion in thesaurus
  auto emotionOpt = pipeline_->thesaurus().findByName(emotionName);
  if (emotionOpt) {
    KellyTypesWound wound;
    wound.description = "Feeling " + emotionName;
    wound.intensity = intensity;
    wound.primaryEmotion.id = emotionOpt->id;
    wound.primaryEmotion.name = emotionOpt->name;
    wound.primaryEmotion.categoryEnum = emotionOpt->categoryEnum;
    wound.primaryEmotion.category = categoryEnumToString(emotionOpt->categoryEnum);
    wound.primaryEmotion.valence = emotionOpt->valence;
    wound.primaryEmotion.arousal = emotionOpt->arousal;
    wound.primaryEmotion.dominance = emotionOpt->dominance;
    wound.primaryEmotion.intensity = emotionOpt->intensity;
    return fromWoundToIntentFrame(wound);
  }

  // Return default frame if emotion not found
  IntentFrame frame;
  frame.meta.ir_version = INTENT_IR_VERSION;
  return frame;
}

GeneratedMidi KellyBrain::generateMidiFromIntentFrame(const IntentFrame &frame, int bars) {
  // Make a copy for validation (prepareIntentFrame modifies the frame)
  IntentFrame validatedFrame = frame;
  prepareIntentFrame(validatedFrame);  // Validate + clamp

  // Use MidiGenerator's new IntentFrame method
  return midiGenerator_->generate(validatedFrame, bars, 0.5f, 0.4f, 0.0f, 0.75f);
}

} // namespace kelly
