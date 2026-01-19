#include "engine/IntentPipeline.h"
#include "common/IntentIRAdapter.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cmath>
#include <cstdint>

namespace kelly {

IntentPipeline::IntentPipeline() : woundProcessor_(thesaurus_) {
}

IntentResult IntentPipeline::process(const Wound& wound) {
    // =====================================================================
    // PHASE 1: Wound → Emotion
    // =====================================================================
    // Analyze wound description and map to emotion thesaurus
    // Uses WoundProcessor for keyword matching and emotion lookup
    EmotionNode emotion = woundProcessor_.processWound(wound);

    // =====================================================================
    // PHASE 2: Emotion → Rule Breaks
    // =====================================================================
    // Generate intentional music theory violations based on emotion
    // Uses RuleBreakEngine to determine which rules to break and why
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateRuleBreaks(emotion);

    // =====================================================================
    // PHASE 3: Compile Musical Parameters
    // =====================================================================
    // Synthesize emotion + rule breaks into concrete musical parameters
    return compileMusicalParams(wound, emotion, ruleBreaks);
}

IntentResult IntentPipeline::processJourney(const SideA& current, const SideB& desired) {
    // =====================================================================
    // PHASE 1: Process both sides to emotions
    // =====================================================================
    // If emotionId is provided, use it directly; otherwise process description
    Wound sideAWound;
    sideAWound.description = current.description;
    sideAWound.intensity = current.intensity;
    sideAWound.urgency = current.intensity;
    sideAWound.source = "sideA";
    sideAWound.expression = current.description;

    Wound sideBWound;
    sideBWound.description = desired.description;
    sideBWound.intensity = desired.intensity;
    sideBWound.urgency = desired.intensity;
    sideBWound.source = "sideB";
    sideBWound.expression = desired.description;

    EmotionNode emotionA = current.emotionId
        ? thesaurus_.findById(*current.emotionId).value_or(
            woundProcessor_.processWound(sideAWound))
        : woundProcessor_.processWound(sideAWound);

    EmotionNode emotionB = desired.emotionId
        ? thesaurus_.findById(*desired.emotionId).value_or(
            woundProcessor_.processWound(sideBWound))
        : woundProcessor_.processWound(sideBWound);

    // =====================================================================
    // Create blended journey emotion
    // =====================================================================
    // Blend emotions: 40% from current (A), 60% toward desired (B)
    // This creates musical tension that resolves as the journey progresses
    float blendedValence = emotionA.valence * 0.4f + emotionB.valence * 0.6f;
    float blendedArousal = emotionA.arousal * 0.4f + emotionB.arousal * 0.6f;
    float blendedIntensity = std::max(emotionA.intensity, emotionB.intensity);

    // Find nearest emotion node in thesaurus to the blended coordinates
    EmotionNode journeyEmotion = thesaurus_.findNearest(
        blendedValence, blendedArousal, blendedIntensity);

    // Create wound representing the journey
    Wound journeyWound;
    journeyWound.description = "Journey from " + emotionA.name + " toward " + emotionB.name;
    journeyWound.intensity = blendedIntensity;
    journeyWound.urgency = blendedIntensity;
    journeyWound.source = "cassette_journey";
    journeyWound.expression = journeyWound.description;

    // =====================================================================
    // PHASE 2: Generate journey-specific rule breaks
    // =====================================================================
    // Rule breaks that serve the transition between emotions
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateJourneyRuleBreaks(emotionA, emotionB);

    // =====================================================================
    // PHASE 3: Compile musical parameters for the journey
    // =====================================================================
    return compileMusicalParams(journeyWound, journeyEmotion, ruleBreaks);
}

// =========================================================================
// PHASE 3: Compile Musical Parameters
// =========================================================================
// Synthesizes emotion coordinates and rule breaks into concrete musical
// parameters that can be used by MIDI generation engines.

IntentResult IntentPipeline::compileMusicalParams(
    const Wound& wound,
    const EmotionNode& emotion,
    const std::vector<RuleBreak>& ruleBreaks
) {
    IntentResult result;

    // Store source data
    result.sourceWound = wound;
    result.emotion = emotion;  // Compatibility field (should match sourceWound.primaryEmotion)
    result.ruleBreaks = ruleBreaks;

    // =====================================================================
    // Base parameters from emotion (via EmotionThesaurus)
    // =====================================================================
    // These are the "default" musical characteristics for this emotion
    result.mode = thesaurus_.suggestMode(emotion);
    result.tempo = thesaurus_.suggestTempoModifier(emotion);
    result.dynamicRange = thesaurus_.suggestDynamicRange(emotion);

    // =====================================================================
    // Default safe values
    // =====================================================================
    // Conservative defaults that can be overridden by rule breaks
    result.allowDissonance = false;
    result.syncopationLevel = 0.3f;
    result.humanization = 0.4f;

    // =====================================================================
    // Apply rule breaks to override/modify parameters
    // =====================================================================
    // Rule breaks represent intentional violations of music theory rules
    // for emotional authenticity. They modify the base parameters.
    for (const auto& rb : ruleBreaks) {
        switch (rb.type) {
            case RuleBreakType::ModalMixture:
                // Allow dissonant intervals and unresolved tensions
                result.allowDissonance = true;
                // Increase dynamic range when using dissonance
                result.dynamicRange = std::min(1.0f, result.dynamicRange + rb.intensity * 0.2f);
                break;

            case RuleBreakType::CrossRhythm:
                // Increase syncopation and off-beat accents
                result.syncopationLevel = std::max(result.syncopationLevel, rb.intensity);
                // More humanization for complex rhythms
                result.humanization = std::max(result.humanization, rb.intensity * 0.8f);
                break;

            case RuleBreakType::DynamicContrast:
                // Expand dynamic range for dramatic expression
                result.dynamicRange = std::max(result.dynamicRange, rb.intensity);
                // Higher dynamics often benefit from more humanization
                if (rb.intensity > 0.7f) {
                    result.humanization = std::max(result.humanization, 0.6f);
                }
                break;

            case RuleBreakType::RegisterShift:
                // Melodic rule breaks (wide leaps, chromaticism) affect
                // generation algorithms rather than direct parameters
                // The melody engine will use this rule break during generation
                break;

            case RuleBreakType::HarmonicAmbiguity:
                // Form rule breaks (structural disruption) affect
                // arrangement and song structure, not direct parameters
                // The arrangement engine will use this rule break
                break;
        }
    }

    // =====================================================================
    // Final parameter validation and clamping
    // =====================================================================
    result.tempo = std::clamp(result.tempo, 0.5f, 2.0f);
    result.dynamicRange = std::clamp(result.dynamicRange, 0.0f, 1.0f);
    result.syncopationLevel = std::clamp(result.syncopationLevel, 0.0f, 1.0f);
    result.humanization = std::clamp(result.humanization, 0.0f, 1.0f);

    return result;
}

IntentFrame IntentPipeline::processToIntentFrame(const Wound& wound, uint64_t session_id) {
    // =====================================================================
    // PHASE 1: Wound → Emotion
    // =====================================================================
    EmotionNode emotion = woundProcessor_.processWound(wound);

    // =====================================================================
    // PHASE 2: Emotion → Rule Breaks
    // =====================================================================
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateRuleBreaks(emotion);

    // =====================================================================
    // PHASE 3: Compile to IntentFrame
    // =====================================================================
    return compileToIntentFrame(wound, emotion, ruleBreaks, session_id);
}

IntentFrame IntentPipeline::processJourneyToIntentFrame(const SideA& current, const SideB& desired, uint64_t session_id) {
    // =====================================================================
    // PHASE 1: Process both sides to emotions
    // =====================================================================
    Wound sideAWound;
    sideAWound.description = current.description;
    sideAWound.intensity = current.intensity;
    sideAWound.urgency = current.intensity;
    sideAWound.source = "sideA";
    sideAWound.expression = current.description;

    Wound sideBWound;
    sideBWound.description = desired.description;
    sideBWound.intensity = desired.intensity;
    sideBWound.urgency = desired.intensity;
    sideBWound.source = "sideB";
    sideBWound.expression = desired.description;

    EmotionNode emotionA = current.emotionId
        ? thesaurus_.findById(*current.emotionId).value_or(
            woundProcessor_.processWound(sideAWound))
        : woundProcessor_.processWound(sideAWound);

    EmotionNode emotionB = desired.emotionId
        ? thesaurus_.findById(*desired.emotionId).value_or(
            woundProcessor_.processWound(sideBWound))
        : woundProcessor_.processWound(sideBWound);

    // =====================================================================
    // Create blended journey emotion
    // =====================================================================
    float blendedValence = emotionA.valence * 0.4f + emotionB.valence * 0.6f;
    float blendedArousal = emotionA.arousal * 0.4f + emotionB.arousal * 0.6f;
    float blendedIntensity = std::max(emotionA.intensity, emotionB.intensity);

    EmotionNode journeyEmotion = thesaurus_.findNearest(
        blendedValence, blendedArousal, blendedIntensity);

    Wound journeyWound;
    journeyWound.description = "Journey from " + emotionA.name + " toward " + emotionB.name;
    journeyWound.intensity = blendedIntensity;
    journeyWound.urgency = blendedIntensity;
    journeyWound.source = "cassette_journey";
    journeyWound.expression = journeyWound.description;

    // =====================================================================
    // PHASE 2: Generate journey-specific rule breaks
    // =====================================================================
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateJourneyRuleBreaks(emotionA, emotionB);

    // =====================================================================
    // PHASE 3: Compile to IntentFrame
    // =====================================================================
    return compileToIntentFrame(journeyWound, journeyEmotion, ruleBreaks, session_id);
}

IntentFrame IntentPipeline::compileToIntentFrame(
    const Wound& wound,
    const EmotionNode& emotion,
    const std::vector<RuleBreak>& ruleBreaks,
    uint64_t session_id
) {
    IntentFrame frame;

    // Meta
    frame.meta.ir_version = INTENT_IR_VERSION;
    frame.meta.intent_id = 0;  // Could generate from wound hash
    frame.meta.session_id = session_id;

    // EmotionState - map from EmotionNode
    frame.emotion.valence = emotion.valence;
    frame.emotion.arousal = emotion.arousal;
    frame.emotion.dominance = emotion.dominance;
    frame.emotion.discrete_id = emotion.id >= 0 ? static_cast<int16_t>(emotion.id) : -1;
    frame.emotion.intensity = emotion.intensity;
    frame.emotion.confidence = emotion.mlConfidence.value_or(0.8f);

    // MusicalIntent - map from emotion and rule breaks
    // Tempo bias: map from tempo modifier (0.5-2.0) to (-1.0 to +1.0)
    float tempo_mod = thesaurus_.suggestTempoModifier(emotion);
    frame.music.tempo_bias = std::clamp((tempo_mod - 1.0f) * 2.0f, -1.0f, 1.0f);

    // Rhythmic density: map from emotion density
    float density = emotion.musicalAttributes.density;
    frame.music.rhythmic_density = density;

    // Groove strength: higher for more syncopation/humanization
    float syncopation = 0.3f;
    float humanization = 0.4f;
    for (const auto& rb : ruleBreaks) {
        if (rb.type == RuleBreakType::CrossRhythm) {
            syncopation = std::max(syncopation, rb.intensity);
            humanization = std::max(humanization, rb.intensity * 0.8f);
        }
    }
    frame.music.groove_strength = std::clamp((syncopation + humanization) / 2.0f, 0.0f, 1.0f);

    // Harmonic tension: map from dissonance and rule breaks
    float tension = emotion.musicalAttributes.dissonance;
    for (const auto& rb : ruleBreaks) {
        if (rb.type == RuleBreakType::ModalMixture) {
            tension = std::max(tension, rb.intensity);
        }
    }
    frame.music.harmonic_tension = std::clamp(tension, 0.0f, 1.0f);

    // Harmonic motion: default moderate, higher for journey
    frame.music.harmonic_motion = 0.5f;

    // Mode preference: map from mode string
    std::string mode = thesaurus_.suggestMode(emotion);
    std::string mode_lower = mode;
    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);
    if (mode_lower.find("minor") != std::string::npos) {
        frame.music.mode_preference = -1;
    } else if (mode_lower.find("major") != std::string::npos) {
        frame.music.mode_preference = 1;
    } else {
        frame.music.mode_preference = 0;
    }

    // Melodic activity: map from arousal
    frame.music.melodic_activity = emotion.arousal;

    // Contour variance: higher for more emotional intensity
    frame.music.contour_variance = std::clamp(emotion.intensity, 0.0f, 1.0f);

    // Dynamic range: map from emotion dynamics
    float dyn_range = thesaurus_.suggestDynamicRange(emotion);
    for (const auto& rb : ruleBreaks) {
        if (rb.type == RuleBreakType::DynamicContrast) {
            dyn_range = std::max(dyn_range, rb.intensity);
        }
    }
    frame.music.dynamic_range = std::clamp(dyn_range, 0.0f, 1.0f);

    // Texture density: map from emotion density
    frame.music.texture_density = density;

    // Time scope - default to immediate
    frame.time.start_bar = -1;
    frame.time.end_bar = -1;
    frame.time.fade_in_beats = 0.0f;
    frame.time.fade_out_beats = 0.0f;

    // Constraints - default to all engines allowed
    frame.constraints.allowed_engines_mask = 0xFFFFFFFF;
    frame.constraints.forbidden_engines_mask = 0;
    frame.constraints.max_cpu_cost = 1.0f;
    frame.constraints.max_event_rate = 1000.0f;

    // Provenance
    frame.provenance.source = SOURCE_UI_DIRECT;  // Default, could be set based on wound.source
    frame.provenance.user_override_weight = 0.5f;

    // Clamp and validate
    prepareIntentFrame(frame);

    return frame;
}

} // namespace kelly
