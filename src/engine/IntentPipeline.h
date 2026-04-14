#pragma once
/*
 * IntentPipeline.h - Three-Phase Wound → Music Intent Compiler
 * ==============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: WoundProcessor, RuleBreakEngine, EmotionThesaurus
 * - Common Layer: IntentIRAdapter, kmidi::IntentFrame (IR v1)
 * - MIDI Layer: MidiGenerator via IntentResult consumers
 * - Engine Layer: KellyBrain (thin wrapper over this pipeline)
 *
 * Purpose: Production pipeline from wound text to IntentResult parameters.
 *
 * Features:
 * - Phase 1–3 orchestration (emotion, rule breaks, parameters)
 * - Optional IntentFrame validation path
 */

#include "common/Types.h"
#include "common/IntentIRAdapter.h"
#include "kmidi/IntentIR.h"
#include "engine/EmotionThesaurus.h"
#include "engine/WoundProcessor.h"
#include "engine/RuleBreakEngine.h"
#include <memory>

namespace kelly {

class IntentPipeline {
public:
    IntentPipeline();

    /**
     * Process a complete intent from wound to musical parameters.
     *
     * This is the main entry point for the emotion engine.
     * Executes all three phases: Wound → Emotion → Musical Params
     *
     * @param wound The emotional wound to process
     * @return IntentResult containing emotion, rule breaks, and musical parameters
     */
    IntentResult process(const Wound& wound);

    /**
     * Process Side A (current state) and Side B (desired state)
     * to create a musical journey between emotions.
     *
     * Creates a blended emotion that transitions from current to desired,
     * generating rule breaks that serve the emotional journey.
     *
     * @param current The current emotional state (Side A)
     * @param desired The desired emotional state (Side B)
     * @return IntentResult representing the journey between states
     */
    IntentResult processJourney(const SideA& current, const SideB& desired);

    /**
     * Process a complete intent from wound to IntentFrame (IR v1).
     *
     * This is the new canonical API that produces IntentFrame directly.
     * Executes all three phases: Wound → Emotion → IntentFrame
     *
     * @param wound The emotional wound to process
     * @param session_id Session identifier for tracking
     * @return IntentFrame containing emotion, musical biases, and metadata
     */
    IntentFrame processToIntentFrame(const Wound& wound, uint64_t session_id = 0);

    /**
     * Process journey to IntentFrame (IR v1).
     *
     * @param current The current emotional state (Side A)
     * @param desired The desired emotional state (Side B)
     * @param session_id Session identifier for tracking
     * @return IntentFrame representing the journey between states
     */
    IntentFrame processJourneyToIntentFrame(const SideA& current, const SideB& desired, uint64_t session_id = 0);

    /**
     * Get direct access to the thesaurus for UI emotion selection.
     *
     * Allows external code to query the emotion thesaurus for:
     * - Finding emotions by name, ID, or category
     * - Getting all 216 emotion nodes
     * - Suggesting musical parameters for emotions
     */
    const EmotionThesaurus& thesaurus() const { return thesaurus_; }
    EmotionThesaurus& thesaurus() { return thesaurus_; }

private:
    EmotionThesaurus thesaurus_;
    WoundProcessor woundProcessor_;
    RuleBreakEngine ruleBreakEngine_;

    // Phase 3: Compile musical parameters
    IntentResult compileMusicalParams(
        const Wound& wound,
        const EmotionNode& emotion,
        const std::vector<RuleBreak>& ruleBreaks
    );

    // Phase 3: Compile to IntentFrame (IR v1)
    IntentFrame compileToIntentFrame(
        const Wound& wound,
        const EmotionNode& emotion,
        const std::vector<RuleBreak>& ruleBreaks,
        uint64_t session_id
    );
};

} // namespace kelly
