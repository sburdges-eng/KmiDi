#pragma once

#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include "engine/WoundProcessor.h"
#include "engine/RuleBreakEngine.h"
#include <memory>

namespace kelly {

/**
 * The three-phase intent processing pipeline.
 * 
 * Phase 0: Core Wound/Desire - "What hurts?"
 * Phase 1: Emotional Intent - Map to 216-node thesaurus
 * Phase 2: Technical Constraints - Which rules to break and why
 */
class IntentPipeline {
public:
    IntentPipeline();
    
    /**
     * Process a complete intent from wound to musical parameters.
     * This is the main entry point for the emotion engine.
     */
    IntentResult process(const Wound& wound);
    
    /**
     * Process Side A (current state) and Side B (desired state)
     * to create a musical journey between emotions.
     */
    IntentResult processJourney(const SideA& current, const SideB& desired);
    
    /**
     * Get direct access to the thesaurus for UI emotion selection
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
};

} // namespace kelly
