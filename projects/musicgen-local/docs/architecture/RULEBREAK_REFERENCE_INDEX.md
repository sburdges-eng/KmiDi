# Rulebreak Reference Index

## Purpose
Provide a canonical crosswalk from music-theory rulebreak concepts to discovered source evidence (`.md`, `.yaml`, headers/sources), and define how `musicgen-local` should consume rulebreak semantics in future implementation.

## Rulebreak Taxonomy
- Harmony
- Rhythm
- Dynamics
- Structure
- Voice Leading
- Texture
- Range
- Chromatic/modal mixture

## Source Crosswalk

| Rulebreak Concept | Description | Emotional Trigger References | Source Files | Interface Surface (type/enum/function) |
|---|---|---|---|---|
| Harmony | Dissonance, cluster choices, unresolved tension for emotional authenticity | Negative/high-intensity states map to non-conventional harmonic behavior | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | `addHarmonyRuleBreaks(...)`, `RuleBreak` |
| Rhythm | Intentional groove/placement/regularity violations | Emotion intensity steers rhythmic instability/drive | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | `addRhythmRuleBreaks(...)`, `RuleBreakType::Rhythm` |
| Dynamics | Extreme/sudden dynamic contrast as expressive violation | Arousal/intensity-based emphasis | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | `addDynamicsRuleBreaks(...)`, `RuleBreakType::Dynamics` |
| Structure | Fragmentation/silence/rest/sectional disruption | Emotion-driven departure from stable form | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | `addStructureRuleBreaks(...)`, `RuleBreakType::Structure` |
| Voice Leading | Parallel motion and unresolved leading behavior on purpose | Grief/negative emotion references include voice-leading violations | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h`, `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/Kelly.h` | `addVoiceLeadingRuleBreaks(...)`, `RuleBreakType::Voice_Leading` |
| Texture | Layer collision/crossing density as emotional device | Category-specific breaks by emotional class | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | `addTextureRuleBreaks(...)`, `RuleBreakType::Texture` |
| Range | Extreme register and leap behavior | Emotion intensity can widen practical range limits | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | `addRangeRuleBreaks(...)`, `RuleBreakType::Range` |
| Chromatic/modal mixture | Non-diatonic/modal borrowing allowance | Config-level mixed mode settings and chromatic guidance | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/config/emotion_node_classifier.yaml`, `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/ML Kelly Training/backup/configs/emotion_node_classifier.yaml`, `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/FINAL_STATUS.md` | `mixed: [dorian, mixolydian]`, chromaticism guidance |

## Justification Requirements (Documented)
- Every rule break requires emotional justification.
- Documented references:
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/CLAUDE_AGENT_GUIDE.md`
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/IntentPipeline.h`
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h`

## Integration Notes for musicgen-local
- Treat all rulebreak behavior as `reference-imported` until local runtime implementation exists.
- Future pipeline shape:
  1. Intent parse -> emotional state extraction.
  2. Rulebreak candidate generation (taxonomy-aligned).
  3. Constraint-weighted generation with selectable break intensity.
  4. Render/export annotations preserving what was broken and why.

## Future Local Implementation Contract
Each rulebreak object should contain:
- `type`
- `description`
- `justification`
- `intensity`
- `application_scope` (`harmony` | `rhythm` | `dynamics` | `structure` | `voice_leading` | `texture` | `range` | `chromatic_modal`)

Compatibility note with current schema:
- `music-graph.schema.json` does not yet define explicit `rulebreaks` nodes.
- Forward-compatible approach: add `generation.rulebreaks[]` in a future schema revision with strict enum + justification requirements.
