# Song Intent Examples

This directory contains complete song intent JSON files demonstrating the three-phase intent schema.

## Files

### kelly_when_i_found_you_sleeping.json

Complete intent file for Kelly's song "When I Found You Sleeping".

**Key Features:**
- **Core Event:** Finding someone loved after they chose to leave - moment time stopped
- **Primary Emotion:** Grief with high vulnerability
- **Rule Break:** HARMONY_ModalInterchange - Progression shifts from Dm (vi) to Bb (IV) in chorus
- **Genre:** Lo-fi bedroom emo / Confessional acoustic
- **Progressions:**
  - Verse: F-C-Am-Dm (the stuck, the cycle)
  - Chorus: F-C-Am-Bb (the drop, the reveal)

**Usage:**
```bash
# Validate the intent
daiw intent validate examples_music-brain/intents/kelly_when_i_found_you_sleeping.json

# Process to generate musical elements
daiw intent process examples_music-brain/intents/kelly_when_i_found_you_sleeping.json

# Save output
daiw intent process examples_music-brain/intents/kelly_when_i_found_you_sleeping.json -o output.json
```

## Intent Schema

All intent files follow the three-phase schema:

1. **Phase 0: Song Root** - Core wound/desire interrogation
   - core_event
   - core_resistance
   - core_longing
   - core_stakes
   - core_transformation

2. **Phase 1: Song Intent** - Emotional intent
   - mood_primary
   - mood_secondary_tension
   - imagery_texture
   - vulnerability_scale
   - narrative_arc

3. **Phase 2: Technical Constraints** - Implementation details
   - technical_genre
   - technical_tempo_range
   - technical_key
   - technical_mode
   - technical_groove_feel
   - technical_rule_to_break
   - rule_breaking_justification

## Creating Your Own

```bash
# Create a new template
daiw intent new --title "My Song" -o my_song_intent.json

# Edit the file to fill in your intent
# Then process it
daiw intent process my_song_intent.json
```

## Philosophy

> "Interrogate Before Generate" - Emotional intent drives technical decisions, not the other way around.

Every rule break requires emotional justification. The tool should make you braver, not finish art for you.
