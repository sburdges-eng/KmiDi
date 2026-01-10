# DAiW Knowledge Base - Upload to Custom GPT

This file should be uploaded as a "Knowledge" file in your Custom GPT configuration.

---

## VERNACULAR TO TECHNICAL TRANSLATION

### Rhythmic Onomatopoeia
- **boots and cats** → Basic 4/4 (kick 1,3 / hihat 2,4) → `groove.pattern: four_on_floor`
- **untz** → EDM kick every beat → `groove.pattern: four_on_floor, tempo: 120-140`
- **boom bap** → Hip-hop (kick 1, snare 2-and/3) → `groove.pattern: boom_bap`
- **chugga** → Palm-muted eighths → `groove.feel: chug, muting: true`
- **brr/brrrap** → Trap hi-hat 32nds → `hihat.subdivision: 32, pattern: trap_roll`

### Timbre & Texture
- **fat/phat** → Full 100-300Hz, light saturation → `eq.low_mid: +3dB, saturation: light`
- **thin** → Cut below 200Hz → `eq.low: -6dB`
- **muddy** → Problem 200-500Hz → `eq.problem: mud, target: 200-500Hz`
- **crispy/crunchy** → 8-12kHz presence → `eq.presence: +2dB, dist: light_saturation`
- **warm** → Analog character, gentle high roll-off → `character: analog, warmth: 0.7`
- **bright** → Boosted 5kHz+ → `eq.high: +3dB`
- **dark** → Rolled-off highs → `eq.high: -4dB, character: dark`
- **punchy** → Fast attack compression → `comp.attack: fast, punch: high`
- **scooped** → Cut 500Hz-2kHz (metal tone) → `eq.mid: -6dB`
- **honky** → Problem 800Hz-1.2kHz → `eq.problem: honk, target: 800-1200Hz`
- **boxy** → Problem 300-600Hz → `eq.problem: box, target: 300-600Hz`
- **glassy** → Clean crystalline highs → `eq.high_shelf: +2dB, dist: none`
- **airy** → 12kHz+ air frequencies → `eq.air: +3dB, target: 12kHz+`
- **sizzle** → Cymbal presence → `eq.cymbal_presence: high`

### Groove & Feel
- **laid back** → Behind beat 5-30ms → `groove.pocket: behind, offset_ms: 15`
- **on top/pushing** → Ahead of beat → `groove.pocket: ahead, offset_ms: -10`
- **in the pocket** → Perfect lock → `groove.pocket: locked`
- **swung** → Triplet timing 60-67% → `groove.swing: 0.62`
- **straight** → Even divisions → `groove.swing: 0.5`
- **tight** → Precise timing → `groove.tightness: 0.95`
- **loose** → Human variation → `groove.humanize: 0.4`
- **driving** → Forward momentum → `groove.energy: forward`
- **breathing** → Organic tempo variation → `tempo.rubato: true`

### Mix & Production
- **glue** → Cohesive (bus comp, shared reverb) → `bus_comp: true, shared_space: true`
- **separation** → Distinct elements → `eq.carve: true, stereo.spread: wide`
- **in your face** → Aggressive, forward → `space: dry, aggression: 0.8`
- **lush** → Rich, layered → `layers: many, fx: [reverb, chorus]`
- **lo-fi/lofi** → Degraded vintage → `character: lo-fi, degradation: 0.6`
- **wet** → Heavy effects → `fx.mix: 0.6`
- **dry** → Minimal effects → `fx.mix: 0.1`
- **buried** → Too quiet → needs level boost

---

## MEME THEORY TO FORMAL THEORY

### Progression Nicknames
| Meme Name | Progression | Formal Name | Emotional Quality |
|-----------|-------------|-------------|-------------------|
| Mario Cadence | ♭VI-♭VII-I | Double Plagal | Triumphant, heroic, video game |
| Creep Progression | I-III-IV-iv | Modal Interchange | Yearning, bittersweet |
| Axis of Awesome | I-V-vi-IV | Axis Progression | Universal pop emotion |
| Andalusian Cadence | i-♭VII-♭VI-V | Phrygian Descent | Spanish, dramatic |
| Doo-Wop Changes | I-vi-IV-V | 50s Progression | Nostalgic, innocent |
| Truck Driver | Up half/whole step | Direct Modulation | Cheap energy boost |

### Mode Characters
| Mode | Meme Name | Character |
|------|-----------|-----------|
| Lydian | Simpsons theme | Bright, floating, magical |
| Dorian | Sadboy mode | Minor but hopeful |
| Phrygian | Evil villain | Dark, exotic, tense |
| Ionian | Generic happy | Stable, resolved |
| Aeolian | Basic sad | Sad, resigned |
| Melodic Minor | Jazzy minor | Sophisticated, tense |
| Locrian | Spooky mode | Unstable, eerie |

---

## RULE-BREAKING DATABASE

### HARMONY_ParallelMotion
- **Rule violated:** Parallel P5/P8 forbidden
- **Emotions:** power, defiance, medieval, unity
- **Examples:** Beethoven Symphony 6, Debussy planing, all power chords
- **Why it works:** Context shift - polyphony rules don't apply to guitars

### HARMONY_ModalInterchange
- **Rule violated:** Stay within diatonic chords
- **Emotions:** bittersweet, nostalgia, hope, earned joy
- **Examples:** Beatles "Norwegian Wood" (♭VII), Radiohead "Creep" (I-III-IV-iv)
- **Why it works:** Creates happy-to-sad ambiguity

### HARMONY_UnresolvedDissonance
- **Rule violated:** Dissonances must resolve by step
- **Emotions:** tension, anxiety, commentary
- **Examples:** Monk's semitone clusters
- **Why it works:** "Wrong notes" are meaningfully wrong

### HARMONY_TritoneSubstitution
- **Rule violated:** Tritone must resolve
- **Emotions:** sophisticated, chromatic
- **Technique:** Replace V7 with ♭II7 (share same tritone)
- **Example:** G7→D♭7→C creates chromatic bass

### HARMONY_Polytonality
- **Rule violated:** Maintain single tonal center
- **Emotions:** chaos, duality, primal
- **Examples:** Stravinsky Rite of Spring (E♭7 + E major), Petrushka chord
- **Why it works:** Neither key wins - permanent tension

### RHYTHM_MeterAmbiguity
- **Rule violated:** Maintain consistent meter
- **Emotions:** floating, unsettled, dreamlike
- **Examples:** Stravinsky "Sacrificial Dance", Radiohead "Pyramid Song"
- **Why it works:** Impossible to predict accents

### RHYTHM_ConstantDisplacement
- **Rule violated:** Cross-rhythms should resolve
- **Emotions:** anxiety, unease, instability
- **Technique:** Shift all hits late consistently

### RHYTHM_TempoFluctuation
- **Rule violated:** Maintain consistent tempo
- **Emotions:** intimacy, vulnerability, organic
- **Technique:** Gradual BPM drift (rubato)

### STRUCTURE_NonResolution
- **Rule violated:** Resolve to tonic at end
- **Emotions:** longing, grief, unfinished
- **Examples:** Chopin Prelude E minor, Radiohead double-tonic complex
- **Why it works:** Silence becomes "negative action"

### PRODUCTION_BuriedVocals
- **Rule violated:** Vocals should be upfront
- **Emotions:** dissociation, intimacy, dreams
- **Examples:** Shoegaze genre

### PRODUCTION_PitchImperfection
- **Rule violated:** Vocals should be in tune
- **Emotions:** vulnerability, honesty, rawness
- **Why it works:** Imperfection signals emotional truth

---

## EMOTION → RULE-BREAK MAPPING

| Emotion | Primary Suggestion | Secondary |
|---------|-------------------|-----------|
| bittersweet | HARMONY_ModalInterchange | |
| nostalgia | HARMONY_ModalInterchange | STRUCTURE_NonResolution |
| longing | STRUCTURE_NonResolution | HARMONY_ModalInterchange |
| grief | STRUCTURE_NonResolution | PRODUCTION_BuriedVocals |
| power | HARMONY_ParallelMotion | |
| defiance | HARMONY_ParallelMotion | |
| anxiety | RHYTHM_ConstantDisplacement | RHYTHM_MeterAmbiguity |
| chaos | HARMONY_Polytonality | RHYTHM_MeterAmbiguity |
| vulnerability | PRODUCTION_PitchImperfection | RHYTHM_TempoFluctuation |
| intimacy | RHYTHM_TempoFluctuation | PRODUCTION_PitchImperfection |
| dissociation | PRODUCTION_BuriedVocals | |
| unfinished | STRUCTURE_NonResolution | |
| tension | HARMONY_UnresolvedDissonance | |

---

## WHY RULE-BREAKING WORKS (Meta-Patterns)

1. **Context Shift** - Rules for vocal polyphony don't apply to distorted guitars
2. **Intentional Signification** - "Wrong" notes work because they're meaningfully wrong
3. **Historical Reclamation** - Parallel 5ths evoke medieval; polytonality evokes pre-tonal
4. **Emotional Authenticity** - Ambiguity expresses complex states better than resolution
5. **Timbral Considerations** - Distortion changes which intervals work (P5s stay clear)

---

## THREE-PHASE INTENT SCHEMA

### Phase 0: Core Wound/Desire
- core_event: What happened?
- core_resistance: What holds you back from saying it?
- core_longing: What do you want to feel?
- core_stakes: What's at risk?
- core_transformation: How should you feel when done?

### Phase 1: Emotional Intent
- mood_primary: Dominant emotion
- mood_secondary_tension: Internal conflict (0.0-1.0)
- imagery_texture: Visual/tactile quality
- vulnerability_scale: Low/Medium/High
- narrative_arc: Climb-to-Climax, Slow Reveal, etc.

### Phase 2: Technical Constraints
- technical_genre: Style
- technical_tempo_range: BPM
- technical_key: Key
- technical_mode: Mode variations
- technical_groove_feel: Pocket/timing
- technical_rule_to_break: Which rule and why

---

*Database compiled November 2025*
*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*
