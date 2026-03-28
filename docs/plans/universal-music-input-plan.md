# Universal Music Input — Implementation Plan

## Context

KmiDi currently has a 4-tab layout (Mix, Inspire, Create, Compose) that fragments the music creation experience. The vision: a dual-pane split-screen where **Side A** (left) contains every possible musical command organized as a browseable taxonomy with search, and **Side B** (right) is a natural language input where anyone can describe music in their own words.

**Critical design principle: Interpretive, not deterministic.** The same words must produce different music depending on context. "Dorian + fast + guitar + overdrive" = metal shred. "Dorian + slow + bass + clean" = blues. Parameters don't add — they *multiply* into emergent meaning through interaction. The system returns probability distributions, not fixed values. Each generation samples from those distributions, shaped by context and user history, so output is never predictable or formulaic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       KmiDi Header                          │
├────────────────────────┬────────────────────────────────────┤
│       SIDE A           │          SIDE B                    │
│   Musical Commands     │     Natural Language               │
│                        │                                    │
│  [🔍 Search...]        │  ┌──────────────────────────┐     │
│                        │  │ "dorian with fast guitar   │     │
│  ▼ Harmony             │  │  licks and overdrive"      │     │
│    ▼ Scales/Modes      │  └──────────────────────────┘     │
│      ◐ Dorian [0.85]   │                                    │
│      ◔ Phrygian [0.10] │  Context cluster: METAL/PROG      │
│    ▼ Chords            │  Confidence: 0.82                  │
│  ▼ Rhythm              │                                    │
│    ◐ Dense [0.90]      │  Interpretation:                   │
│  ▼ Articulation        │  [dorian 85%] [fast 16ths]         │
│    ◐ Legato [0.70]     │  [high gain] [tight picking]       │
│                        │                                    │
│  ── Interpretive ──●── │  [🎲 Reinterpret]                  │
│  Literal    Surprise   │  [📌 Pin: tempo] [📌 Pin: mode]    │
├────────────────────────┴────────────────────────────────────┤
│  ◐ Dorian·85% · ~140-170 BPM · Metal/Prog           [▶ Gen]│
└─────────────────────────────────────────────────────────────┘
```

Key differences from deterministic model:
- Nodes show **probability weights** (◐ 0.85) not binary on/off (✓/✗)
- Summary bar shows **ranges** (~140-170 BPM) not fixed values (155 BPM)
- **Interpretive slider** controls variance: Literal (reproducible) ↔ Surprise (high entropy)
- **Reinterpret button** re-samples from the same distributions for a different take
- **Pin** locks specific parameters so they survive reinterpretation

---

## Core Data Model: Interpretation Context

### The `InterpretationContext` — replaces deterministic mapping

The key insight: musical parameters don't have independent meanings. They form **context clusters** — combinations that activate genre/style fingerprints. The system recognizes these clusters and uses them to weight all other parameters.

```typescript
// src/data/interpretationContext.ts

/** A probability distribution over possible values */
interface ParamDistribution {
  type: 'gaussian' | 'discrete' | 'range';
  // Gaussian: center + spread
  center?: number;
  spread?: number;  // std deviation — higher = more surprise
  // Discrete: weighted options
  weights?: Record<string, number>;  // e.g. { dorian: 0.85, phrygian: 0.10, aeolian: 0.05 }
  // Range: min/max with bias
  min?: number;
  max?: number;
  bias?: number;  // where in the range to center sampling
}

/** A cluster of co-occurring parameters that imply a genre/style */
interface ContextCluster {
  id: string;                          // e.g. "metal-shred", "blues-slow"
  label: string;
  genreHints: string[];                // ["metal", "prog", "djent"]
  triggerConditions: TriggerRule[];     // what combination of inputs activates this
  parameterOverrides: Record<string, ParamDistribution>;  // how this cluster shifts all params
  confidence: number;                  // how strongly this cluster is activated (0-1)
}

/** What activates a context cluster */
interface TriggerRule {
  requiredNodes: string[];             // taxonomy nodes that must be active
  requiredKeywords?: string[];         // NLP keywords that trigger this
  minMatchCount?: number;              // how many conditions must match (default: all)
}
```

**Example context clusters:**

| Cluster | Triggers | Effect on Parameters |
|---------|----------|---------------------|
| `metal-shred` | dorian + fast + guitar + overdrive/distortion | tempo: gaussian(160, 15), density: 0.9+, articulation: legato+alternate picking, dynamics: f-ff |
| `blues-slow` | pentatonic + slow + bass/guitar + clean | tempo: gaussian(72, 8), timing: behind-beat, density: sparse, dynamics: mp, space: 0.3+ |
| `jazz-modal` | dorian + piano/sax + walking bass | tempo: gaussian(130, 20), swing: 0.7+, harmonic_rhythm: fast, voicings: drop-2/shell |
| `ambient-pad` | lydian + slow + pad/synth + reverb | tempo: gaussian(65, 10), density: sparse, texture: thick, dynamics: pp-p, space: 0.4+ |
| `funk-groove` | dorian/mixolydian + syncopated + bass + clean | tempo: gaussian(105, 10), groove: 0.9+, timing: ahead, density: medium, accents: strong |

The same "dorian" node participates in 5 completely different clusters. Its *meaning* depends on what else is active.

---

## Phase 1: Data Layer

### 1. Taxonomy Data — `src/data/taxonomyTree.ts`

Same 10-category tree structure, but each leaf node maps to **distributions**, not values:

```typescript
interface TaxonomyNode {
  id: string;
  label: string;
  labelKey?: string;                    // i18n key
  children?: TaxonomyNode[];
  // What this node contributes to the interpretation (as distributions)
  paramContributions?: Record<string, ParamDistribution>;
  // Which context clusters this node participates in
  clusterMemberships?: string[];        // e.g. ["metal-shred", "blues-slow", "jazz-modal"]
  keywords: string[];
  description?: string;
}
```

A node like "Dorian" doesn't say "set key_mode to dorian." It says "I contribute mode_weight {dorian: 0.85, phrygian: 0.10, aeolian: 0.05} and I participate in clusters [metal-shred, blues-slow, jazz-modal, funk-groove]."

### 2. Context Cluster Registry — `src/data/contextClusters.ts`

Static data defining ~30-50 context clusters covering major genre/style combinations. Each cluster specifies:
- Trigger conditions (which taxonomy nodes + keywords activate it)
- Parameter overrides as distributions
- Genre hints for display

Sources to populate from:
- `emotional_mapping.py` EMOTIONAL_PRESETS (already has mode_weights as distributions, tempo ranges)
- `emotion_thesaurus.py` musical hints per emotion
- IntentBuilder.tsx constants (groove feels, narrative arcs)
- Backend rule-break types and bass patterns

### 3. Interpretation Engine — `src/data/interpretationEngine.ts`

Pure TypeScript functions (runs client-side for instant feedback):

```typescript
/** Given active nodes + text keywords, find matching context clusters */
function resolveContextClusters(
  activeNodes: Set<string>,
  keywords: string[],
  clusters: ContextCluster[]
): ScoredCluster[];

/** Merge all active distributions into a single InterpretationResult */
function interpret(
  activeNodes: Set<string>,
  scoredClusters: ScoredCluster[],
  pinnedParams: Map<string, any>,     // user-pinned values (locked)
  interpretiveLevel: number,           // 0 = literal, 1 = max surprise
  userPriors?: UserPriors              // learned preferences
): InterpretationResult;

/** Sample concrete values from an InterpretationResult */
function sample(result: InterpretationResult, seed?: number): CompleteSongIntentRequest;
```

The `interpret()` function:
1. Collects `paramContributions` from all active taxonomy nodes
2. Identifies which context clusters are activated (and how strongly)
3. Applies cluster overrides — these shift/narrow the distributions based on genre context
4. Applies pinned parameters (locked by user, not affected by interpretation)
5. Scales all distribution spreads by `interpretiveLevel` (0 = tight/deterministic, 1 = wide/surprising)
6. Applies user priors from `PreferenceAnalyzer` (shifts distributions toward user's historical preferences)
7. Returns `InterpretationResult` — a bundle of `ParamDistribution` objects

The `sample()` function draws concrete values from those distributions, producing a `CompleteSongIntentRequest` that can be sent to `/generate`.

### 4. Unified Intent Hook — `src/hooks/useUnifiedIntent.ts`

```typescript
interface UnifiedIntentState {
  activeNodes: Set<string>;              // manually selected taxonomy nodes
  nlpDetectedNodes: Set<string>;         // nodes detected from Side B text
  pinnedParams: Map<string, any>;        // user-pinned (locked) parameters
  interpretiveLevel: number;             // 0-1 slider value
  interpretation: InterpretationResult;  // current probability distributions
  sampledIntent: CompleteSongIntentRequest;  // last sampled concrete values
  activeClusters: ScoredCluster[];       // which genre/style clusters are active
  naturalLanguageText: string;
  isValid: boolean;
  summary: string;                       // "Dorian·85% · ~140-170 BPM · Metal/Prog"
}
```

Key methods:
- `toggleNode(id)` — add/remove from activeNodes, re-interpret
- `applyNLPResult(parseResult)` — update nlpDetectedNodes, re-interpret
- `pinParam(field, value)` / `unpinParam(field)` — lock/unlock specific values
- `setInterpretiveLevel(0-1)` — adjust variance
- `reinterpret()` — re-sample from same distributions (new random seed)
- `generate()` — send `sampledIntent` to `/generate`

---

## Phase 2: Backend — Contextual NLP Parsing

### 5. Text-to-Intent Service — `music_brain/nlp/text_to_intent_service.py`

Wraps existing modules but returns **distributions, not values**:

```python
class TextToIntentService:
    def parse(self, text: str, locale: str = "en") -> InterpretationResult:
        # 1. Parse emotion (existing TextEmotionParser)
        parsed = self.emotion_parser.parse(text)

        # 2. Get musical parameters as ranges/weights (existing get_parameters_for_state)
        #    Already returns mode_weights as distributions + tempo_min/max as ranges
        params = get_parameters_for_state(parsed.to_emotional_state())

        # 3. Extract musical keywords from text for cluster matching
        keywords = self.extract_musical_keywords(text)

        # 4. Match context clusters
        clusters = self.match_clusters(keywords, params)

        # 5. Apply user preference priors if available
        if self.user_prefs:
            params = self.apply_user_priors(params, self.user_prefs)

        # 6. Return distributions + cluster activations + taxonomy node IDs
        return InterpretationResult(
            param_distributions=self.params_to_distributions(params),
            activated_clusters=clusters,
            activated_taxonomy_ids=self.params_to_taxonomy_ids(params, keywords),
            confidence=parsed.confidence,
            detected_keywords=keywords,
        )
```

The existing `MusicalParameters` already uses:
- `mode_weights: Dict[str, float]` — probability distributions over modes
- `tempo_min/tempo_max` — ranges, not fixed values
- `space_probability` — continuous 0-1

So the infrastructure is already probabilistic. We just need to expose that properly instead of collapsing to single values.

### 6. `/parse-text` Endpoint — add to `music_brain/api.py`

```
POST /parse-text
{
  "text": "dorian with fast guitar licks and lots of overdrive",
  "locale": "en",
  "user_id": "default"
}
→ {
  "param_distributions": {
    "mode_weights": { "dorian": 0.85, "phrygian": 0.10, "aeolian": 0.05 },
    "tempo": { "type": "gaussian", "center": 160, "spread": 15 },
    "density": { "type": "range", "min": 0.8, "max": 1.0, "bias": 0.9 },
    "timing_feel": { "type": "discrete", "weights": { "on": 0.6, "ahead": 0.4 } },
    ...
  },
  "activated_clusters": [
    { "id": "metal-shred", "confidence": 0.82, "label": "Metal/Prog Shred" }
  ],
  "activated_taxonomy_ids": ["harmony.scales.dorian", "rhythm.density.dense", ...],
  "detected_keywords": ["dorian", "fast", "guitar", "overdrive"],
  "confidence": 0.82
}
```

### 7. Keyword Extraction — `music_brain/nlp/keyword_extractor.py`

Extracts musical keywords from free text. Not just emotion words (which TextEmotionParser handles) but **instrument names, technique names, genre markers, tempo words, texture words**:

```python
MUSICAL_KEYWORD_GROUPS = {
    "instruments": {"guitar", "bass", "piano", "synth", "drums", "sax", "strings", ...},
    "techniques": {"overdrive", "distortion", "clean", "fingerpicking", "slap", "bowing", ...},
    "tempo_words": {"fast", "slow", "driving", "laid-back", "rushing", "crawling", ...},
    "density_words": {"busy", "sparse", "thick", "thin", "minimal", "dense", "lush", ...},
    "genre_markers": {"bluesy", "jazzy", "metal", "punk", "ambient", "funky", ...},
    "feel_words": {"groovy", "tight", "loose", "swinging", "straight", "mechanical", ...},
}
```

These keywords feed into cluster matching. "Overdrive" alone doesn't set a parameter — but "overdrive + fast + guitar" activates the `metal-shred` cluster which shifts everything.

---

## Phase 3: Frontend Components

### New files in `src/components/UniversalMusicInput/`:

| Component | Purpose |
|-----------|---------|
| `UniversalMusicInput.tsx` | Root dual-pane layout |
| `SideAPane.tsx` | Left pane: tree + search + interpretive slider |
| `TaxonomyTree.tsx` | Collapsible tree with **probability indicators** on each node |
| `CommandPalette.tsx` | Search with fuzzy filtering |
| `TaxonomyNodeChip.tsx` | Shows node label + weight (◐ 0.85) + pin button |
| `SideBPane.tsx` | Right pane: text + context feedback |
| `NaturalLanguageInput.tsx` | Textarea with debounced parsing |
| `ContextFeedback.tsx` | Shows detected cluster ("Metal/Prog"), confidence, detected keywords as tags |
| `InterpretationPanel.tsx` | Shows current distributions visually (mini bar charts for mode_weights, range sliders for tempo) |
| `SuggestionChips.tsx` | "Try also: [add syncopation] [drop the tempo]" |
| `IntentSummaryBar.tsx` | Bottom bar: summary with ranges + Generate + Reinterpret |
| `ToolDrawer.tsx` | Collapsible panel for existing components |
| `InterpretiveSlider.tsx` | Literal ↔ Surprise slider |
| `PinControls.tsx` | Per-parameter pin/unpin toggles |

### `ContextFeedback.tsx` — the key differentiator

This component shows **how the system is interpreting** the combined input:
- Active context cluster with confidence: "Metal/Prog Shred (82%)"
- Each detected keyword highlighted in the text
- Mini distribution visualizations: mode_weights as a horizontal stacked bar, tempo as a range with bell curve
- "Reinterpret" button that re-samples without changing input
- Comparison: "If you change to clean instead of overdrive → Blues/Slow (76%)"

This transparency turns the interpretation from a black box into a conversation.

---

## Phase 4: App.tsx Restructure

Replace the 4-tab router with `<UniversalMusicInput />`. Existing components move into `ToolDrawer`.

---

## Phase 5: User Learning Integration

### Connect to existing `learning/` module

The `PreferenceAnalyzer` and `UserPreferenceModel` already track:
- Parameter adjustments (what users change after generation)
- Emotion selections and preferred ranges
- MIDI acceptance/rejection
- Suggestion interactions

Wire this into the interpretation engine as **user priors**:
- If a user consistently bumps tempo up after "dreamy" generations, their personal "dreamy" distribution shifts higher
- If they always remove the phrygian suggestions, phrygian gets down-weighted for them
- Preferences are loaded from `~/.kelly/user_preferences.json` (existing path)

This happens in the `/parse-text` endpoint when `user_id` is provided.

---

## Phase 6: Bidirectional Sync (Probabilistic)

- **Side A → interpretation**: Toggling a node re-runs `interpret()` with the new active set. Context clusters may activate/deactivate. All distributions update. Side B's ContextFeedback updates to show the new interpretation.
- **Side B → interpretation**: Typing text → `/parse-text` → returns distributions + cluster activations + taxonomy IDs. Side A shows probability weights on each node (not just on/off). ContextFeedback shows the detected cluster.
- **Pinned params**: When a user pins a value (e.g., pins tempo to 120 BPM), that value is locked regardless of interpretation changes. Pinned params show with a 📌 icon on Side A.
- **Reinterpret**: Same distributions, new random sample. Generate button always samples fresh unless user pins everything.
- **Interpretive slider**: At 0 (Literal), distributions collapse to their centers — nearly deterministic. At 1 (Surprise), spreads widen — high variance, genuinely unpredictable.

---

## Phase 7: Styling

Same as before: CSS Grid, `--accent` vars, responsive stacking at `<768px`.

Additional visual elements:
- Probability indicators: partial-fill circles (◐) or small horizontal bars next to node labels
- Distribution mini-charts in ContextFeedback panel
- Interpretive slider styled as a gradient from "precise" to "wild"
- Pinned params get a subtle lock icon and fixed-width display

---

## Build Sequence

| # | Task | Milestone |
|---|------|-----------|
| 1 | `src/data/contextClusters.ts` — define ~30 genre/style clusters | Data: trigger rules + distribution overrides for major genres |
| 2 | `src/data/taxonomyTree.ts` — 10 categories, leaves with distributions + cluster memberships | Data: complete tree, every leaf has paramContributions |
| 3 | `src/data/interpretationEngine.ts` — resolve clusters, interpret, sample | Logic: given nodes + keywords → distributions → concrete intent |
| 4 | `src/hooks/useUnifiedIntent.ts` — state + interpret + pin + reinterpret | State: all interactions update interpretation correctly |
| 5 | `TaxonomyTree.tsx` + `CommandPalette.tsx` + `TaxonomyNodeChip.tsx` | Visual: tree with probability weights, search works |
| 6 | `SideAPane.tsx` + `InterpretiveSlider.tsx` | Visual: left pane with tree + slider |
| 7 | `NaturalLanguageInput.tsx` + `ContextFeedback.tsx` + `InterpretationPanel.tsx` | Visual: right pane with text + interpretation display |
| 8 | `SideBPane.tsx` + `SuggestionChips.tsx` | Visual: right pane complete |
| 9 | `IntentSummaryBar.tsx` + `PinControls.tsx` | Visual: bottom bar with ranges + Generate + Reinterpret |
| 10 | `UniversalMusicInput.tsx` | Visual: full dual-pane layout |
| 11 | Modify `App.tsx` — replace 4 tabs | App loads with new layout |
| 12 | `ToolDrawer.tsx` — re-parent existing components | All old components accessible |
| 13 | Backend: `keyword_extractor.py` + `text_to_intent_service.py` + `/parse-text` | API: text → distributions + clusters |
| 14 | `src/hooks/useTextParse.ts` — wire Side B to backend | End-to-end: text → interpretation → highlighted nodes |
| 15 | Bidirectional probabilistic sync | Both panes reflect interpretation with probabilities |
| 16 | User learning integration — priors from PreferenceAnalyzer | Interpretations personalize over time |
| 17 | Offline fallback — client-side keyword → cluster matching | Works without API |
| 18 | Responsive CSS + distribution visualizations | Polish |

---

## Files Summary

**Create (20 files):**
- `src/data/contextClusters.ts` — genre/style cluster definitions
- `src/data/taxonomyTree.ts` — musical command taxonomy
- `src/data/interpretationEngine.ts` — probabilistic interpretation logic
- `src/hooks/useUnifiedIntent.ts` — unified state management
- `src/hooks/useTextParse.ts` — debounced NLP parsing
- `src/components/UniversalMusicInput/UniversalMusicInput.tsx`
- `src/components/UniversalMusicInput/SideAPane.tsx`
- `src/components/UniversalMusicInput/TaxonomyTree.tsx`
- `src/components/UniversalMusicInput/CommandPalette.tsx`
- `src/components/UniversalMusicInput/TaxonomyNodeChip.tsx`
- `src/components/UniversalMusicInput/SideBPane.tsx`
- `src/components/UniversalMusicInput/NaturalLanguageInput.tsx`
- `src/components/UniversalMusicInput/ContextFeedback.tsx`
- `src/components/UniversalMusicInput/InterpretationPanel.tsx`
- `src/components/UniversalMusicInput/SuggestionChips.tsx`
- `src/components/UniversalMusicInput/IntentSummaryBar.tsx`
- `src/components/UniversalMusicInput/ToolDrawer.tsx`
- `src/components/UniversalMusicInput/InterpretiveSlider.tsx`
- `music_brain/nlp/__init__.py`
- `music_brain/nlp/text_to_intent_service.py`
- `music_brain/nlp/keyword_extractor.py`

**Modify (5 files):**
- `src/App.tsx` — replace 4-tab layout with `<UniversalMusicInput />`
- `src/hooks/useMusicBrain.ts` — add `parseText()` method
- `src/types/Intent.ts` — add distribution types + `ParseTextResponse`
- `src/index.css` — dual-pane layout + probability visualization styles
- `music_brain/api.py` — add `POST /parse-text` endpoint

**Existing files leveraged (not modified):**
- `music_brain/data_utils/emotional_mapping.py` — already has mode_weights as distributions, tempo ranges
- `music_brain/emotion/text_emotion_parser.py` — NLP emotion parsing
- `music_brain/learning/user_preferences.py` — user preference tracking
- `music_brain/learning/preference_analyzer.py` — preference analysis for priors
- `music_brain/kelly_companion/emotion_thesaurus.py` — 6×6×6 emotion hierarchy

---

## Verification

1. **Context sensitivity**: Type "dorian fast guitar overdrive" → cluster shows "Metal/Prog", tempo ~160. Change "overdrive" to "clean" and "fast" to "slow" → cluster shifts to "Blues", tempo ~72. Same mode, different music.
2. **Probability display**: Side A nodes show weight indicators, not binary checkmarks
3. **Reinterpret**: Click Reinterpret with same input → different concrete values sampled from same distributions
4. **Interpretive slider**: At 0 (Literal), reinterpret produces nearly identical results. At 1 (Surprise), results vary widely.
5. **Pin**: Pin tempo to 120 → tempo stays 120 regardless of text changes or reinterpretation
6. **User learning**: After several sessions where user adjusts tempo up, the system's tempo distributions shift higher for similar inputs
7. **Generate**: Click Generate → sends sampled `CompleteSongIntentRequest` to `/generate` → music plays
8. **Tool Drawer**: All existing components accessible
9. **Offline**: Without API, basic keyword matching still activates clusters and highlights nodes
