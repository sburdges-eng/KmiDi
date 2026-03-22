# KmiDi Frontend Architecture

Complete reference for LLMs and developers working with the KmiDi frontend codebase.

## Entry Point

```
src/main.tsx
  └─ <StrictMode>
       └─ <ErrorBoundary>
            └─ <AppConsole />        ← the real entry point (App.tsx is legacy/unmounted)
```

`main.tsx` mounts `AppConsole` into `#root`. The `ErrorBoundary` catches render errors and displays a recovery UI.

---

## Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│ UPPER DECK (160px fixed)                                    │
│ ┌──────────────┬─────────────────────┬────────────────────┐ │
│ │  Transport   │     Timeline        │  VUMeter + Mixer   │ │
│ │  Play/Stop   │  Bar ruler + time   │  Master + channels │ │
│ │  Record      │                     │                    │ │
│ │  Tempo 40-300│                     │                    │ │
│ └──────────────┴─────────────────────┴────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ LOWER DECK (flex 1fr)                                       │
│ ┌──────────────────────┬────────────────────────────────────┐│
│ │     SIDE A            │         SIDE B                    ││
│ │  Musical Commands     │    Natural Language               ││
│ │                       │                                   ││
│ │  [Search... ⌘K]       │  ┌─────────────────────────┐     ││
│ │                       │  │ textarea (user types     │     ││
│ │  ▼ HARMONY      [2]   │  │ music descriptions)     │     ││
│ │    ▼ Scales & Modes   │  └─────────────────────────┘     ││
│ │      ◐ Dorian [85%]   │                                   ││
│ │  ▼ RHYTHM        [1]   │  [Metal/Prog Shred] 100%         ││
│ │  ► DYNAMICS            │  [dorian] [guitar] [overdrive]   ││
│ │  ► ARTICULATION        │                                   ││
│ │  ► TIMBRE        [2]   │  Mode: ████ dorian 85% ██ 15%   ││
│ │  ► EXPRESSION          │  Tempo: ~145–175 BPM             ││
│ │  ► STRUCTURE           │  Confidence: 92%  [🎲 Reinterp] ││
│ │  ► PRODUCTION          │                                   ││
│ │                       │                                   ││
│ │  Literal ───●─── Surprise │                               ││
│ ├──────────────────────┴────────────────────────────────────┤│
│ │ Metal/Prog Shred · dorian 85% · ~145-175 BPM  [🎲] [▶ Gen] ● Online ││
│ ├───────────────────────────────────────────────────────────┤│
│ │ [▲ Tools]  (collapsible drawer with existing components)  ││
│ └───────────────────────────────────────────────────────────┘│
│ Session: "What are you making?"                    ● Online  │
└─────────────────────────────────────────────────────────────┘
```

**CSS files:**
- `src/index.css` — Design system tokens + UMI component styles
- `src/console-shell.css` — Shell layout (upper/lower deck grids)

---

## File Map

### Core Files

| File | Purpose |
|------|---------|
| `src/main.tsx` | React mount, ErrorBoundary wrapping |
| `src/AppConsole.tsx` | Main shell: upper deck + lower deck + tool drawer wiring |
| `src/ErrorBoundary.tsx` | Catches render errors, displays recovery UI |

### Type Definitions

| File | Exports |
|------|---------|
| `src/types/Intent.ts` | `CompleteSongIntentRequest`, `StructureSection`, `TrackIntent` — the final output sent to `/generate` |
| `src/types/Interpretation.ts` | `ParamDistribution`, `ParseTextResponse` — probabilistic interpretation types |

### Hooks

| File | Exports | Purpose |
|------|---------|---------|
| `src/hooks/useMusicBrain.ts` | `useMusicBrain()`, `buildGeneratePayload()` | API client for all backend endpoints |
| `src/hooks/useTextParse.ts` | `useTextParse()` | Debounced NLP text parsing with offline fallback |

### Data Layer

| File | Exports | Purpose |
|------|---------|---------|
| `src/data/taxonomyTree.ts` | `TAXONOMY_TREE`, `TaxonomyNode`, `flattenTaxonomy()` | 200+ node tree of every musical command |
| `src/data/contextClusters.ts` | `CONTEXT_CLUSTERS`, `ContextCluster`, `TriggerRule` | 35 genre/style clusters with trigger conditions |
| `src/data/interpretationEngine.ts` | `interpret()`, `sample()`, `resolveContextClusters()`, `mergeDistributions()` | Probabilistic interpretation logic |

### Universal Music Input Components

| File | Component | Props |
|------|-----------|-------|
| `UniversalMusicInput.tsx` | `UniversalMusicInput` | `apiStatus`, `toolDrawerContent?` |
| `SideAPane.tsx` | `SideAPane` | `taxonomyNodes`, `nodeWeights`, `nlpDetectedNodes`, `pinnedNodes`, `searchQuery`, `interpretiveLevel`, + callbacks |
| `SideBPane.tsx` | `SideBPane` | `text`, `isParsing`, `parseResult`, `onTextChange`, `onReinterpret` |
| `TaxonomyTree.tsx` | `TaxonomyTree` | `nodes`, `nodeWeights`, `nlpDetectedNodes`, `pinnedNodes`, `searchQuery`, + callbacks |
| `TaxonomyNodeChip.tsx` | `TaxonomyNodeChip` | `id`, `label`, `weight?`, `isNlpDetected?`, `isPinned?`, `onToggle`, `onPin?` |
| `CommandPalette.tsx` | `CommandPalette` | `searchQuery`, `onSearchChange`, `activeCount` |
| `NaturalLanguageInput.tsx` | `NaturalLanguageInput` | `value`, `isParsing`, `onChange` |
| `ContextFeedback.tsx` | `ContextFeedback` | `parseResult`, `onReinterpret` |
| `IntentSummaryBar.tsx` | `IntentSummaryBar` | `summary`, `isValid`, `isGenerating`, `apiStatus`, `onGenerate`, `onReinterpret` |
| `InterpretiveSlider.tsx` | `InterpretiveSlider` | `value` (0-1), `onChange` |
| `ToolDrawer.tsx` | `ToolDrawer` | `children` |

### Existing Components (in upper deck or tool drawer)

| File | Component | Props |
|------|-----------|-------|
| `SideA/Transport.tsx` | `Transport` | `isPlaying`, `isRecording`, `tempo`, `onPlayPause`, `onStop`, `onRecord`, `onTempoChange` |
| `SideA/Mixer.tsx` | `Mixer` | `channels[]`, `onChannelChange(id, patch)` |
| `SideA/Timeline.tsx` | `Timeline` | `bars`, `tempo` |
| `SideA/VUMeter.tsx` | `VUMeter` | `value` (0-1), `isActive` |
| `SideA/CompactMeter.tsx` | `CompactMeter` | `label`, `value`, `minDb`, `maxDb`, `clip?`, `disabled`, `valueIsLinear` |
| `ui/Knob.tsx` | `Knob` | `value`, `min`, `max`, `step`, `stepFine`, `onChange`, `label?` |
| `SideB/EmotionWheel.tsx` | `EmotionWheel` | `selected`, `onSelect` |
| `SideB/GhostWriter.tsx` | `GhostWriter` | `seed`, `output`, `onGenerate` |
| `SideB/Interrogator.tsx` | `Interrogator` | `starter`, `onAsk` |
| `LyricPanel.tsx` | `LyricPanel` | (none — uses useMusicBrain internally) |
| `SpectoCloudPanel.tsx` | `SpectoCloudPanel` | `lastGeneratedAudioPath?` |

---

## Key Types

### CompleteSongIntentRequest (the final output)

```typescript
interface CompleteSongIntentRequest {
  core_desire: string;           // narrative/emotional intention (1-1000 chars)
  mood_primary: string;          // primary mood (1-100 chars)
  genre: string;                 // musical genre (1-100 chars)
  tempo?: number;                // BPM, clamped 40-300
  key_mode: string;              // e.g., "A dorian", "C major"
  structure: StructureSection[]; // [{name, bars, repetitions}]
  instruments: TrackIntent[];    // [{instrument, techniques[]}]
  groove_feel?: string;          // "Straight/Driving", "Laid Back", "Swung", etc.
  narrative_arc?: string;        // "Climb-to-Climax", "Slow Reveal", etc.
  rule_to_break?: string | null; // intentional theory violation
  rule_justification?: string | null;
}
```

### ParamDistribution (probabilistic values)

```typescript
interface ParamDistribution {
  type: 'gaussian' | 'discrete' | 'range';
  center?: number;               // gaussian: mean
  spread?: number;               // gaussian: std deviation
  weights?: Record<string, number>; // discrete: {option: probability}
  min?: number;                  // range: lower bound
  max?: number;                  // range: upper bound
  bias?: number;                 // range: preferred point within range
}
```

### ParseTextResponse (from POST /parse-text)

```typescript
interface ParseTextResponse {
  param_distributions: Record<string, ParamDistribution>;
  activated_clusters: Array<{id: string; confidence: number; label: string}>;
  activated_taxonomy_ids: string[];
  detected_keywords: string[];
  confidence: number;  // 0-1
}
```

### TaxonomyNode (tree node)

```typescript
interface TaxonomyNode {
  id: string;                    // dot-notation: "harmony.scales.dorian"
  label: string;                 // display: "Dorian"
  labelKey?: string;             // i18n key for future multi-language
  children?: TaxonomyNode[];
  paramContributions?: Record<string, ParamDistribution>;
  clusterMemberships?: string[]; // which context clusters this participates in
  keywords: string[];            // for search + NLP matching
  description?: string;
}
```

---

## Data Flow

### NLP Pipeline (text → music generation)

```
User types text in Side B textarea
        │
        ▼
handleTextChange(text)
  ├─ setNaturalText(text)
  └─ parseDebounced(text)  ──── 300ms debounce ────┐
                                                     │
                                                     ▼
                                          useTextParse hook
                                            │
                                            ▼
                                   brain.parseText(text, locale)
                                            │
                                            ▼
                                   POST /parse-text (backend)
                                   TextToIntentService.parse()
                                     ├─ TextEmotionParser → emotion
                                     ├─ KeywordExtractor → musical terms
                                     ├─ Cluster matching → genre/style
                                     └─ Distribution blending
                                            │
                                            ▼
                                   ParseTextResponse
                                     {param_distributions,
                                      activated_clusters,
                                      activated_taxonomy_ids,
                                      detected_keywords,
                                      confidence}
                                            │
                              ┌─────────────┴──────────────┐
                              ▼                            ▼
                    Side A: TaxonomyTree          Side B: ContextFeedback
                    highlights nodes with         shows cluster badge,
                    probability weights           keywords, mode bar,
                    (dashed border =               tempo range,
                     NLP-detected)                confidence %
                              │                            │
                              └─────────────┬──────────────┘
                                            ▼
                                    IntentSummaryBar
                                    "Metal/Prog Shred · dorian 85% · ~145-175 BPM"
                                            │
                                            ▼ (user clicks Generate)
                                    handleGenerate()
                                     ├─ buildKeyModeFromWeights() → "C dorian"
                                     ├─ buildInstrumentsFromNodes() → [{instrument, techniques}]
                                     ├─ getGrooveFeel() → "Straight/Driving"
                                     ├─ getNarrativeArc() → "Climb-to-Climax"
                                     └─ brain.generateMusic(payload)
                                            │
                                            ▼
                                    POST /generate (backend)
```

### Bidirectional Sync

```
SIDE A (Manual)                          SIDE B (NLP)
━━━━━━━━━━━━━━━━                        ━━━━━━━━━━━━━━━
User clicks node                         User types text
       │                                        │
       ▼                                        ▼
toggleNode(id)                           parseDebounced(text)
       │                                        │
       ▼                                        ▼
activeNodes.add(id)                      parseResult.activated_taxonomy_ids
       │                                        │
       └────────────────┬───────────────────────┘
                        │
                        ▼
              nodeWeights: Map<string, number>
              ┌─ manual nodes → weight 1.0
              └─ NLP nodes → weight (confidence × 0.85)

              Conflict: manual always wins (NLP won't override)
              Visual: manual = solid border, NLP = dashed border
```

### Interpretive Slider Effect

```
interpretiveLevel = 0.0 (Literal)
  └─ Distributions collapse to centers
  └─ Nearly deterministic output
  └─ "Reinterpret" produces same result

interpretiveLevel = 1.0 (Surprise)
  └─ Full distribution spreads
  └─ High variance sampling
  └─ "Reinterpret" produces very different results
```

---

## State Management

### AppConsole State

| State | Type | Where Used |
|-------|------|------------|
| `isPlaying` | boolean | Transport, VUMeter animation loop |
| `isRecording` | boolean | Transport |
| `tempo` | number (40-300) | Transport, Timeline |
| `masterVu` | number (0-1) | VUMeter (animated when playing) |
| `channels` | Channel[] | Mixer (animated when playing) |
| `selectedEmotion` | {base, intensity, detail} | EmotionWheel → GhostWriter |
| `ghostText` | string | GhostWriter output |
| `apiStatus` | 'checking' \| 'online' \| 'offline' | UniversalMusicInput, SessionBar |
| `apiError` | string \| null | Error display bar |

### UniversalMusicInput State

| State | Type | Purpose |
|-------|------|---------|
| `searchQuery` | string | Filters taxonomy tree |
| `naturalText` | string | User's free-text input |
| `activeNodes` | Set\<string\> | Manually selected taxonomy nodes |
| `pinnedNodes` | Set\<string\> | Locked parameter values |
| `interpretiveLevel` | number (0-1) | Controls sampling variance |
| `isGenerating` | boolean | Disables generate button |

### Derived State (computed every render)

| Derived | Source | Computation |
|---------|--------|-------------|
| `nlpDetectedNodes` | `parseResult` | `new Set(parseResult.activated_taxonomy_ids)` |
| `nodeWeights` | `activeNodes` + `parseResult` | Manual → 1.0, NLP → conf × 0.85 |
| `summary` | `parseResult` + `activeNodes` | Top cluster + top mode + tempo range |
| `isValid` | `activeNodes` + `parseResult` | Has selections OR has detected keywords |

---

## Context Clusters (Genre/Style Detection)

The interpretation system doesn't map parameters independently. Instead, it recognizes **context clusters** — combinations of parameters that imply a genre/style. The same parameter means different things depending on context.

### Example: Dorian Mode in Different Contexts

| Active Nodes / Keywords | Detected Cluster | Tempo | Feel |
|------------------------|------------------|-------|------|
| dorian + fast + guitar + overdrive | Metal/Prog Shred | ~160 BPM | Driving, dense |
| dorian + slow + bass + clean + bluesy | Slow Blues | ~72 BPM | Behind-beat, sparse |
| dorian + swing + sax + walking bass | Modal Jazz | ~130 BPM | Swung, medium |
| dorian + syncopated + bass + funky | Funk Groove | ~105 BPM | Ahead, groovy |
| dorian + slow + piano + lo-fi | Lo-fi Chill | ~78 BPM | Behind-beat, sparse |

### Cluster Activation

A cluster activates when its `triggerConditions` are met (matching nodes + keywords ≥ `minMatchCount`). Score = matchCount / (minMatchCount + 1), capped at 1.0. Top 5 clusters kept.

### Distribution Merging

When multiple clusters activate, their `parameterOverrides` are blended:
- **Gaussian**: weighted average of centers, max of spreads
- **Discrete**: weighted average of option probabilities, then normalize
- **Range**: weighted average of min/max/bias

---

## CSS Design System

### Variables (from `:root` in index.css)

```css
/* Typography */
--font-display: 'Fraunces', Georgia, serif;
--font-ui: 'IBM Plex Sans', system-ui, sans-serif;
--font-mono: ui-monospace, SFMono-Regular, Menlo, monospace;

/* Surfaces (dark → light) */
--surface-inset: #0a0a0c;    /* deepest: inputs, textareas */
--surface: #0f0f12;           /* panels */
--surface-raised: #16161a;    /* cards, buttons */
--surface-overlay: #1e1e23;   /* hover states */

/* Accent: warm amber */
--accent: #c9a227;
--accent-hover: #d4b03d;
--accent-muted: rgba(201, 162, 39, 0.18);  /* node fill bars, badges */
--accent-focus: rgba(201, 162, 39, 0.35);

/* Text */
--text-primary: #f2f2f1;
--text-secondary: #9c9c9a;
--text-muted: #5e5e5c;

/* Spacing (4px rhythm) */
--space-1: 4px;  --space-2: 8px;  --space-3: 12px;
--space-4: 16px; --space-5: 24px;
--radius: 6px;   --radius-lg: 8px;
```

### UMI CSS Classes

| Class | Purpose |
|-------|---------|
| `.umi-container` | CSS Grid: `1fr 1fr` columns, `1fr auto` rows |
| `.umi-side-a`, `.umi-side-b` | Panel with border, flex column, overflow hidden |
| `.umi-tree-scroll` | Scrollable area for taxonomy tree |
| `.umi-tree-header` | Category header with expand arrow + badge |
| `.umi-tree-children` | Indented child container |
| `.umi-node-chip` | Leaf node button with fill bar animation |
| `.umi-node-chip.active` | Amber border, primary text color |
| `.umi-node-chip.nlp-detected` | Dashed amber border (NLP-detected, not manual) |
| `.umi-node-chip-fill` | Absolute-positioned background fill (width = weight%) |
| `.umi-node-chip-weight` | Monospace weight percentage label |
| `.umi-node-chip-pin` | Pin icon (📌/📍) |
| `.umi-search-box` | Search input container with icon |
| `.umi-slider-input` | Range input styled as gradient bar |
| `.umi-nl-textarea` | Large textarea for natural language input |
| `.umi-parsing-indicator` | Animated dot + "interpreting..." |
| `.umi-cluster-badge` | Genre/style label with confidence |
| `.umi-keyword-tag` | Small monospace tag for detected keywords |
| `.umi-dist-segments` | Flex container for mode distribution bar |
| `.umi-summary-bar` | Bottom bar spanning both columns |
| `.umi-generate-btn` | Amber CTA button |
| `.umi-api-status` | Online/offline dot indicator |
| `.umi-drawer` | Collapsible panel for existing components |

### Responsive

```css
@media (max-width: 768px) {
  .umi-container { grid-template-columns: 1fr; }  /* stack vertically */
}
```

---

## API Endpoints Used by Frontend

| Endpoint | Method | Used By | Gate |
|----------|--------|---------|------|
| `/health` | GET | `useMusicBrain.healthCheck()` | `USE_EXTERNAL_API` |
| `/parse-text` | POST | `useMusicBrain.parseText()` | **No gate** (always calls local API) |
| `/generate` | POST | `useMusicBrain.generateMusic()` | `USE_EXTERNAL_API` |
| `/interrogate` | POST | `useMusicBrain.interrogate()` | `USE_EXTERNAL_API` |
| `/lyrics` | GET/POST | `useMusicBrain.getUserLyrics/setUserLyrics()` | `USE_EXTERNAL_API` |
| `/emotions` | GET | `useMusicBrain.getEmotions()` | `USE_EXTERNAL_API` |
| `/spectocloud/render` | POST | `useMusicBrain.renderSpectocloud()` | `USE_EXTERNAL_API` |
| `/audio/classify` | POST | `useMusicBrain.classifyAudio()` | `USE_EXTERNAL_API` |

`USE_EXTERNAL_API` is controlled by `VITE_KMIDI_USE_API=true` env var. Default is `false` (all gated endpoints reject). The `/parse-text` endpoint bypasses this gate because it's a local development tool.

---

## Offline Behavior

When the Python API is unavailable:

1. `healthCheck()` fails → `apiStatus = 'offline'`
2. `parseText()` call fails → `useTextParse` catches error and runs `offlineFallback(text)`
3. Offline fallback does client-side keyword matching against a hardcoded `KEYWORD_TO_TAXONOMY` map (~30 common terms)
4. Returns `ParseTextResponse` with `activated_taxonomy_ids` and `detected_keywords` but no `activated_clusters` or `param_distributions`
5. Side A still highlights detected nodes (from keywords), but no cluster detection or distribution visualization
6. Generate button is disabled (`apiStatus !== 'online'`)

---

## Taxonomy Tree Structure (10 Categories)

```
harmony/
  ├─ keys/           (C, C#, Db, D, ... B)
  ├─ scales/         (major, minor, dorian, phrygian, lydian, mixolydian, locrian,
  │                   harmonic-minor, melodic-minor, pentatonic-major, pentatonic-minor,
  │                   blues, whole-tone, chromatic, diminished)
  ├─ chords/         (major-triad, minor-triad, dim, aug, maj7, min7, dom7, sus2, sus4,
  │                   power-chord, dom9, dom7alt)
  ├─ progressions/   (I-V-vi-IV, ii-V-I, twelve-bar-blues, i-VII-VI-VII)
  └─ techniques/     (modal-interchange, tritone-sub, pedal-point, voice-leading)

melody/
  ├─ contour/        (ascending, descending, arch, wave, static)
  ├─ activity/       (sparse, moderate, active, virtuosic)
  └─ ornamentation/  (trill, grace-note, bend, slide)

rhythm/
  ├─ time-signatures/ (4-4, 3-4, 6-8, 5-4, 7-8)
  ├─ patterns/       (four-on-floor, backbeat, shuffle, clave, breakbeat, boom-bap)
  ├─ syncopation/    (none, light, heavy, offbeat)
  ├─ density/        (sparse, medium, dense)
  ├─ swing/          (straight, light-swing, heavy-swing)
  ├─ groove-feel/    (straight-driving, laid-back, swung, syncopated, rubato,
  │                   mechanical, organic, push-pull)
  └─ polyrhythm/     (2-against-3, 3-against-4)

dynamics/
  ├─ levels/         (pp, p, mp, mf, f, ff)
  └─ changes/        (crescendo, decrescendo, sforzando)

articulation/       (staccato, legato, vibrato, portamento, tremolo, pizzicato,
                     muted, palm-mute, fingerpicking, strumming, slap, tapping, harmonics)

texture/
  ├─ density/        (monophonic, thin, moderate, thick, massive)
  └─ space/          (intimate, room, hall, cathedral, infinite)

structure/
  ├─ sections/       (intro, verse, pre-chorus, chorus, bridge, outro, build, drop,
  │                   breakdown, solo)
  └─ narrative-arc/  (climb-to-climax, slow-reveal, rise-and-fall, sudden-shift,
                      descent, spiral)

timbre/
  ├─ instruments/    (piano, electric-piano, organ, acoustic-guitar, electric-guitar,
  │                   bass-guitar, upright-bass, synth-lead, synth-pad, synth-bass,
  │                   strings, brass, sax, drums, choir, 808, violin, cello, flute, harp)
  ├─ color/          (bright, warm, dark, gritty)
  └─ effects/        (distortion, overdrive, clean, reverb, delay, chorus-effect,
                      wah, compression, bitcrusher, lo-fi)

expression/
  └─ mood/           (nostalgic, melancholic, grief, tender, romantic, hopeful,
                      joyful, energetic, fierce, mysterious, ethereal, calm)

production/
  ├─ register/       (sub-bass, bass, mid, high)
  ├─ spatial/        (mono, narrow-stereo, wide-stereo)
  └─ mix-style/      (raw, polished, lo-fi, saturated)
```

---

## Context Clusters (35 defined)

| ID | Label | Key Triggers |
|----|-------|-------------|
| `metal-shred` | Metal/Prog Shred | fast + guitar + overdrive/distortion |
| `blues-slow` | Slow Blues | slow + blues + clean + pentatonic |
| `blues-chicago` | Chicago Blues | blues + shuffle + overdrive |
| `jazz-modal` | Modal Jazz | jazz + modal + swing + sax/piano |
| `jazz-bebop` | Bebop | fast + jazz + sax/trumpet |
| `jazz-ballad` | Jazz Ballad | slow + jazz + piano + gentle |
| `ambient-pad` | Ambient Pad | ambient + pad + ethereal + reverb |
| `ambient-drone` | Ambient Drone | drone + minimal + ambient |
| `funk-groove` | Funk Groove | funk + syncopated + bass |
| `pop-upbeat` | Upbeat Pop | pop + upbeat + bright |
| `pop-ballad` | Pop Ballad | ballad + slow + piano |
| `edm-drop` | EDM Drop | edm + drop + build + synth |
| `edm-house` | House | house + four-on-floor |
| `edm-dnb` | Drum & Bass | dnb + breakbeat + fast |
| `hip-hop-boom-bap` | Boom Bap | hip-hop + boom bap |
| `hip-hop-trap` | Trap | trap + 808 |
| `lo-fi-chill` | Lo-fi Chill | lo-fi + chill + vinyl + tape |
| `classical-romantic` | Romantic Classical | classical + strings + legato |
| `punk-rock` | Punk Rock | punk + fast + distortion |
| `grunge` | Grunge | grunge + distortion + quiet-loud |
| `indie-folk` | Indie Folk | indie + folk + acoustic + fingerpicking |
| `country-traditional` | Country | country + twang + fiddle |
| `reggae` | Reggae | reggae + offbeat + dub |
| `bossa-nova` | Bossa Nova | bossa + brazilian + nylon |
| `r-and-b-slow` | Slow R&B | r&b + smooth + slow jam |
| `prog-rock` | Progressive Rock | prog + complex + odd meter |
| `post-rock` | Post-Rock | atmospheric + reverb + crescendo |
| `synthwave-retro` | Synthwave/Retrowave | synthwave + 80s + arpeggiated |
| `latin-salsa` | Latin Salsa | salsa + clave + brass |
| `gospel` | Gospel | gospel + choir + organ |
| `cinematic-epic` | Cinematic Epic | cinematic + orchestral + dramatic |
| `neo-soul` | Neo-Soul | neo-soul + warm + groove |
| `world-afrobeat` | Afrobeat | afrobeat + polyrhythm + brass |
| `video-game-chiptune` | Chiptune/8-bit | chiptune + 8-bit + retro |

Each cluster defines `parameterOverrides` as `Record<string, ParamDistribution>` that shift tempo, density, dynamics, timing, groove, swing, and other parameters when activated.
