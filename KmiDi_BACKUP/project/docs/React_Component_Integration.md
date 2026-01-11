# React Component Integration

## Overview

This document describes the React component architecture found in the Google Drive "GOOGLE KELLY INFO" directory and how it integrates with the KmiDi project's Python backend.

## Source Location

**Google Drive Path:** `/Users/seanburdges/Library/CloudStorage/GoogleDrive-sburdges@gmail.com/My Drive/GOOGLE KELLY INFO/TEST UPLOADS/`

## Architecture Philosophy

The unified app architecture combines "Side A" (DAW/Work State) and "Side B" (Emotion/Dream State) into a single cohesive workspace. The philosophy is **"Interrogate Before Generate"** - emotion drives technical choices.

## Component Structure

### Main Application

**File:** `App.unified.tsx`

The main unified application component that integrates both DAW and emotion interfaces.

#### Key Features

- **Unified Layout**: Single workspace combining emotion selection and DAW functionality
- **Panel Management**: Collapsible emotion panel (left) and mixer panel (right)
- **Keyboard Shortcuts**:
  - `⌘E` / `Ctrl+E`: Toggle emotion panel
  - `⌘M` / `Ctrl+M`: Toggle mixer panel
  - `Space`: Play/pause
- **Emotion State Indicator**: Always visible header showing current emotion
- **Real-time Playback**: Timer-based playback with animation frames

#### Component Hierarchy

```
UnifiedApp
├── Header (emotion state, panel toggles)
├── Main Content
│   ├── Emotion Panel (Side B)
│   │   ├── EmotionWheel
│   │   ├── Interrogator
│   │   ├── RuleBreaker
│   │   └── GhostWriter
│   ├── Timeline & Transport (Side A)
│   │   ├── Timeline
│   │   ├── Transport
│   │   └── UnifiedBridge
│   └── Mixer Panel (Side A)
│       ├── Mixer
│       └── EmotionToMixerBridge
└── Footer (tempo, tracks, Logic Pro export)
```

### State Management

**File:** `useUnifiedStore.ts`

Uses Zustand for state management with `subscribeWithSelector` middleware.

#### State Structure

```typescript
interface UnifiedStoreState {
  // Side A (DAW)
  isPlaying: boolean;
  tempo: number;
  tracks: Track[];

  // Side B (Emotion)
  selectedEmotion: string | null;
  completedIntent: Record<string, unknown> | null;
  songIntent: SongIntent;

  // UI Layout
  emotionPanelState: 'collapsed' | 'minimal' | 'expanded';
  mixerPanelState: 'collapsed' | 'minimal' | 'expanded';

  // Bridge Actions
  applyEmotionToMixer: () => void;
  applyGeneratedHarmony: () => void;
}
```

#### Key Types

- **Track**: Audio/MIDI track with emotion-driven parameters
- **SongIntent**: Three-phase intent schema (Phase 0, 1, 2)
- **GeneratedMusic**: Output from Music Brain processing
- **Clip**: Timeline clip with emotion metadata

### Bridge Components

#### UnifiedBridge

**File:** `UnifiedBridge.tsx`

Invisible component that connects emotion state to DAW parameters. Processes intent changes and applies emotion to mixer.

**Key Functions:**
- Processes intent when it changes
- Applies emotion to mixer automatically
- Calls `useMusicBrain` hook to generate music
- Logs bridge actions for debugging

#### EmotionToMixerBridge

**File:** `EmotionToMixerBridge.tsx`

Visual feedback component showing how emotional choices affect mixer parameters.

**Features:**
- Displays emotion-to-mixer mappings
- Shows reverb, delay, compression, warmth settings
- Visual indicators for parameter changes
- Real-time updates based on emotion selection

**Emotion Mappings:**

| Emotion | Reverb | Delay | Compression | Warmth |
|---------|--------|-------|-------------|--------|
| Grief | High (70%) | Medium (40%) | Low (30%) | Low (20%) |
| Joy | Low (30%) | Low (20%) | Medium (50%) | High (70%) |
| Rage | Low (20%) | Minimal (10%) | Very High (90%) | Low (10%) |
| Anxiety | Medium (45%) | High (70%) | Medium (50%) | Low (15%) |
| Longing | High (65%) | Medium (50%) | Low (25%) | Medium (35%) |

### Side A Components (DAW)

#### Timeline

Displays audio/MIDI clips on tracks. Supports:
- Multiple tracks
- Clip editing
- Time-based navigation
- Emotion-tagged clips

#### Mixer

Audio mixing interface with:
- Track volume/pan controls
- Emotion-driven parameter overrides
- Reverb, delay, compression, warmth
- Mute/solo/arm per track

#### Transport

Playback controls:
- Play/pause/stop
- Tempo display
- Time signature
- Current time position

### Side B Components (Emotion)

#### EmotionWheel

Visual emotion selection interface. Maps to the 17 mood options from the intent schema.

#### Interrogator

Three-phase interrogation interface:
- **Phase 0**: Core Wound/Desire
- **Phase 1**: Emotional Intent
- **Phase 2**: Technical Constraints

#### RuleBreaker

Interface for selecting and justifying rule-breaking choices. Maps to `RULE_BREAKING_EFFECTS` from the Python schema.

#### GhostWriter

AI-assisted music generation component. Uses completed intent to generate musical elements.

### Logic Pro Export

**File:** `LogicProExport.tsx`

Exports project data to Logic Pro format with:
- Track information
- MIDI data
- Emotion metadata
- Intent schema data

## Integration with Python Backend

### API Endpoints

The React components connect to the Python backend through:

1. **Music Brain API**: Processes song intent and generates music
   - Endpoint: `/api/music_brain/process_intent`
   - Input: `CompleteSongIntent` (JSON)
   - Output: `GeneratedMusic` (harmony, tempo, key, mixer params)

2. **Instrument Selector API**: Gets samples for emotion/instrument pairs
   - Endpoint: `/api/instruments/samples`
   - Input: `{ emotion: string, instrument: string }`
   - Output: `{ samples: string[] }`

### Type Synchronization

The TypeScript types should match the Python schema:

```typescript
// TypeScript (useUnifiedStore.ts)
interface SongIntent {
  coreEmotion: string | null;
  coreEvent?: string;
  coreResistance?: string;
  coreLonging?: string;
  vulnerabilityScale?: 'Low' | 'Medium' | 'High';
  technicalKey?: string;
  ruleBreakJustification?: string;
}

// Python (intent_schema.py)
@dataclass
class CompleteSongIntent:
    song_root: SongRoot  # core_event, core_resistance, core_longing
    song_intent: SongIntent  # mood_primary, vulnerability_scale
    technical_constraints: TechnicalConstraints  # technical_key, rule_breaking_justification
```

### Data Flow

```
User selects emotion (EmotionWheel)
  ↓
User completes interrogation (Interrogator)
  ↓
Intent sent to UnifiedBridge
  ↓
UnifiedBridge calls useMusicBrain hook
  ↓
Hook sends POST to /api/music_brain/process_intent
  ↓
Python backend processes intent
  ↓
Returns GeneratedMusic
  ↓
UnifiedBridge updates store
  ↓
Timeline/Mixer update with generated music
```

## Dependencies

### Required Packages

```json
{
  "react": "^18.0.0",
  "zustand": "^4.0.0",
  "framer-motion": "^10.0.0",
  "lucide-react": "^0.200.0"
}
```

### Optional Packages

- `@tanstack/react-query`: For API calls (if not using custom hooks)
- `axios` or `fetch`: For HTTP requests
- `midi-writer-js`: For MIDI file generation

## Integration Strategy

### Option 1: Reference Architecture

Use the Google Drive components as a reference for implementing similar functionality in the project's `web/` directory. This preserves the architecture while adapting to project-specific needs.

### Option 2: Direct Integration

Copy components to the project and adapt:
1. Copy components to `web/components/` or `src/components/`
2. Update import paths
3. Create matching API endpoints in Python backend
4. Implement `useMusicBrain` hook
5. Update types to match Python schema

### Option 3: Hybrid Approach

Extract key patterns and types:
1. Extract TypeScript types to `web/types/unified.ts`
2. Create simplified bridge components
3. Implement core functionality incrementally

## Recommended Approach

Given the project's current state (minimal `web/` directory), **Option 1 (Reference Architecture)** is recommended:

1. **Document the architecture** (this document)
2. **Extract TypeScript types** for future use
3. **Create API endpoints** in Python backend
4. **Implement components incrementally** as needed

## Type Definitions

### Complete TypeScript Types

See `web/types/unified.ts` (to be created) for complete type definitions matching the Python schema.

Key types to extract:
- `SongIntent` (matches `CompleteSongIntent`)
- `Track` (with emotion overrides)
- `GeneratedMusic`
- `EmotionToMixerMapping`

## Next Steps

1. **Create API endpoints** in Python backend (`api/main.py`)
2. **Extract TypeScript types** from Google Drive components
3. **Implement `useMusicBrain` hook** for API calls
4. **Create simplified bridge components** for initial integration
5. **Test integration** with Python backend

## Related Documentation

- [Song Intent Schema](../music_brain/data/song_intent_schema.yaml)
- [Intent Schema Python Module](../music_brain/session/intent_schema.py)
- [Emotion Instrument Library](./Emotion_Instrument_Library.md)
- [Google Drive Integration Summary](./GOOGLE_DRIVE_INTEGRATION.md)
