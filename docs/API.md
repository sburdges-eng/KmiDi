# KmiDi API Reference

**Version:** 1.0  
**Updated:** 2026-01-18  
**Purpose:** Complete API reference for all KmiDi interfaces

## Overview

KmiDi exposes three main API layers:

1. **C FFI API** - Low-level C interface for KellyBrain integration
2. **Tauri Commands** - High-level Rust commands for React frontend
3. **React Hooks** - TypeScript interfaces for React components

## C FFI API Reference

### Core Functions

#### kelly_brain_create()
```c
KellyBrain* kelly_brain_create(void);
```
**Purpose:** Create new KellyBrain instance  
**Returns:** Pointer to KellyBrain instance, or NULL on failure  
**Memory:** Caller must call `kelly_brain_destroy()` to free  

#### kelly_brain_initialize()
```c
KellyErrorCode kelly_brain_initialize(KellyBrain* brain, const char* data_path);
```
**Purpose:** Initialize KellyBrain with emotion data  
**Parameters:**
- `brain` - KellyBrain instance  
- `data_path` - Path to data directory  
**Returns:** `KELLY_SUCCESS` on success, error code on failure  
**Example:**
```c
KellyBrain* brain = kelly_brain_create();
KellyErrorCode result = kelly_brain_initialize(brain, "./data");
if (result == KELLY_SUCCESS) {
    // Ready to use
}
```

#### kelly_brain_from_text()
```c
char* kelly_brain_from_text(KellyBrain* brain, const char* text);
```
**Purpose:** Generate intent from text description  
**Parameters:**
- `brain` - Initialized KellyBrain instance
- `text` - Text description of emotional state  
**Returns:** JSON string containing IntentResult, or NULL on error  
**Memory:** Caller must call `kelly_free_string()` on result  
**Example:**
```c
char* intent_json = kelly_brain_from_text(brain, "I feel lost and confused");
if (intent_json) {
    // Process JSON result
    kelly_free_string(intent_json);
}
```

### MIDI Generation Functions

#### kelly_brain_generate_midi()
```c
char* kelly_brain_generate_midi(KellyBrain* brain, const char* intent_json, int bars);
```
**Purpose:** Generate MIDI from intent result  
**Parameters:**
- `brain` - KellyBrain instance
- `intent_json` - JSON IntentResult from previous call
- `bars` - Number of bars to generate (1-64)  
**Returns:** JSON string containing GeneratedMidi, or NULL on error  
**Memory:** Caller must call `kelly_free_string()` on result

#### kelly_brain_generate_midi_with_params()
```c
char* kelly_brain_generate_midi_with_params(
    KellyBrain* brain, 
    const char* intent_json, 
    int bars, 
    int bpm, 
    const char* key_signature
);
```
**Purpose:** Generate MIDI with specific musical parameters  
**Parameters:**
- `brain` - KellyBrain instance
- `intent_json` - JSON IntentResult
- `bars` - Number of bars (1-64)
- `bpm` - Beats per minute (1-300)
- `key_signature` - Key signature (e.g., "C", "Dm", "F#")  
**Returns:** JSON string containing GeneratedMidi

### State Query Functions

#### kelly_brain_get_emotion_state()
```c
char* kelly_brain_get_emotion_state(const KellyBrain* brain);
```
**Purpose:** Get current emotional state  
**Returns:** JSON with valence, arousal, dominance, complexity values  
**Example JSON:**
```json
{
  "valence": 0.2,
  "arousal": 0.7,
  "dominance": 0.5,
  "complexity": 0.6
}
```

#### kelly_brain_get_available_emotions()
```c
char* kelly_brain_get_available_emotions(const KellyBrain* brain);
```
**Purpose:** Get list of available emotions from thesaurus  
**Returns:** JSON array of emotion definitions  
**Example JSON:**
```json
{
  "emotions": [
    {"name": "joy", "category": "positive"},
    {"name": "sadness", "category": "negative"},
    {"name": "anger", "category": "negative"}
  ]
}
```

### Parameter Update Functions

#### kelly_brain_set_emotion_parameters()
```c
KellyErrorCode kelly_brain_set_emotion_parameters(
    KellyBrain* brain, 
    float valence, 
    float arousal, 
    float dominance
);
```
**Purpose:** Update emotional parameters directly  
**Parameters:**
- `valence` - Valence value (-1.0 to 1.0)
- `arousal` - Arousal value (0.0 to 1.0)  
- `dominance` - Dominance value (0.0 to 1.0)
**Returns:** Error code
**Side Effects:** May trigger registered callbacks

### Error Handling

#### Error Codes
```c
typedef enum {
    KELLY_SUCCESS = 0,
    KELLY_ERROR_NULL_POINTER = -1,
    KELLY_ERROR_INITIALIZATION_FAILED = -2,
    KELLY_ERROR_INVALID_PARAMETER = -3,
    KELLY_ERROR_JSON_PARSE_ERROR = -4,
    KELLY_ERROR_MEMORY_ALLOCATION = -5,
    KELLY_ERROR_FILE_NOT_FOUND = -6,
    KELLY_ERROR_UNKNOWN = -999
} KellyErrorCode;
```

#### kelly_get_error_message()
```c
const char* kelly_get_error_message(KellyErrorCode error_code);
```
**Purpose:** Get human-readable error message  
**Returns:** Static string describing error (do not free)

#### kelly_get_last_error()
```c
const char* kelly_get_last_error(void);
```
**Purpose:** Get last error message (thread-local)  
**Returns:** Error message or NULL if no error

### Memory Management

#### kelly_free_string()
```c
void kelly_free_string(char* ptr);
```
**Purpose:** Free memory allocated by kelly_* functions  
**Parameters:** Pointer to free (can be NULL)  
**Usage:** Must be called on all char* returned by kelly functions

## Tauri Commands API

### C++ Backend Commands

#### kelly_brain_initialize
```rust
async fn kelly_brain_initialize(data_path: String) -> Result<bool, String>
```
**Purpose:** Initialize KellyBrain with data directory  
**Parameters:**
- `data_path` - Path to emotion data directory  
**Returns:** `true` on success  
**Example:**
```typescript
const success = await invoke('kelly_brain_initialize', { 
  dataPath: './data' 
});
```

#### kelly_brain_from_text
```rust
async fn kelly_brain_from_text(text: String) -> Result<IntentResult, String>
```
**Purpose:** Generate intent from text description  
**Parameters:**
- `text` - Emotional description  
**Returns:** IntentResult object  
**Example:**
```typescript
const intent = await invoke('kelly_brain_from_text', { 
  text: 'I feel hopeful but uncertain' 
});
```

#### kelly_brain_from_emotion
```rust
async fn kelly_brain_from_emotion(emotion_name: String, intensity: f32) -> Result<IntentResult, String>
```
**Purpose:** Generate intent from emotion name and intensity  
**Parameters:**
- `emotion_name` - Name of emotion  
- `intensity` - Intensity value (0.0 to 1.0)  
**Returns:** IntentResult object

#### kelly_brain_generate_midi
```rust
async fn kelly_brain_generate_midi(intent: IntentResult, bars: i32) -> Result<GeneratedMidi, String>
```
**Purpose:** Generate MIDI from intent  
**Parameters:**
- `intent` - IntentResult from previous call  
- `bars` - Number of bars (1-64)  
**Returns:** GeneratedMidi object

#### kelly_brain_set_emotion_parameters
```rust
async fn kelly_brain_set_emotion_parameters(valence: f32, arousal: f32, dominance: f32) -> Result<bool, String>
```
**Purpose:** Update emotion parameters directly  
**Parameters:**
- `valence` - (-1.0 to 1.0)
- `arousal` - (0.0 to 1.0)  
- `dominance` - (0.0 to 1.0)
**Returns:** `true` on success

### State Management Commands

#### get_kelly_brain_state
```rust
async fn get_kelly_brain_state() -> Result<KellyBrainState, String>
```
**Purpose:** Get complete current state  
**Returns:** KellyBrainState object  
**Example:**
```typescript
const state = await invoke('get_kelly_brain_state');
console.log('Initialized:', state.initialized);
console.log('Emotion:', state.emotion_state);
```

#### subscribe_to_state_events
```rust
async fn subscribe_to_state_events(subscriber_id: String) -> Result<bool, String>
```
**Purpose:** Register for state change notifications  
**Parameters:**
- `subscriber_id` - Unique identifier for subscriber  
**Returns:** `true` on success

### Event Management Commands

#### add_event_listener
```rust
async fn add_event_listener(listener_id: String) -> Result<bool, String>
```
**Purpose:** Register event listener for real-time updates  
**Parameters:**
- `listener_id` - Unique identifier for listener  
**Returns:** `true` on success

### Legacy/Fallback Commands

#### generate_music (hybrid)
```rust
async fn generate_music(request: GenerateRequest) -> Result<serde_json::Value, String>
```
**Purpose:** Generate music with automatic fallback  
**Behavior:**
1. Tries C++ backend if initialized
2. Falls back to Python HTTP API
3. Returns consistent format

## React Hooks API

### useKellyBrain Hook

#### Primary Interface
```typescript
const {
  state,              // Current KellyBrain state
  error,              // Last error message
  loading,            // Loading indicator
  initialize,         // Initialize function
  fromText,           // Generate intent from text
  fromEmotion,        // Generate intent from emotion
  generateMidi,       // Generate MIDI from intent
  setEmotionParameters, // Update emotion values
  isInitialized,      // Computed: initialization status
  isProcessing,       // Computed: processing status
} = useKellyBrain();
```

#### State Object
```typescript
interface KellyBrainState {
  initialized: boolean;           // Is KellyBrain ready?
  emotion_state: EmotionState | null;  // Current emotion values
  current_intent: IntentResult | null; // Last generated intent
  current_midi: GeneratedMidi | null;  // Last generated MIDI
  processing: boolean;            // Is operation in progress?
  last_update: string | null;     // Timestamp of last update
}
```

#### EmotionState Type
```typescript
interface EmotionState {
  valence: number;     // -1.0 to 1.0 (negative to positive)
  arousal: number;     // 0.0 to 1.0 (calm to excited)
  dominance: number;   // 0.0 to 1.0 (submissive to dominant)
  complexity: number;  // 0.0 to 1.0 (simple to complex)
}
```

#### IntentResult Type
```typescript
interface IntentResult {
  core_wound: string;         // Identified core emotional wound
  core_desire: string;        // Core emotional desire
  emotional_intent: string;   // Processed emotional intent
  valence: number;            // Emotional valence
  arousal: number;            // Emotional arousal
  dominance: number;          // Emotional dominance
  complexity: number;         // Musical complexity
  progression: string[];      // Chord progression
  key: string;               // Musical key
  bpm: number;               // Beats per minute
  genre: string;             // Musical genre
}
```

#### GeneratedMidi Type
```typescript
interface GeneratedMidi {
  bars: number;              // Number of bars
  bpm: number;               // Beats per minute
  key: string;               // Musical key
  time_signature: string;    // Time signature
  tracks: MidiTrack[];       // MIDI tracks
}

interface MidiTrack {
  name: string;              // Track name
  channel: number;           // MIDI channel
  events: MidiEvent[];       // MIDI events
}

interface MidiEvent {
  event_type: number;        // Event type (note on/off, etc.)
  timestamp: number;         // Timestamp in song
  note: number;              // MIDI note number
  velocity: number;          // Note velocity
  duration: number;          // Note duration
}
```

### useKellyBrainEvents Hook

#### Event Monitoring
```typescript
const {
  events,           // Array of recent events
  isListening,      // Is event listening active?
  startListening,   // Start event monitoring
  stopListening,    // Stop event monitoring
  clearEvents,      // Clear event history
} = useKellyBrainEvents();
```

#### Event Types
```typescript
interface KellyEvent {
  type: string;       // Event type identifier
  data: any;          // Event-specific data
  timestamp: string;  // ISO timestamp
}
```

**Event Types:**
- `kelly-brain-initialized` - KellyBrain initialization complete
- `kelly-emotion-update` - Emotion state changed
- `kelly-intent-generated` - New intent generated
- `kelly-midi-generated` - MIDI generation complete
- `kelly-processing-started` - Operation started
- `kelly-processing-completed` - Operation completed
- `kelly-error` - Error occurred
- `kelly-connection-status` - Connection status changed

### useSimpleKellyBrain Hook

#### Simplified Interface
```typescript
const {
  initialized,           // Simple boolean status
  processing,            // Simple processing indicator
  error,                 // Last error
  emotionState,          // Current emotion
  currentMidi,           // Current MIDI
  initialize,            // Init function
  generateMusicFromText, // One-step text → MIDI
  setEmotion,           // Set emotion parameters
} = useSimpleKellyBrain();
```

#### generateMusicFromText Function
```typescript
async function generateMusicFromText(
  text: string, 
  bars: number = 8
): Promise<GeneratedMidi | null>
```
**Purpose:** One-step music generation from text  
**Flow:** text → intent → MIDI  
**Example:**
```typescript
const midi = await generateMusicFromText('melancholy winter evening', 16);
if (midi) {
  console.log('Generated', midi.tracks.length, 'tracks');
}
```

### useMusicBrain Hook (Enhanced)

#### Hybrid Integration
```typescript
const {
  // Original functions (now with C++ integration)
  getEmotions,          // Get emotions (C++ first, HTTP fallback)
  generateMusic,        // Generate music (C++ first, HTTP fallback)
  interrogate,          // Interrogate (C++ first, HTTP fallback)
  
  // Direct KellyBrain access
  kellyBrain,          // Full KellyBrain hook object
  initializeKellyBrain, // Direct initialization
  generateFromText,     // Direct text processing
  generateFromEmotion,  // Direct emotion processing
  setEmotionParameters, // Direct emotion setting
  
  // Legacy functions
  getHumanizerConfig,   // Python API only
  setUserLyrics,       // Python API only
  getUserLyrics,       // Python API only
} = useMusicBrain();
```

## Data Types Reference

### Error Handling Types

#### KellyError (Rust)
```rust
pub struct KellyError {
    pub code: KellyErrorCode,
    pub message: String,
}
```

#### KellyErrorCode (C/Rust)
```rust
pub enum KellyErrorCode {
    Success = 0,
    NullPointer = -1,
    InitializationFailed = -2,
    InvalidParameter = -3,
    JsonParseError = -4,
    MemoryAllocation = -5,
    FileNotFound = -6,
    Unknown = -999,
}
```

### Musical Data Types

#### Emotional Intent (Request Format)
```typescript
interface EmotionalIntent {
  core_wound?: string;        // Optional core wound
  core_desire?: string;       // Optional core desire
  emotional_intent: string;   // Required emotional description
  technical?: {               // Optional technical constraints
    key?: string;             // Musical key
    bpm?: number;             // Beats per minute
    progression?: string[];    // Chord progression
    genre?: string;           // Musical genre
  };
}
```

#### Generate Request (Legacy API)
```typescript
interface GenerateRequest {
  intent: EmotionalIntent;     // Emotional intent
  output_format?: string;      // Output format preference
}
```

#### Interrogate Request (Legacy API)
```typescript
interface InterrogateRequest {
  message: string;            // Message to interrogate
  session_id?: string;        // Optional session ID
  context?: any;              // Optional context
}
```

## API Usage Examples

### Basic Emotion Processing

```typescript
// Initialize
const { initialize, fromText, generateMidi } = useKellyBrain();

await initialize('./data');

// Process emotion
const intent = await fromText('feeling nostalgic about lost love');

// Generate music
const midi = await generateMidi(intent, 8);

console.log('Generated MIDI with', midi.tracks.length, 'tracks');
```

### Real-time Parameter Updates

```typescript
// Set up real-time emotion control
const { setEmotionParameters } = useKellyBrain();
const { startListening } = useKellyBrainEvents();

// Start listening for updates
await startListening();

// Update emotion in real-time
await setEmotionParameters(0.5, 0.8, 0.3); // valence, arousal, dominance

// Events will be emitted automatically
```

### Advanced MIDI Generation

```typescript
// Generate with specific parameters
const { generateMidiWithParams } = useKellyBrain();

const midi = await generateMidiWithParams(
  intent,
  16,      // bars
  140,     // BPM
  'Em'     // key
);

// Process MIDI tracks
for (const track of midi.tracks) {
  console.log(`Track ${track.name}: ${track.events.length} events`);
}
```

### Error Handling Patterns

```typescript
// Comprehensive error handling
try {
  const intent = await fromText(userInput);
  const midi = await generateMidi(intent, 8);
  
  // Success handling
  setResult(midi);
  
} catch (error) {
  // Error handling
  if (error.includes('not initialized')) {
    await initialize('./data');
    // Retry operation
  } else if (error.includes('invalid parameter')) {
    setValidationError('Please check your input');
  } else {
    setGenericError('Music generation failed');
  }
}
```

### Fallback API Usage

```typescript
// Use hybrid approach for backward compatibility
const { generateMusic } = useMusicBrain();

const result = await generateMusic({
  intent: {
    emotional_intent: 'feeling hopeful',
    technical: {
      key: 'C',
      bpm: 120,
      genre: 'ambient'
    }
  }
});

// Result includes source information
console.log('Generated via:', result.source); // 'kelly_brain_cpp' or 'python_http'
```

## Event API Reference

### Event Subscription

```typescript
import { listen } from '@tauri-apps/api/event';

// Listen to specific events
const unlisten = await listen('kelly-emotion-update', (event) => {
  const { valence, arousal, dominance } = event.payload;
  updateEmotionDisplay(valence, arousal, dominance);
});

// Cleanup
await unlisten();
```

### Available Events

#### kelly-brain-initialized
```json
{
  "success": true,
  "version": "KellyBrain FFI v1.0.0"
}
```

#### kelly-emotion-update
```json
{
  "valence": 0.2,
  "arousal": 0.7,
  "dominance": 0.5,
  "complexity": 0.6
}
```

#### kelly-intent-generated
```json
{
  "core_wound": "abandonment",
  "emotional_intent": "seeking connection",
  "valence": -0.3,
  "arousal": 0.6,
  "key": "Em",
  "bpm": 85,
  "progression": ["Em", "C", "G", "D"]
}
```

#### kelly-midi-generated
```json
{
  "bars": 8,
  "bpm": 85,
  "key": "Em",
  "tracks": [
    {
      "name": "Melody",
      "channel": 1,
      "events": [...]
    }
  ]
}
```

#### kelly-processing-started / kelly-processing-completed
```json
{
  "operation": "generate_midi",
  "timestamp": "2026-01-18T10:30:00Z"
}
```

#### kelly-error
```json
{
  "message": "Failed to generate MIDI: invalid parameters",
  "timestamp": "2026-01-18T10:30:00Z"
}
```

## Performance Considerations

### API Call Optimization

**High-Frequency Operations:**
- Use event system instead of polling
- Batch parameter updates
- Cache results where appropriate
- Use throttling for UI updates

**Best Practices:**
```typescript
// Good: Use events for real-time updates
useKellyBrainEvents().startListening();

// Avoid: Polling state rapidly
// setInterval(() => invoke('get_kelly_brain_state'), 100); // DON'T DO THIS
```

### Memory Management

**FFI Calls:**
- Rust automatically frees returned strings
- C++ pre-allocates buffers
- No manual memory management needed in TypeScript

**Large Data:**
- MIDI data can be substantial (>1MB for long pieces)
- Consider streaming for long compositions
- Implement progress callbacks for large operations

## Integration Testing

### API Testing

```bash
# Test C++ FFI directly
cd build/debug && ./KellyTests

# Test Rust integration
cd src-tauri && cargo test

# Test Tauri commands
npm run test:integration
```

### Manual Testing

```typescript
// Test initialization
await invoke('kelly_brain_initialize', { dataPath: './data' });

// Test basic flow
const intent = await invoke('kelly_brain_from_text', { text: 'test emotion' });
const midi = await invoke('kelly_brain_generate_midi', { intent, bars: 4 });

console.log('API integration successful');
```

---

This API reference provides complete documentation for all KmiDi interfaces. For implementation examples and advanced usage patterns, see the `DEVELOPMENT.md` guide and inline code documentation.