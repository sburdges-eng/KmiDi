# iDAW Agent Prompts & Quick Commands

## Quick Copy-Paste Setup

### One-Liner Full Setup
```bash
curl -sSL https://raw.githubusercontent.com/sburdges-eng/iDAWi/main/setup-idaw.sh | bash -s full
```

### Manual Setup (Copy All)
```bash
# Clone and enter repo
git clone https://github.com/sburdges-eng/iDAWi.git && cd iDAWi

# Create all config directories
mkdir -p .claude .devcontainer .github/workflows src-tauri/src/{audio,commands,python} src/{components,emotion,hooks,stores} python/idaw_generator

# Install dependencies
npm install
pip install numpy midiutil black mypy

# Start development
npm run tauri dev
```

---

## Claude Code Prompts

### Session Start Prompt
```
I'm working on iDAW, a Tauri 2.0 DAW with emotion-driven music generation.

Stack: Rust (audio engine) + React/TypeScript (UI) + Python (ML generation)

Key files:
- src-tauri/src/audio/ - Audio processing (lock-free, real-time safe)
- src/emotion/ - Emotion thesaurus (6×6×6 = 216 nodes)
- src/stores/dawStore.ts - Zustand state management
- python/idaw_generator/ - MIDI generation from emotions

Rules:
1. Audio thread: No allocations, no locks, no blocking
2. Parameters: Use SmoothedParameter with atomics
3. IPC: Debounce Tauri invoke calls (16ms minimum)
4. Python: Async generation, return MIDI bytes only

Current task: [DESCRIBE YOUR TASK]
```

### Audio Processor Prompt
```
Create a Rust audio processor for [EFFECT_NAME].

Requirements:
1. Implement AudioProcessor trait from src-tauri/src/audio/traits.rs
2. Parameters: [LIST: e.g., gain (0-2), frequency (20-20000Hz)]
3. Use SmoothedParameter for all controllable values
4. Report accurate latency_samples() if any lookahead
5. Include unit tests

Template:
```rust
pub struct [EffectName] {
    // parameters
}

impl AudioProcessor for [EffectName] {
    fn process(&mut self, buffer: &mut AudioBuffer) { }
    fn latency_samples(&self) -> u64 { 0 }
    fn reset(&mut self) { }
    fn prepare(&mut self, config: AudioConfig) { }
    fn name(&self) -> &str { "[EffectName]" }
}
```
```

### React Component Prompt
```
Create a React component for [COMPONENT_NAME].

Requirements:
1. TypeScript with strict types
2. Use Zustand store (useDAWStore) for state
3. Tauri invoke for backend communication
4. Tailwind CSS (core utilities only)
5. Proper memoization for performance

Connect to: [STORE_SLICE or TAURI_COMMAND]
```

### Emotion Mapping Prompt
```
Add emotion mapping for [EMOTION_NAME].

V-A-D coordinates: valence=[X], arousal=[Y], dominance=[Z]

Update these files:
1. src/emotion/thesaurus.ts - Add to EMOTION_NAMES, verify mapEmotionToMusic()
2. python/idaw_generator/__init__.py - Update _emotion_to_params()

Musical characteristics: [DESCRIBE: tempo, mode, dynamics, etc.]
```

---

## Copilot Chat Prompts

### @workspace Commands
```
@workspace /explain the audio graph processing order in src-tauri/src/audio/graph.rs

@workspace /fix the latency compensation calculation for parallel plugin chains

@workspace /tests for the SmoothedParameter struct

@workspace /new create a compressor audio processor following the project conventions
```

### Inline Suggestions Setup
Add to `.vscode/settings.json`:
```json
{
  "github.copilot.advanced": {
    "inlineSuggestCount": 3,
    "listCount": 10
  },
  "github.copilot.editor.enableAutoCompletions": true,
  "github.copilot.editor.enableCodeActions": true
}
```

### Copilot Comment Triggers
```rust
// Generate: AudioProcessor for parametric EQ with 3 bands
// Each band has: frequency (20-20000Hz log), gain (-12 to +12 dB), Q (0.1 to 10)

// Implement: Lock-free ring buffer for audio streaming
// Size: configurable, Thread-safe: yes, Overwrite: oldest on full

// Convert: This synchronous file read to async Tauri command
```

```typescript
// Create: Zustand slice for mixer state with tracks, sends, and master

// Component: Waveform display with WebGL rendering and zoom/scroll

// Hook: useAudioEngine that syncs transport state with Rust backend
```

---

## GitHub Copilot Agent Workflows

### Issue Commands
Create issues with these titles for Copilot to understand:

```markdown
## [Feature] Add reverb audio processor
@copilot implement

Create a reverb effect with:
- Room size parameter (0-1)
- Damping parameter (0-1)  
- Wet/dry mix (0-1)
- Pre-delay (0-100ms)

Follow AudioProcessor trait pattern.
```

```markdown
## [Bug] Parameter smoothing causes clicks at fast changes
@copilot review

SmoothedParameter in src-tauri/src/audio/parameters.rs
produces audible clicks when target changes rapidly.

Expected: Smooth transitions regardless of rate
Actual: Clicks when changing >10x per second
```

### PR Review Commands
```markdown
@copilot review this PR for:
- Real-time safety (no allocations in audio thread)
- Proper error handling
- Type safety
- Performance concerns
```

### Auto-labeling Workflow
```yaml
# .github/workflows/auto-label.yml
name: Auto Label

on:
  issues:
    types: [opened]
  pull_request:
    types: [opened]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v5
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          configuration-path: .github/labeler.yml
```

```yaml
# .github/labeler.yml
rust:
  - src-tauri/**/*.rs
  
typescript:
  - src/**/*.ts
  - src/**/*.tsx
  
python:
  - python/**/*.py
  
audio:
  - src-tauri/src/audio/**
  - src/hooks/useAudioEngine.ts
  
emotion:
  - src/emotion/**
  - python/idaw_generator/**
```

---

## Codespace Quick Actions

### Start Development
```bash
# In Codespace terminal
npm run tauri dev
```

### Run All Tests
```bash
cargo test --manifest-path src-tauri/Cargo.toml
npm run test
python -m pytest python/
```

### Format All Code
```bash
cargo fmt --manifest-path src-tauri/Cargo.toml
npx prettier --write "src/**/*.{ts,tsx}"
black python/
```

### Pre-commit Check
```bash
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings && \
npm run lint && \
npm run type-check && \
black --check python/ && \
mypy python/
```

---

## File Templates

### New Audio Processor
```bash
cat > src-tauri/src/audio/NEW_EFFECT.rs << 'EOF'
use super::{AudioBuffer, AudioConfig, AudioProcessor};
use super::parameters::SmoothedParameter;

pub struct NewEffect {
    param1: SmoothedParameter,
    config: Option<AudioConfig>,
}

impl NewEffect {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            param1: SmoothedParameter::new(1.0, 10.0, sample_rate),
            config: None,
        }
    }
    
    pub fn set_param1(&self, value: f32) {
        self.param1.set_target(value);
    }
}

impl AudioProcessor for NewEffect {
    fn process(&mut self, buffer: &mut AudioBuffer) {
        for frame in 0..buffer.frames {
            self.param1.tick();
            let gain = self.param1.get();
            
            for ch in 0..buffer.channels {
                let sample = buffer.get(ch, frame);
                buffer.set(ch, frame, sample * gain);
            }
        }
    }
    
    fn latency_samples(&self) -> u64 { 0 }
    
    fn reset(&mut self) {
        // Reset internal state
    }
    
    fn prepare(&mut self, config: AudioConfig) {
        self.config = Some(config);
    }
    
    fn name(&self) -> &str { "NewEffect" }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_effect() {
        let mut effect = NewEffect::new(44100);
        let mut buffer = AudioBuffer::new(2, 512, 44100);
        effect.prepare(AudioConfig {
            sample_rate: 44100,
            buffer_size: 512,
            channels: 2,
            bit_depth: super::super::BitDepth::F32,
        });
        effect.process(&mut buffer);
    }
}
EOF
```

### New React Component
```bash
cat > src/components/NewComponent.tsx << 'EOF'
import React, { useMemo, useCallback } from 'react';
import { useDAWStore } from '../stores/dawStore';
import { invoke } from '@tauri-apps/api/core';

interface NewComponentProps {
  trackId: string;
}

export function NewComponent({ trackId }: NewComponentProps) {
  const track = useDAWStore((state) => state.tracks.get(trackId));
  const setTrackVolume = useDAWStore((state) => state.setTrackVolume);
  
  const handleVolumeChange = useCallback(async (value: number) => {
    setTrackVolume(trackId, value);
    await invoke('set_track_volume', { trackId, volumeDb: linearToDb(value) });
  }, [trackId, setTrackVolume]);
  
  if (!track) return null;
  
  return (
    <div className="flex flex-col p-4 bg-gray-800 rounded-lg">
      <span className="text-white font-medium">{track.name}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={track.volume}
        onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
        className="w-full mt-2"
      />
    </div>
  );
}

function linearToDb(linear: number): number {
  return 20 * Math.log10(Math.max(linear, 0.0001));
}
EOF
```

### New Emotion Node
```bash
cat >> src/emotion/thesaurus.ts << 'EOF'

// Add to EMOTION_NAMES object:
// 'e_X_Y_Z': 'NewEmotionName',

// Verify mapEmotionToMusic handles the V/A/D values correctly
// Update Python generator if needed
EOF
```

---

## Debugging Commands

### Rust Audio Engine
```bash
# Run with debug logging
RUST_LOG=debug cargo run --manifest-path src-tauri/Cargo.toml

# Profile audio performance
cargo flamegraph --manifest-path src-tauri/Cargo.toml -- --bench audio

# Check for memory issues
cargo valgrind --manifest-path src-tauri/Cargo.toml test
```

### React Frontend
```bash
# Start with React DevTools
TAURI_DEBUG=true npm run tauri dev

# Profile React performance
npm run dev -- --profile
```

### Python Generator
```bash
# Test generation directly
python -c "from idaw_generator import generate; print(generate(valence=0.5, arousal=0.3, dominance=0.2, duration=10.0))"

# Profile generation
python -m cProfile -s cumtime python/idaw_generator/__init__.py
```

---

## Environment Variables

```bash
# Development
export TAURI_DEBUG=true
export RUST_LOG=debug
export RUST_BACKTRACE=1

# Production build
export TAURI_PRIVATE_KEY="your-key"
export TAURI_KEY_PASSWORD="your-password"

# Python path for Rust bridge
export PYTHONPATH="${PWD}/python:${PYTHONPATH}"
```

---

## Git Workflow

```bash
# Feature branch
git checkout -b feature/[feature-name]

# Commit with conventional format
git commit -m "feat(audio): add parametric EQ processor"
git commit -m "fix(ui): resolve waveform rendering at high zoom"
git commit -m "docs: update emotion mapping documentation"

# Push and create PR
git push -u origin feature/[feature-name]
gh pr create --fill
```
