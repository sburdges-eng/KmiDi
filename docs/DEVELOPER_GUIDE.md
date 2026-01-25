# KmiDi Developer Guide

**Version:** 1.0  
**Date:** January 18, 2026  
**Target Audience:** Developers working on KmiDi

## Overview

This guide provides comprehensive information for developers working on the KmiDi project, including component patterns, integration patterns, build system, and development workflows.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Component Patterns](#component-patterns)
3. [State Management](#state-management)
4. [Persistence](#persistence)
5. [Styling Guidelines](#styling-guidelines)
6. [Build System](#build-system)
7. [Integration Patterns](#integration-patterns)
8. [Testing](#testing)
9. [Code Style](#code-style)

## Project Structure

### Directory Layout

```
KmiDi/
├── src/
│   ├── components/          # React UI components
│   ├── hooks/               # React hooks
│   ├── utils/               # Utility functions
│   ├── audio/               # Audio processing (C++)
│   ├── engine/              # Music engine (C++)
│   ├── ml/                  # ML inference (C++)
│   ├── dsp/                 # Pure DSP (C++)
│   ├── bridge/              # FFI bridge code
│   └── App.tsx              # Main React app
├── src-tauri/               # Tauri Rust bridge
├── docs/                    # Documentation
├── tests/                   # Test files
└── scripts/                 # Build and utility scripts
```

### Key Directories

- **`src/components/`** - React components following Spec 02/03/04 requirements
- **`src/hooks/`** - Custom React hooks for state and persistence
- **`src/utils/`** - Utility functions (persistence, helpers)
- **`src/dsp/`** - Pure DSP code (no framework dependencies)
- **`src/audio/`** - Audio analysis (migrating from JUCE)
- **`src/engine/`** - Music generation engine
- **`src/ml/`** - ML inference and feature extraction

## Component Patterns

### Component Structure

All React components should follow this structure:

```typescript
/**
 * ComponentName - Brief description
 * 
 * Detailed description of component purpose and usage.
 * Per Spec XX: [Specification reference]
 */

import React from 'react';

export interface ComponentNameProps {
  // Props with JSDoc comments
  /** Description of prop */
  propName: string;
}

/**
 * ComponentName component
 * 
 * @param props - Component props
 * @returns React component
 */
export const ComponentName: React.FC<ComponentNameProps> = ({ propName }) => {
  // Component implementation
  return (
    <div>
      {/* Component JSX */}
    </div>
  );
};

export default ComponentName;
```

### Three-Panel Layout Components

Per Spec 02, the application uses a three-panel layout:

1. **InspectorPanel** (`src/components/InspectorPanel.tsx`)
   - Left panel, read-only
   - Displays selected item details
   - Uses semantic tokens and 4pt grid

2. **Timeline** (`src/components/Timeline.tsx`)
   - Center panel, primary interface
   - Grid rules and zoom behavior
   - Track scaling per Spec 04

3. **BrowserPanel** (`src/components/BrowserPanel.tsx`)
   - Right panel, utility
   - File/media browser
   - Follows visual system

### Visual System Compliance

All components must:

- Use Tailwind semantic tokens (no hardcoded colors)
- Follow 4pt baseline grid for spacing
- Use minimum 11pt font size
- Use minimum 44pt touch targets
- Support dark-first design

Example:

```typescript
<div className="p-4 bg-bg-secondary border border-border-light rounded">
  <button className="px-4 py-2 bg-accent-primary text-white rounded min-h-touch min-w-touch">
    Action
  </button>
</div>
```

## State Management

### React Hooks

Use React hooks for component state:

```typescript
const [state, setState] = useState<StateType>(initialValue);
```

### Custom Hooks

Custom hooks are available in `src/hooks/`:

- **`usePersistence`** - Panel and window state persistence
- **`useMusicBrain`** - Music Brain API integration
- **`useIntentIR`** - Intent IR frame management

### Persistence Hooks

```typescript
import { usePanelPersistence, useWindowPreferences, useSidePreference } from '../hooks/usePersistence';

// Panel state
const { panelState, updatePanelState } = usePanelPersistence();

// Window preferences
const { preferences, updatePreferences } = useWindowPreferences();

// Side A/B preference
const { sideA, updateSidePreference } = useSidePreference();
```

## Persistence

### Persistence Utilities

Persistence is handled by `src/utils/persistence.ts`:

- Uses Tauri store API when available
- Falls back to localStorage in browser mode
- Automatically saves/loads state

### Saving State

```typescript
import { savePanelState, saveWindowPreferences } from '../utils/persistence';

await savePanelState({
  inspector: { visible: true, width: 300 },
  timeline: { visible: true },
  browser: { visible: true, width: 250 },
});

await saveWindowPreferences({
  width: 1200,
  height: 800,
  sideA: true,
});
```

### Loading State

```typescript
import { loadPanelState, loadWindowPreferences } from '../utils/persistence';

const panelState = await loadPanelState();
const prefs = await loadWindowPreferences();
```

## Styling Guidelines

### Tailwind Semantic Tokens

Always use semantic tokens from `tailwind.config.js`:

**Backgrounds:**
- `bg-primary` - Primary background
- `bg-secondary` - Secondary background
- `bg-tertiary` - Tertiary background

**Text:**
- `text-primary` - Primary text
- `text-secondary` - Secondary text
- `text-tertiary` - Tertiary text

**Accents:**
- `accent-primary` - Primary accent (blue)
- `accent-success` - Success (green)
- `accent-warning` - Warning (orange)
- `accent-error` - Error (red)

**Borders:**
- `border-light` - Light border
- `border-medium` - Medium border

### Spacing (4pt Grid)

Use spacing units from Tailwind config:
- `p-1` = 4px
- `p-2` = 8px
- `p-3` = 12px
- `p-4` = 16px
- etc.

### Touch Targets

All interactive elements must meet minimum 44pt:
- Use `min-h-touch` and `min-w-touch` classes
- Or ensure padding creates 44pt minimum

### Accessibility

- Add `aria-label` to all interactive elements
- Use `role` attributes where appropriate
- Support keyboard navigation
- Ensure color contrast meets WCAG AA

## Build System

### Prerequisites

- **Node.js** 18+ and npm
- **Rust** (for Tauri)
- **CMake** (for C++ builds)
- **Python** 3.9+ (for ML backend)

### Development Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Start Tauri dev
npm run dev:tauri

# Build C++ code
npm run build:cpp
```

### Build Commands

- `npm run build` - Build React app
- `npm run build:cpp` - Build C++ code (debug)
- `npm run build:cpp-release` - Build C++ code (release)
- `npm run tauri build` - Build Tauri app

### Type Checking

```bash
npm run lint:ts  # TypeScript type checking
```

## Integration Patterns

### Tauri Integration

Tauri commands are defined in `src-tauri/src/`:

```typescript
import { invoke } from '@tauri-apps/api/core';

const result = await invoke('command_name', { param: value });
```

### C++ Bridge Integration

C++ functions are exposed via FFI bridge:

```typescript
// Bridge functions are available through Tauri commands
// or direct FFI calls (depending on implementation)
```

### ML Backend Integration

ML backend is accessed via HTTP API:

```typescript
import { useMusicBrain } from './hooks/useMusicBrain';

const { generateMusic, getEmotions } = useMusicBrain();
const result = await generateMusic(payload);
```

## Testing

### Component Testing

Components should be tested with React Testing Library:

```typescript
import { render, screen } from '@testing-library/react';
import { InspectorPanel } from './InspectorPanel';

test('renders inspector panel', () => {
  render(<InspectorPanel selectedItem={mockItem} />);
  expect(screen.getByText('Inspector')).toBeInTheDocument();
});
```

### Integration Testing

Integration tests verify component interactions:

```bash
npm run test:integration
```

## Code Style

### TypeScript

- Use strict mode
- Prefer interfaces over types for props
- Use explicit return types for functions
- Avoid `any` type

### React

- Use functional components
- Use hooks for state management
- Memoize expensive computations
- Extract reusable logic to custom hooks

### Naming Conventions

- **Components:** PascalCase (`InspectorPanel`)
- **Files:** Match component name (`InspectorPanel.tsx`)
- **Hooks:** camelCase with `use` prefix (`usePersistence`)
- **Utilities:** camelCase (`persistence.ts`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_VALUE`)

## Component Documentation

### JSDoc Comments

All components should have JSDoc comments:

```typescript
/**
 * ComponentName - Brief description
 * 
 * Detailed description including:
 * - Purpose
 * - Key features
 * - Usage examples
 * - Related components
 * 
 * @example
 * ```tsx
 * <ComponentName prop="value" />
 * ```
 */
```

### Props Documentation

Document all props:

```typescript
interface ComponentProps {
  /** Description of prop */
  propName: string;
  
  /** Optional prop description */
  optionalProp?: number;
}
```

## Related Documentation

- `docs/ARCHITECTURE.md` - System architecture
- `docs/AI_CONTROL_LAYER.md` - AI architecture boundaries
- `docs/HOST_GLUE_ARCHITECTURE.md` - Host integration patterns
- `docs/STRUCTURE_CROSS_EXAMINATION/` - Compliance reports
- `docs/specs/` - Specification documents

## Getting Help

- Check existing documentation in `docs/`
- Review component examples in `src/components/`
- Check test files in `tests/` for usage examples
- Review cross-examination reports for compliance requirements
