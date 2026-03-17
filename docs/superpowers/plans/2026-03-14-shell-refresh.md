# Shell Refresh — "The Console" Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat 4-tab shell with a split-deck console layout: persistent monitoring strip (upper), vertical nav rail + creative workspace (lower), session bar, noise grain, per-mode atmospheric glow, and ≤200ms transitions.

**Architecture:** Two new files — `src/AppConsole.tsx` (standalone shell with NavRail, MonitoringStrip, SessionBar inlined) and `src/console-shell.css` (all new layout/transition/texture styles). These are review-only files; the existing `App.tsx` and `index.css` are untouched. The new CSS depends on existing `:root` design tokens from `index.css`.

**Tech Stack:** React 19, TypeScript 5.8, CSS (using existing design tokens), Vite 7

**Spec:** `docs/superpowers/specs/2026-03-14-shell-refresh-design.md`

**Intentional omissions from current App.tsx:**

- **Session log panel** — The scrollable `<ul>` of all interactions (currently in Inspire mode) is replaced by the SessionBar showing only the last interaction. The full log is not surfaced in any mode. This is intentional per the spec.
- **Reset button** — The `km-titlebar` Reset button (`setIsPlaying(false)`) is removed. The Transport Stop button in the upper deck serves the same purpose.
- **`logRef` and scroll behavior** — Removed since the full session log panel no longer exists.

**Spec deviation (section 9.1 vs 11):** The spec lists `MonitoringStrip.tsx`, `NavRail.tsx`, and `SessionBar.tsx` as separate files (section 9.1) but also says the deliverable is "a single standalone `.tsx` file...with inlined components" (section 11). This plan follows section 11 — all shell components are inlined in `AppConsole.tsx` for review. They can be extracted to separate files during integration.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/console-shell.css` | Create | All CSS for the console shell: layout grid, upper deck, lower deck, nav rail, session bar, noise grain overlay, per-mode glow, mode transitions, panel styles, responsive/reduced-motion |
| `src/AppConsole.tsx` | Create | Complete shell with NavRail, MonitoringStrip, SessionBar defined inline. All state/logic from current App.tsx preserved. Imports all existing components + `console-shell.css` |

No existing files are modified.

---

## Chunk 1: Console Shell CSS

### Task 1: Base Layout Grid

**Files:**
- Create: `src/console-shell.css`

- [ ] **Step 1: Create the CSS file with root layout**

```css
/* ═══════════════════════════════════════════════════
   KmiDi Console Shell — split-deck layout
   Depends on :root tokens from index.css
   ═══════════════════════════════════════════════════ */

/* ─── Full-viewport shell ─── */
.console-shell {
  display: grid;
  grid-template-rows: 160px 1fr;
  width: 100%;
  min-height: 100vh;
  background: var(--surface);
  position: relative;
  overflow: hidden;
}

/* ─── Upper Deck: monitoring strip ─── */
.upper-deck {
  display: grid;
  grid-template-columns: 200px 1fr 180px;
  align-items: stretch;
  background: var(--surface-inset);
  border-bottom: 1px solid var(--border-strong);
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.03) inset;
  padding: var(--space-3) var(--space-4);
  gap: 0;
  position: relative;
  z-index: 2;
}

.upper-deck__zone {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 var(--space-4);
}

.upper-deck__zone:not(:last-child) {
  border-right: 1px solid var(--border);
}

.upper-deck__zone:first-child {
  padding-left: 0;
}

.upper-deck__zone:last-child {
  padding-right: 0;
}

/* Wordmark in transport zone */
.console-wordmark {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.03em;
  margin: 0 0 var(--space-2) 0;
}

/* ─── Lower Deck: nav rail + workspace + session bar ─── */
.lower-deck {
  display: grid;
  grid-template-columns: 48px 1fr;
  grid-template-rows: 1fr 32px;
  overflow: hidden;
  position: relative;
}
```

- [ ] **Step 2: Verify file is valid CSS**

Run: `cat src/console-shell.css | head -5`
Expected: Comment header appears

- [ ] **Step 3: Commit**

```bash
git add src/console-shell.css
git commit -m "feat(shell): base console layout grid — upper deck, lower deck, zones"
```

---

### Task 2: Nav Rail Styles

**Files:**
- Modify: `src/console-shell.css`

- [ ] **Step 1: Add nav rail CSS**

Append to `src/console-shell.css`:

```css
/* ─── Nav Rail ─── */
.nav-rail {
  grid-column: 1;
  grid-row: 1 / -1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: var(--space-5);
  gap: var(--space-4);
  z-index: 1;
}

.nav-rail__item {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: none;
  background: transparent;
  cursor: pointer;
  padding: 0;
  border-radius: var(--radius);
  transition: color 120ms ease;
}

.nav-rail__item svg {
  width: 20px;
  height: 20px;
  color: var(--text-muted);
  transition: color 120ms ease;
}

.nav-rail__item:hover svg {
  color: var(--text-secondary);
}

.nav-rail__item[aria-selected="true"] svg {
  color: var(--text-primary);
}

/* Active LED */
.nav-rail__item[aria-selected="true"]::before {
  content: '';
  position: absolute;
  left: -2px;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent-muted);
  animation: led-throb 200ms ease-out;
}

@keyframes led-throb {
  0% { transform: translateY(-50%) scale(1); opacity: 0.5; }
  50% { transform: translateY(-50%) scale(1.8); opacity: 1; }
  100% { transform: translateY(-50%) scale(1); opacity: 1; }
}

/* Tooltip */
.nav-rail__tooltip {
  position: absolute;
  left: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%);
  background: var(--surface-overlay);
  color: var(--text-primary);
  font-family: var(--font-ui);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
  padding: 4px 8px;
  border-radius: 4px;
  white-space: nowrap;
  pointer-events: none;
  opacity: 0;
  transition: opacity 120ms ease;
  z-index: 10;
}

.nav-rail__item:hover .nav-rail__tooltip,
.nav-rail__item:focus-visible .nav-rail__tooltip {
  opacity: 1;
}

.nav-rail__item:focus-visible {
  outline: 2px solid var(--accent-focus);
  outline-offset: 2px;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/console-shell.css
git commit -m "feat(shell): nav rail styles — LED indicator, tooltip, focus states"
```

---

### Task 3: Workspace, Session Bar, Panel Styles

**Files:**
- Modify: `src/console-shell.css`

- [ ] **Step 1: Add workspace and session bar CSS**

Append to `src/console-shell.css`:

```css
/* ─── Workspace ─── */
.workspace {
  grid-column: 2;
  grid-row: 1;
  position: relative;
  overflow-y: auto;
  padding: var(--space-4);
}

/* ─── Session Bar ─── */
.session-bar {
  grid-column: 2;
  grid-row: 2;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-4);
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-ui);
}

.session-bar__text {
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 80%;
}

.session-bar__status {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: 11px;
}

.session-bar__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-muted);
}

.session-bar__dot--online {
  background: var(--success);
  box-shadow: 0 0 4px var(--success);
}

.session-bar__dot--offline {
  background: var(--danger);
}

/* ─── Console Panels ─── */
.console-panel {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  transition: border-color 150ms ease;
}

.console-panel:hover {
  border-color: var(--border-hover);
}

.console-panel__title {
  font-family: var(--font-ui);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 400;
  margin: 0 0 var(--space-3) 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/console-shell.css
git commit -m "feat(shell): workspace, session bar, and console panel styles"
```

---

### Task 4: Mode Layouts

**Files:**
- Modify: `src/console-shell.css`

- [ ] **Step 1: Add per-mode grid layouts**

Append to `src/console-shell.css`:

```css
/* ─── Mode Layouts ─── */

/* Inspire: left 40% emotion wheel, right 60% split interrogator/ghostwriter */
.mode-inspire {
  display: grid;
  grid-template-columns: 2fr 3fr;
  grid-template-rows: 1fr 1fr;
  gap: var(--space-3);
  height: 100%;
}

.mode-inspire > :first-child {
  grid-row: 1 / -1;
}

/* Create: full-width starters, 2-col middle, full-width spectocloud */
.mode-create {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto 1fr auto;
  gap: var(--space-3);
  height: 100%;
}

.mode-create > :first-child {
  grid-column: 1 / -1;
}

.mode-create > :last-child {
  grid-column: 1 / -1;
}

/* Compose: full bleed */
.mode-compose {
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: 1fr;
  height: 100%;
}

/* Mix Detail: full bleed */
.mode-mix-detail {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
  height: 100%;
  align-content: start;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/console-shell.css
git commit -m "feat(shell): per-mode grid layouts — inspire, create, compose, mix detail"
```

---

### Task 5: Noise Grain, Atmospheric Glow, Transitions

**Files:**
- Modify: `src/console-shell.css`

- [ ] **Step 1: Add texture, glow, and motion CSS**

Append to `src/console-shell.css`:

```css
/* ─── Noise Grain Overlay ─── */
.console-shell::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  background-repeat: repeat;
  background-size: 256px 256px;
  pointer-events: none;
  z-index: 0;
  opacity: 0.4;
}

/* ─── Per-Mode Atmospheric Glow ─── */
.workspace::before {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  transition: background 200ms ease;
  border-radius: inherit;
}

.workspace--inspire::before {
  background: radial-gradient(ellipse at 20% 80%, rgba(201, 162, 39, 0.04) 0%, transparent 60%);
}

.workspace--create::before {
  background: radial-gradient(ellipse at 50% 50%, rgba(45, 138, 94, 0.04) 0%, transparent 60%);
}

.workspace--compose::before {
  background: radial-gradient(ellipse at 50% 20%, rgba(242, 242, 241, 0.03) 0%, transparent 60%);
}

.workspace--mix-detail::before {
  background: none;
}

/* ─── Mode Transitions (200ms max) ─── */
.mode-enter {
  animation: mode-in 120ms ease-out forwards;
}

/* Applied to .workspace when transitioning out */
.workspace--exiting > * {
  animation: mode-out 80ms ease-out forwards;
}

@keyframes mode-out {
  from { opacity: 1; }
  to { opacity: 0; }
}

@keyframes mode-in {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ─── Reduced Motion ─── */
@media (prefers-reduced-motion: reduce) {
  .mode-enter,
  .workspace--exiting > * {
    animation-duration: 100ms;
  }

  @keyframes mode-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes led-throb {
    from { opacity: 1; }
    to { opacity: 1; }
  }

  .workspace::before {
    transition: none;
  }
}

/* ─── Upper Deck Compact Overrides ─── */
.upper-deck .transport-grid {
  gap: var(--space-2);
}

.upper-deck .timeline {
  overflow: hidden;
}

.upper-deck .timeline-meta {
  font-size: 11px;
}

.upper-deck .vu-wrap {
  padding: 0;
}

.upper-deck .mixer-grid {
  display: flex;
  gap: var(--space-2);
  flex-wrap: nowrap;
}

.upper-deck .mixer-strip {
  flex: 1;
  min-width: 0;
}

.upper-deck .mixer-label {
  font-size: 10px;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/console-shell.css
git commit -m "feat(shell): noise grain, atmospheric glow, mode transitions, reduced motion"
```

---

## Chunk 2: AppConsole.tsx

### Task 6: Create AppConsole.tsx — NavRail and SessionBar inline components

**Files:**
- Create: `src/AppConsole.tsx`

- [ ] **Step 1: Create the file with imports, types, and inline NavRail/SessionBar**

```tsx
import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Transport } from './components/SideA/Transport';
import { Mixer } from './components/SideA/Mixer';
import { Timeline } from './components/SideA/Timeline';
import { VUMeter } from './components/SideA/VUMeter';
import { EmotionWheel } from './components/SideB/EmotionWheel';
import { GhostWriter } from './components/SideB/GhostWriter';
import { Interrogator } from './components/SideB/Interrogator';
import IntentBuilder from './components/IntentBuilder';
import { QuickStartPanel } from './components/QuickStartPanel';
import LyricPanel from './components/LyricPanel';
import { SpectoCloudPanel } from './components/SpectoCloudPanel';
import { MusicCustomizer } from './components/MusicCustomizer';
import { useMusicBrain } from './hooks/useMusicBrain';
import './console-shell.css';

type Mode = 'mix-detail' | 'inspire' | 'create' | 'compose';

type Channel = {
  id: string;
  name: string;
  level: number;
  pan: number;
};

type SelectedEmotion = {
  base: string;
  intensity: string;
  detail: string;
};

/* ─── Nav Rail Icons (inline SVG) ─── */

const NAV_ITEMS: { id: Mode; label: string; icon: JSX.Element }[] = [
  {
    id: 'mix-detail',
    label: 'Mix Detail',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="10" cy="10" r="7" />
        <circle cx="10" cy="10" r="2" />
      </svg>
    ),
  },
  {
    id: 'inspire',
    label: 'Inspire',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 3a7 7 0 0 1 0 14" />
        <circle cx="10" cy="10" r="7" />
      </svg>
    ),
  },
  {
    id: 'create',
    label: 'Create',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 3l2 5h5l-4 3.5 1.5 5L10 13l-4.5 3.5L7 11.5 3 8h5z" />
      </svg>
    ),
  },
  {
    id: 'compose',
    label: 'Compose',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="3" width="6" height="6" rx="1" />
        <rect x="11" y="3" width="6" height="6" rx="1" />
        <rect x="3" y="11" width="6" height="6" rx="1" />
        <rect x="11" y="11" width="6" height="6" rx="1" />
      </svg>
    ),
  },
];

function NavRail({
  activeMode,
  onModeChange,
}: {
  activeMode: Mode;
  onModeChange: (mode: Mode) => void;
}) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    const currentIndex = NAV_ITEMS.findIndex((item) => item.id === activeMode);
    let nextIndex = currentIndex;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      nextIndex = (currentIndex + 1) % NAV_ITEMS.length;
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      nextIndex = (currentIndex - 1 + NAV_ITEMS.length) % NAV_ITEMS.length;
    } else {
      return; // Let Enter/Space fall through to native button click
    }

    onModeChange(NAV_ITEMS[nextIndex].id);
  };

  return (
    <nav
      className="nav-rail"
      role="tablist"
      aria-orientation="vertical"
      aria-label="Studio mode"
      onKeyDown={handleKeyDown}
    >
      {NAV_ITEMS.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={activeMode === item.id}
          aria-controls="workspace-panel"
          id={`tab-${item.id}`}
          tabIndex={activeMode === item.id ? 0 : -1}
          className="nav-rail__item"
          onClick={() => onModeChange(item.id)}
        >
          {item.icon}
          <span className="nav-rail__tooltip">{item.label}</span>
        </button>
      ))}
    </nav>
  );
}

function SessionBar({
  lastInteraction,
  apiStatus,
}: {
  lastInteraction: string;
  apiStatus: 'checking' | 'online' | 'offline';
}) {
  const statusLabel = apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking';
  const dotClass = apiStatus === 'online'
    ? 'session-bar__dot session-bar__dot--online'
    : apiStatus === 'offline'
      ? 'session-bar__dot session-bar__dot--offline'
      : 'session-bar__dot';

  return (
    <footer className="session-bar" aria-live="polite">
      <p className="session-bar__text">{lastInteraction}</p>
      <span className="session-bar__status">
        <span className={dotClass} aria-hidden="true" />
        {statusLabel}
      </span>
    </footer>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/AppConsole.tsx
git commit -m "feat(shell): AppConsole with NavRail and SessionBar inline components"
```

---

### Task 7: AppConsole main component — state and upper deck

**Files:**
- Modify: `src/AppConsole.tsx`

- [ ] **Step 1: Add the main App component with all state and upper deck render**

Append to `src/AppConsole.tsx`:

```tsx
export default function AppConsole() {
  const [mode, setMode] = useState<Mode>('inspire');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [tempo, setTempo] = useState(120);
  const [masterVu, setMasterVu] = useState(0);
  const playTickRef = useRef(0);
  const [channels, setChannels] = useState<Channel[]>([
    { id: 'kick', name: 'Kick', level: 0.65, pan: 0.5 },
    { id: 'snare', name: 'Snare', level: 0.55, pan: 0.5 },
    { id: 'bass', name: 'Bass', level: 0.72, pan: 0.5 },
    { id: 'pad', name: 'Pad', level: 0.42, pan: 0.5 },
  ]);
  const [selectedEmotion, setSelectedEmotion] = useState<SelectedEmotion | null>(null);
  const [ghostText, setGhostText] = useState('');
  const [interactions, setInteractions] = useState<string[]>(['What are you making?']);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [lastAudioPath, setLastAudioPath] = useState<string | undefined>();
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([]);

  const brain = useMusicBrain();

  useEffect(() => {
    brain.healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const timelineBars = useMemo(() => Math.max(8, tempo > 140 ? 24 : tempo > 95 ? 16 : 12), [tempo]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => {
      playTickRef.current += 1;
      setMasterVu(() => {
        const wave = Math.sin(playTickRef.current * 0.35) * 0.4 + Math.cos(playTickRef.current * 0.18) * 0.2;
        const next = 0.35 + wave * 0.4;
        return Number(Math.min(0.94, Math.max(0.12, next)).toFixed(3));
      });
      setChannels((prev) =>
        prev.map((channel, index) => {
          const wave = Math.sin(playTickRef.current * 0.3 + index * 0.65);
          return { ...channel, level: Number((0.35 + Math.max(-0.18, Math.min(0.58, wave * 0.28))).toFixed(3)) };
        }),
      );
    }, 120);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const handleGhostGenerate = useCallback(async (localText: string) => {
    if (apiStatus === 'online') {
      try {
        await brain.setUserLyrics(localText);
        const lyrics = await brain.getUserLyrics();
        setGhostText(lyrics.lyrics ?? lyrics.generated ?? localText);
        return;
      } catch { /* fall through */ }
    }
    setGhostText(localText);
  }, [apiStatus, brain]);

  const handleInterrogatorAsk = useCallback(async (question: string) => {
    setInteractions((prev) => [...prev, `You: ${question}`]);
    if (apiStatus === 'online') {
      try {
        const response = await brain.interrogate({ message: question });
        setInteractions((prev) => [...prev, `KmiDi: ${response.reply}`]);
        return;
      } catch { /* fall through */ }
    }
    setInteractions((prev) => [
      ...prev,
      `KmiDi: Shaped around ${selectedEmotion?.base ?? 'your direction'}.`,
    ]);
  }, [apiStatus, brain, selectedEmotion]);

  const handleQuickStart = useCallback(async (template: {
    id: string; name: string; config: { bpm?: number; key?: string };
  }) => {
    setInteractions((prev) => [...prev, `Started: ${template.name} (${template.config.key ?? 'C'}, ${template.config.bpm ?? 120} BPM)`]);
    if (template.config.bpm) setTempo(template.config.bpm);
  }, []);

  const lastInteraction = interactions[interactions.length - 1] ?? '';

  // Mode transition: track previous mode for exit animation
  const [displayMode, setDisplayMode] = useState<Mode>(mode);
  const [transitioning, setTransitioning] = useState(false);

  useEffect(() => {
    if (mode === displayMode) return;
    setTransitioning(true);
    const timer = setTimeout(() => {
      setDisplayMode(mode);
      setTransitioning(false);
    }, 80); // 80ms exit, then swap to new mode (which plays 120ms enter)
    return () => clearTimeout(timer);
  }, [mode, displayMode]);

  return (
    <div className="console-shell">
      {/* ─── Upper Deck: Monitoring Strip ─── */}
      <header className="upper-deck">
        <div className="upper-deck__zone">
          <h1 className="console-wordmark">KmiDi</h1>
          <Transport
            isPlaying={isPlaying}
            isRecording={isRecording}
            tempo={tempo}
            onPlayPause={() => { setIsPlaying((v) => !v); setIsRecording(false); }}
            onStop={() => { setIsPlaying(false); playTickRef.current = 0; setMasterVu(0); }}
            onRecord={() => setIsRecording((v) => !v)}
            onTempoChange={setTempo}
          />
        </div>
        <div className="upper-deck__zone">
          <Timeline bars={timelineBars} tempo={tempo} />
        </div>
        <div className="upper-deck__zone">
          <VUMeter value={masterVu} isActive={isPlaying} />
          <Mixer
            channels={channels}
            onChannelChange={(channelId, patch) => {
              setChannels((prev) => prev.map((ch) => ch.id === channelId ? { ...ch, ...patch } : ch));
            }}
          />
        </div>
      </header>

      {/* ─── Lower Deck ─── */}
      <div className="lower-deck">
        <NavRail activeMode={mode} onModeChange={setMode} />

        <div
          id="workspace-panel"
          role="tabpanel"
          aria-labelledby={`tab-${mode}`}
          tabIndex={0}
          className={`workspace workspace--${displayMode} ${transitioning ? 'workspace--exiting' : ''}`}
        >
          {/* Mode content — next step (uses displayMode, not mode) */}
        </div>

        <SessionBar lastInteraction={lastInteraction} apiStatus={apiStatus} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/AppConsole.tsx
git commit -m "feat(shell): AppConsole main component — state, upper deck, lower deck structure"
```

---

### Task 8: Mode content rendering

**Files:**
- Modify: `src/AppConsole.tsx`

- [x] **Step 1: Replace the mode content placeholder with actual mode rendering**

Replace the `{/* Mode content — next step (uses displayMode, not mode) */}` comment inside the workspace div with the following. Note: all conditions use `displayMode` (not `mode`) so the outgoing content stays visible during the 80ms exit fade:

```tsx
          {displayMode === 'mix-detail' && (
            <section className="mode-mix-detail mode-enter" aria-label="Mix detail">
              <article className="console-panel">
                <h2 className="console-panel__title">Mixer</h2>
                <Mixer
                  channels={channels}
                  onChannelChange={(channelId, patch) => {
                    setChannels((prev) => prev.map((ch) => ch.id === channelId ? { ...ch, ...patch } : ch));
                  }}
                />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Master</h2>
                <VUMeter value={masterVu} isActive={isPlaying} />
              </article>
            </section>
          )}

          {displayMode === 'inspire' && (
            <section className="mode-inspire mode-enter" aria-label="Inspiration board">
              <article className="console-panel">
                <h2 className="console-panel__title">Mood</h2>
                <EmotionWheel onSelect={(emotion) => setSelectedEmotion(emotion)} selected={selectedEmotion} />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Ask</h2>
                <Interrogator
                  starter={selectedEmotion ? `How should this feel: ${selectedEmotion.base}` : 'Ask something'}
                  onAsk={handleInterrogatorAsk}
                />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Lyric Spark</h2>
                <GhostWriter
                  seed={selectedEmotion ? `${selectedEmotion.base} ${selectedEmotion.intensity}` : ''}
                  onGenerate={handleGhostGenerate}
                  output={ghostText}
                />
              </article>
            </section>
          )}

          {displayMode === 'create' && (
            <section className="mode-create mode-enter" aria-label="Quick creation tools">
              <article className="console-panel">
                <h2 className="console-panel__title">Starters</h2>
                <QuickStartPanel onTemplateSelect={handleQuickStart} />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Sound Palette</h2>
                <MusicCustomizer
                  selectedGenre={selectedGenre}
                  selectedEmotion={selectedMood}
                  selectedTechniques={selectedTechniques}
                  onGenreChange={(genre) => { setSelectedGenre(genre); setInteractions((p) => [...p, `Genre: ${genre}`]); }}
                  onEmotionChange={(emotion) => { setSelectedMood(emotion); setInteractions((p) => [...p, `Emotion: ${emotion}`]); }}
                  onTechniquesChange={(techs) => { setSelectedTechniques(techs); }}
                />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Lyrics</h2>
                <LyricPanel />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Spectocloud</h2>
                <SpectoCloudPanel lastGeneratedAudioPath={lastAudioPath} />
              </article>
            </section>
          )}

          {displayMode === 'compose' && (
            <section className="mode-compose mode-enter" aria-label="Intent Builder">
              <IntentBuilder />
            </section>
          )}
```

- [x] **Step 2: Verify TypeScript compiles**

Run: `npx tsc --noEmit`
Expected: No errors (AppConsole.tsx uses same component APIs as App.tsx)

- [x] **Step 3: Commit**

```bash
git add src/AppConsole.tsx
git commit -m "feat(shell): mode content rendering — inspire, create, compose, mix detail"
```

---

### Task 9: Final verification

**Files:**
- Verify: `src/AppConsole.tsx`, `src/console-shell.css`

- [ ] **Step 1: Run TypeScript check**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 2: Run Vite build check**

Run: `npx vite build --mode development 2>&1 | tail -5`
Expected: Build succeeds (AppConsole.tsx is a tree-shakeable module — it compiles even if not mounted)

- [ ] **Step 3: Verify file structure**

Run: `ls -la src/AppConsole.tsx src/console-shell.css`
Expected: Both files exist

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(shell): console shell refresh — complete review files ready for critique"
```
