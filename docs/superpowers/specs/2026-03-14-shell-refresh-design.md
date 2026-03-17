# KmiDi Shell Refresh — "The Console"

Split-deck layout with persistent monitoring strip, vertical nav rail, cinematic mode transitions, and elevated visual identity.

---

## 1. Motivation

- Navigation feels flat — horizontal tabs don't convey hierarchy or workflow
- Visual identity is weak — needs more personality, atmosphere, memorable character
- Doesn't feel like a DAW — needs professional music software presence
- Inspiration: not a traditional DAW — closer to Teenage Engineering / Roli / Endlesss
- Information architecture fully rethought
- Serves three interaction postures: lean-back explorer, hands-on builder, conversational collaborator

---

## 2. Shell Architecture

### 2.1 Two-Deck Split

```
┌─────────────────────────────────────────────────────┐
│  UPPER DECK — Monitoring Strip (160px fixed)        │
│  ┌──────────┬───────────────────────┬──────────────┐│
│  │Transport │      Timeline         │  VU + Mixer  ││
│  │▶ ■ ● 120│  ▌▌▌▌ ▌▌▌▌ ▌▌▌▌ ▌▌▌▌ │  ████  ▕▕▕▕ ││
│  └──────────┴───────────────────────┴──────────────┘│
├─┬───────────────────────────────────────────────────┤
│N│  LOWER DECK — Creative Workspace (remaining)      │
│A│                                                   │
│V│   Active mode content fills this space             │
│ │                                                   │
│R│                                                   │
│A│───────────────────────────────────────────────────│
│I│  SESSION BAR — last interaction + status      ◉ API│
│L│                                                   │
└─┴───────────────────────────────────────────────────┘
```

**Upper Deck (160px fixed height):**
- Always visible — the heartbeat of KmiDi
- Surface: `--surface-inset` (deepest dark, recessed behind glass)
- Bottom edge: 1px `--border-strong` + hairline inner highlight (`0 1px 0 rgba(255,255,255,0.03) inset`)
- Three zones separated by subtle vertical dividers (`--border`):

| Zone | Width | Content |
|------|-------|---------|
| Transport | ~200px fixed | Play/Pause, Stop, Record (icon-only), editable tempo (amber on interaction) |
| Timeline | flex-grow | Condensed horizontal bar visualization — thin vertical bar lines + amber playhead |
| Meters | ~180px fixed | Compact vertical VU bar + 4 horizontal mini-mixer level bars (K, S, B, P labels) |

**Lower Deck (remaining viewport):**
- Surface: `--surface` base
- Contains: nav rail (left) + active mode content (right) + session bar (bottom)

**Session Bar (bottom of lower deck, ~32px):**
- Thin strip showing last session interaction text + API status indicator (●)
- Surface: flush with lower deck
- Text: `--text-muted`, 12px

### 2.2 KmiDi Wordmark

Top-left corner of upper deck. Fraunces, 15px, `--text-primary`. Small and confident — not a billboard.

---

## 3. Navigation

### 3.1 Nav Rail

Vertical strip on the far-left edge of the lower deck. 48px wide. No background — flush with the lower deck surface.

**Four modes stacked vertically:**

| Position | Icon | Hover Label | Maps To |
|----------|------|-------------|---------|
| 1 | ◎ | Mix Detail | Expanded mixer (full channel strips) |
| 2 | ◐ | Inspire | EmotionWheel + Interrogator + GhostWriter |
| 3 | ✦ | Create | QuickStart + MusicCustomizer + LyricPanel + Spectocloud |
| 4 | ▧ | Compose | IntentBuilder (full workspace) |

**Icon styling:**
- Size: 20px
- Default: `--text-muted`
- Hover: `--text-secondary`, 120ms ease
- Active: `--text-primary`

**Active LED indicator:**
- 4px circle, `--accent` color
- Positioned on left edge of icon
- Glow: `box-shadow: 0 0 6px var(--accent-muted)`
- On activation: single subtle throb pulse, then steady

**Hover labels:** Tooltip appearing to the right of the icon on hover, 120ms fade-in.

**Keyboard:** Arrow up/down navigates. Enter activates. Focus ring uses `--accent-focus`.

---

## 4. Mode Layouts

All mode content fills the lower deck workspace (full width minus 48px nav rail, full height minus 32px session bar).

### 4.1 Inspire

```
┌────────────────────┬────────────────────────────┐
│                    │                            │
│   Emotion Wheel    │      Interrogator          │
│   (40% width)      │      (chat/dialogue)       │
│                    │                            │
│                    ├────────────────────────────┤
│                    │                            │
│                    │      Lyric Spark           │
│                    │      (GhostWriter)         │
└────────────────────┴────────────────────────────┘
```

- Left column: 40% — EmotionWheel, full height
- Right column: 60% — Interrogator (top, dominant) + GhostWriter (bottom)
- Conversational AI gets the most real estate

### 4.2 Create

```
┌──────────────────────────────────────────────────┐
│              Starters (QuickStartPanel) ~80px     │
├────────────────────────┬─────────────────────────┤
│                        │                         │
│    Sound Palette       │       Lyrics            │
│    (MusicCustomizer)   │       (LyricPanel)      │
│                        │                         │
├────────────────────────┴─────────────────────────┤
│              Spectocloud (visualization)          │
└──────────────────────────────────────────────────┘
```

- Starters: full width, compact (~80px) — a launcher, not a destination
- Two-column middle: Sound Palette (left) + Lyrics (right)
- Spectocloud: full width bottom, ambient visualization

### 4.3 Compose

```
┌──────────────────────────────────────────────────┐
│                                                  │
│              IntentBuilder                       │
│              (full width, full height)            │
│                                                  │
└──────────────────────────────────────────────────┘
```

IntentBuilder gets the entire workspace. No competing panels.

### 4.4 Mix Detail

```
┌──────────────────────────────────────────────────┐
│                                                  │
│              Expanded Mixer                      │
│              (full channel strips, faders,        │
│               pan, labels)                       │
│                                                  │
└──────────────────────────────────────────────────┘
```

Initially renders the existing Mixer + VUMeter components at full workspace size. A dedicated expanded mixer component with tall faders and per-channel detail is deferred to a future pass.

---

## 5. Visual Identity

### 5.1 Depth Layers

Three distinct layers creating spatial hierarchy:

| Layer | Surface Token | Role |
|-------|---------------|------|
| Recessed | `--surface-inset` | Upper deck (monitoring) — embedded hardware readout |
| Workspace | `--surface` | Lower deck background |
| Raised | `--surface-raised` | Panels within lower deck |

### 5.2 Noise Grain Overlay

Subtle CSS noise texture across the entire shell:
- Tiny repeating SVG or base64 noise pattern
- ~3% opacity
- Static (no animation)
- Eliminates "flat digital void" — gives materiality like brushed metal or matte rubber

### 5.3 Per-Mode Atmospheric Glow

Subtle radial gradient in the lower deck background, behind content:

| Mode | Glow | Position | Opacity |
|------|------|----------|---------|
| Inspire | Warm amber | Bottom-left | ~4% |
| Create | Cool teal | Center | ~4% |
| Compose | Neutral warm white | Top-center | ~3% |
| Mix Detail | None | — | — (clean, clinical) |

Glow crossfades during mode transitions (200ms).

### 5.4 Typography

| Element | Font | Size | Weight | Style |
|---------|------|------|--------|-------|
| Panel titles | IBM Plex Sans | 11px | 400 | Uppercase, `letter-spacing: 0.08em`, `--text-muted` |
| Section headers | IBM Plex Sans | 13px | 600 | `--text-secondary` |
| KmiDi wordmark | Fraunces | 15px | 700 | `--text-primary`, optical kerning |
| Body / controls | IBM Plex Sans | 14px | 400 | `--text-primary` |

Panel titles whisper — silk-screened onto the surface like hardware labels.

### 5.5 Panel Borders

Double-layer etched treatment:
- Inner: 1px `--border`
- Outer: `box-shadow: 0 1px 3px rgba(0,0,0,0.3)`

Creates "etched into the surface" look, not "floating card."

### 5.6 Nav Rail LED

- 4px circle, `--accent`
- `box-shadow: 0 0 6px var(--accent-muted)` glow
- Single throb on activation, then steady

---

## 6. Motion

**Hard constraint: all transitions ≤ 200ms total.**

### 6.1 Mode Transitions

1. Outgoing panels: `opacity: 1 → 0`, 80ms ease-out, all simultaneously
2. No pause
3. Incoming panels: `opacity: 0 → 1`, `translateY: 8px → 0`, 120ms ease-out, all simultaneously (no stagger)
4. Background glow crossfade: 200ms, concurrent

### 6.2 Micro-Interactions

| Element | Effect | Duration |
|---------|--------|----------|
| Nav icon hover | Brightness shift (`--text-muted` → `--text-secondary`) | 120ms |
| Nav LED activation | Single throb pulse, then steady | 200ms |
| Upper deck playhead | Smooth `translateX` | Per tick |
| VU meters | CSS transition on height | 80ms |
| Panel hover | Border `--border` → `--border-hover` | 150ms |

### 6.3 Reduced Motion

`prefers-reduced-motion`:
- All transitions become instant opacity fades (100ms, no translate)
- LED pulse disabled
- Background glow changes are instant

---

## 7. Accessibility

- Nav rail: `role="tablist"` with `aria-orientation="vertical"`, `aria-label="Studio mode"`
- Each nav icon: `role="tab"`, `aria-selected`, `aria-controls`, stable `id`
- Lower deck workspace: `role="tabpanel"`, `aria-labelledby` active tab
- Focus ring: `--accent-focus` on `:focus-visible`
- Arrow key navigation on nav rail
- `prefers-reduced-motion` fully supported
- Session bar: `aria-live="polite"` for interaction updates

---

## 8. Existing Design Tokens (Preserved)

All existing CSS custom properties from the design system are preserved:
- Amber accent system (`--accent`, `--accent-hover`, `--accent-muted`, `--accent-focus`)
- Surface ladder (`--surface-inset`, `--surface`, `--surface-raised`, `--surface-overlay`)
- Border system (`--border`, `--border-hover`, `--border-strong`)
- Text hierarchy (`--text-primary`, `--text-secondary`, `--text-muted`)
- Semantic colors (`--danger`, `--success`)
- Spacing scale (`--space-1` through `--space-5`)
- Radius tokens (`--radius`, `--radius-lg`)
- Font families (`--font-display`, `--font-ui`)

---

## 9. Component API Contract

**No component API changes.** Existing component props and callbacks are unchanged:
- Transport, Mixer, Timeline, VUMeter — same props
- EmotionWheel, GhostWriter, Interrogator — same props
- QuickStartPanel, MusicCustomizer, LyricPanel, SpectoCloudPanel — same props
- IntentBuilder — same props

The shell restructure affects only:
- `App.tsx` — layout, navigation, state for active mode
- `index.css` — new shell classes, panel classes, transitions, grain overlay, atmospheric glows
- Potentially new wrapper components for Upper Deck zones (condensed Transport, condensed Mixer/VU)

### 9.1 New Components (Thin Wrappers)

| Component | Purpose |
|-----------|---------|
| `MonitoringStrip.tsx` | Upper deck container — renders existing Transport, Timeline, VU+Mixer constrained via CSS to 160px strip height. No new condensed component variants needed; existing components adapt within the height constraint. |
| `NavRail.tsx` | Vertical nav rail with icon buttons, LED indicators, tooltips |
| `SessionBar.tsx` | Bottom bar with last interaction text + API status |

These are layout/shell components only. They compose existing components, not replace them.

---

## 10. Scope & Constraints

**In scope:**
- App.tsx shell restructure (two-deck, nav rail, session bar)
- Remove `max-width: 1120px` constraint — shell fills viewport (Tauri window)
- index.css new shell/layout/transition styles
- New thin wrapper components (MonitoringStrip, NavRail, SessionBar)
- Transport, Timeline, VUMeter, Mixer always rendered (lifted out of conditional mode blocks into upper deck)
- Noise grain overlay
- Per-mode atmospheric glow
- Typography refinements for panel titles
- Nav rail icons as inline SVGs (not Unicode glyphs, for cross-platform consistency)
- Mode transition animations (≤200ms)
- Accessibility (ARIA, keyboard, reduced motion)

**Out of scope:**
- Individual component internals (EmotionWheel, IntentBuilder, etc.)
- New features or API changes
- Responsive/mobile layout (desktop-first, Tauri app)
- Light mode
- Audio engine integration
- Tauri window chrome

---

## 11. Deliverable

A single standalone `.tsx` file containing the complete refreshed shell (App + MonitoringStrip + NavRail + SessionBar inlined) with accompanying CSS, for the user to critique personally before any integration into the main codebase.
