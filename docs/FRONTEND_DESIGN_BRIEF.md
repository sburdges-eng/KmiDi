# KmiDi / iDAW — Frontend design brief

Structured brief for downstream implementation and review. Product: AI-powered music creation (intent builder, emotion wheel, mix console, song builder). Intended feel: studio-style, typography-led, minimal, focused tool with clear hierarchy, restrained color, and deliberate spacing.

---

## 1. Design system

### 1.1 Typography

| Role | Font | Use |
|------|------|-----|
| Display | Fraunces | App title (KmiDi), emphasis |
| UI / body | IBM Plex Sans | Controls, labels, body copy |

- Inter and generic system stacks are removed from the global stack.
- Fonts loaded via Google Fonts (preconnect + stylesheet) in `index.html`.

### 1.2 Color

| Token / use | Value / rule |
|-------------|--------------|
| Background | `#0c0c0d` |
| Surfaces | `#141416`, `#1a1a1d`, `#222226` (neutral, non–blue-tinted) |
| Accent | Warm amber `#c9a227`; hover `#d4b03d` |
| Accent (muted / focus) | Muted and focus ring values derived from amber (e.g. `rgba(201, 162, 39, 0.2)`, `rgba(201, 162, 39, 0.35)`) |
| Borders | Light; 7–18% white opacity |
| Text primary | `#f2f2f1` |
| Text secondary | `#9c9c9a` |
| Text muted | `#6b6b69` |
| Semantic | Danger and success use project-defined values (e.g. `--danger`, `--success`) for consistency |

- Single warm accent only (amber). No purple/indigo or competing primary accents.
- No generic gradients; palette must remain readable on dark backgrounds.

### 1.3 Rhythm and layout

| Token | Value |
|-------|--------|
| Base unit | 4px |
| Spacing | `--space-1` through `--space-5` (4px, 8px, 12px, 16px, 24px) |
| Radius | `--radius`: 6px; `--radius-lg`: 8px |
| Panels | Use `--radius-lg` and design-system surfaces |

### 1.4 Tabs

- Pattern: Segmented control (not pill-with-fill).
- Chrome: Border + raised surface; active tab = solid amber background.
- Focus: `:focus-visible` rings use `--accent-focus` (amber, reduced opacity).
- No pill-style fill for inactive state.

### 1.5 Forms and controls

- Input / select / textarea focus: Amber border + subtle box-shadow (design-system focus token).
- Range thumbs: Use accent (amber) from design system.
- VU / meter fills: Use accent / green / danger palette (no purple/indigo).
- Buttons: Visible `:focus-visible` state using accent-focus.

### 1.6 Motion

- `prefers-reduced-motion` supported: animations and transitions minimized when user prefers reduced motion.

### 1.7 Emotion wheel layout

- Default: 3-column grid.
- Small screens: Stack to single column (responsive breakpoint).

---

## 2. Shell semantics (App.tsx)

| Element | Requirement |
|---------|-------------|
| Tab list | Uses `role="tablist"`; `aria-label="Studio mode"`. |
| Tabs | Each tab has `role="tab"`, `aria-selected`, `aria-controls="main-content"`, and stable `id` (e.g. `tab-<mode>`). |
| Tab panel | Main content container has `id="main-content"`, `role="tabpanel"`, `aria-labelledby` pointing to active tab id, and `tabIndex={0}` where appropriate. |
| Reset playback | Button has `aria-label="Reset playback"`. |
| Navigation | Tab list wrapped in `<nav>` (or equivalent semantic container). |
| Titlebar | `aria-hidden="true"` applied where the titlebar is decorative or redundant for assistive tech. |

- Semantics are additive; component behavior and APIs are unchanged.

---

## 3. Palette constraints (IntentBuilder)

### 3.1 Section colors (SECTION_COLORS)

- Replaced with a limited, non-rainbow palette.
- Allowed hues / roles: slate, green, amber, rust, plum, teal, red (or equivalent distinct, muted tones).
- Criteria: Readable on dark background; no generic gradients; distinct enough for section identity.

### 3.2 Moods (MOODS)

- Updated to muted, distinct hues.
- Palette should include warm grays, teals, amber, and red where thematically appropriate.
- Intended to feel consistent with the rest of the UI and the single-accent system.

---

## 4. Touched files

| File | Changes |
|------|--------|
| `index.html` | Document title set to “KmiDi — iDAW”; preconnect and Google Fonts link for Fraunces and IBM Plex Sans. |
| `src/index.css` | New design tokens; typography; shell; tabs; buttons; panels; forms; QuickStart, MusicCustomizer, and Intent/Generate workspace styles; emotion wheel grid; responsive and reduced-motion behavior. |
| `src/App.tsx` | Tab semantics (role, aria-selected, aria-controls, ids); main panel id/role/aria-labelledby; Reset playback aria-label; tabs wrapped in `<nav>`; titlebar `aria-hidden` where appropriate. |
| `src/components/IntentBuilder.tsx` | `SECTION_COLORS` and `MOODS` updated to the constrained system palette (no purple/indigo; limited, readable-on-dark palette). |

---

## 5. Known validated outcomes

- **TypeScript:** `npx tsc --noEmit` passes.
- **Dependencies:** No new dependencies added.
- **APIs:** Existing component APIs preserved.
- **Class names:** Existing class names preserved; styling changes are token- and value-level.
- **Design:** Single warm accent (amber); typography-led hierarchy; consistent surfaces and borders; accessibility semantics and visible focus applied to shell and controls; reduced motion supported.
- **Compatibility:** Aligns with current codebase structure; no breaking changes to component contracts.

---

## 6. Out of scope for this brief

- Implementation steps or code-level proposals.
- Redesign of flows or new features.
- Mixer/Timeline visual alignment (not specified as validated in this brief).
- Empty/loading/error state components (not specified as part of this pass).
- Tab keyboard behavior beyond ARIA (e.g. arrow keys) — not asserted as implemented.
- Font delivery (e.g. self-hosted vs Google Fonts) or brand/legal constraints.
- Light/dark mode switching.
- Tauri/desktop shell visuals beyond the described shell and controls.

---

*Brief generated for implementation/review. No speculative additions.*
