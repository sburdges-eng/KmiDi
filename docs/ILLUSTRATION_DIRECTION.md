# KmiDi / iDAW — Background illustration direction

Production-minded art direction for restrained, musical background illustration. Supports concentration during long writing sessions without copying any specific brand or plugin.

---

## Design principles

**Feeling:** Studio-like, exact, intimate, engineered, musical, calm under pressure, immersive without spectacle.

**Role of illustration:** Add atmosphere and depth; support hierarchy. Never compete with controls, labels, or text. Think etched lacquer, printed panel markings, routing overlays, technical annotations—embedded into the material of the interface, not hero art or wallpaper.

**Palette (KmiDi tokens):**
- Base: `--surface-inset` (#0c0c0e), `--surface` (#131316), `--surface-raised` (#1b1b1f)
- Drawing: low-contrast neutrals from `--text-muted` (#6b6b69), `--text-secondary` (#9c9c9a), and rgba(255,255,255,0.06–0.12)
- Amber: `--accent` (#c9a227) and `--accent-muted` (rgba(201,162,39,0.2) or lower) only where a single focal accent is needed
- No purple; no gradients as style; no bright illustration colors; no glossy chrome or glassmorphism

**Style:** Sparse line-based drafting; abstract signal diagrams; timing/arrangement marks; lightly engraved geometry; subtle cabinet/speaker/instrument contour abstractions; waveform and resonance traces; tonal arcs and tension lines; measured negative space. Asymmetrical but balanced; imagery in edges, corners, panel backs, empty states; center of working areas kept clean.

---

## Zone 1 — Intent Builder background accents

**Visual concept:** The Intent Builder is where mood, desire, and technical intent are set. Background accents should suggest “signal before sound”—routing, calibration, and intention. Use: very faint horizontal reference lines (like ruler ticks or grid annotations); a single soft arc or tension line in one corner suggesting a phrase curve or envelope; optional minimal block/section markers that echo arrangement structure without labeling it. No icons; no literal instruments. The emotion wheel and form controls remain the only strong visual focus.

**Placement:** Confined to the outer margins and corners of the Intent Builder container (e.g. bottom-right and top-left corners, or along the left edge only). Do not cross the central area where the mood wheel and song-details form sit.

**Density:** Very low. One to three small graphic elements per view; most of the panel remains empty.

**Line weight:** 0.5–1px effective (e.g. 1px at 6–10% opacity, or 0.5px at 10–14% opacity). Strokes only; no fills except at very low opacity if a soft arc is used.

**Opacity:** 6–12% white or `--text-muted` at 30–50% opacity. Never above the level of `--border`.

**Amber:** None in this zone, or at most one tiny accent (e.g. the tip of a single arc) at `--accent-muted` or lower, to echo “intent” without pulling focus.

**Musical idea:** Calibration, routing, the moment before the first note—structure implied, not stated.

**Implementation note:** A single SVG with `position: absolute; inset: 0; pointer-events: none;` and a small `<g>` in one corner, strokes using `currentColor` or a CSS variable, opacity on the group. No animation.

---

## Zone 2 — Mixer / Timeline background support

**Visual concept:** The Mixer and Timeline panels are the control room: levels, time, and structure. Background support should feel like technical panel markings—channel divisions, time-base references, or resonance traces that sit behind the strips and the timeline. Use: very subtle vertical or diagonal lines suggesting channel boundaries or phase; faint horizontal bands suggesting bars or measure boundaries; a minimal waveform silhouette (single cycle or damped oscillation) in a corner. Nothing that reads as a literal meter or fader; everything recedes behind the real UI.

**Placement:** Mixer: along the bottom or the right edge of the mixer panel, behind the strips. Timeline: along the bottom edge or the left edge of the timeline panel, never overlapping the main timeline track or transport. Prefer one “quiet” side (e.g. right for mixer, left for timeline) so the eye is not pulled both ways.

**Density:** Low. A few lines or one small waveform trace per panel; the majority of the panel is untouched.

**Line weight:** 0.5–1px. Slightly more presence than Intent Builder is acceptable (e.g. 8–14% opacity) because these panels are already busier, but still below border strength.

**Opacity:** 8–14% white or equivalent neutral. Must stay below `--border` and below any VU or timeline fill.

**Amber:** Optional: a single resonance peak or timeline “now” marker in amber at `--accent-muted` (e.g. 15–20% opacity) to tie the zone to the accent system. Use sparingly—one instance per panel at most.

**Musical idea:** Signal path and time—channels and bars as a quiet grid, the room’s reference marks.

**Implementation note:** SVG or CSS linear gradients (very subtle, 1–2 stops) as panel background layer. If SVG, use a `<pattern>` for repeating lines so the asset stays small. No animation.

---

## Zone 3 — Empty-state backgrounds

**Visual concept:** Empty states (e.g. Lyric Spark “Waiting for a spark.”, Spectocloud “Generate a piece first…”, no lyrics yet) should feel like a held breath—inviting but not loud. Use: a single abstract shape—e.g. a soft arc (tonal curve), a minimal speaker or diaphragm contour, or a few widely spaced resonance lines—that suggests “signal possible here.” No icons; no characters; no decorative blobs. The copy remains the primary message; the graphic is atmosphere.

**Placement:** Centered or slightly off-center in the empty-state container, but scaled so it sits largely in the lower two-thirds or to one side, leaving the main copy area clear. Do not place behind the text block.

**Density:** Very low. One motif per empty state; plenty of negative space.

**Line weight:** 0.5–1px; can use a slightly softer fill for a single arc (e.g. 3–6% opacity) to avoid a hard edge.

**Opacity:** 5–10% for lines; if a soft fill is used for an arc, 3–6%. Must not distract from the empty-state message.

**Amber:** Optional: a single accent on the motif (e.g. the crest of an arc, or one resonance line) at `--accent-muted` (10–18% opacity) to suggest “ready to receive.” One accent per empty state.

**Musical idea:** Anticipation—the room is set, waiting for input.

**Implementation note:** Inline SVG or a small SVG asset per empty-state component; one `<path>` or a few `<line>`/`<path>` elements. Class or data-attribute for theming (e.g. `data-zone="empty-state"`). No animation, or at most a very slow opacity ease-in on mount (e.g. 0 → 1 over 0.6s) so it doesn’t compete with copy.

---

## Zone 4 — Modal / overlay backing treatment

**Visual concept:** Modals and overlays (e.g. song-length warning, confirmations) need a clear focus on the content. The backing should gently separate the overlay from the rest of the app without adding visual noise. Use: a minimal frame—e.g. corner brackets, a very light border treatment, or a single soft arc at the edge of the overlay—that reads as “focused surface” rather than decoration. Alternatively, a barely-there grid or crosshair at the corners to suggest alignment/measurement. No imagery in the center.

**Placement:** At the outer edge of the modal (inside the overlay, along the border or in the corners). Never in the center; never behind the primary CTA or the main copy.

**Density:** Minimal. Corner marks only, or one short arc per side at most.

**Line weight:** 0.5–1px.

**Opacity:** 6–10% white or neutral. Lighter than the modal’s border so the border remains the primary edge.

**Amber:** Optional: one corner or one short segment in `--accent-muted` (low opacity) to tie the overlay to the accent system—e.g. the “active” corner or the start of an arc. Single instance per overlay.

**Musical idea:** The overlay as a “take” or a measured moment—precise, contained.

**Implementation note:** Pseudo-elements or a small SVG fragment in the modal component; strokes only. Consider `mix-blend-mode` (e.g. soft-light) at very low opacity if it helps the marks feel etched rather than floating. No animation.

---

## Zone 5 — Side-panel / inspector-area atmospheric support

**Visual concept:** Side panels (e.g. Activity Feed, Creative Assistant, song-details column) are secondary reading and control areas. Atmospheric support should suggest “notes at the margin”—sidebar annotations, subtle routing, or a single contour (cabinet, baffle, or instrument silhouette) reduced to a few lines. The goal is depth and continuity with the rest of the interface, not a second focal point.

**Placement:** Along the outer edge of the side panel (opposite the content)—e.g. right edge if the panel is on the left, or along the bottom. Alternatively, a narrow strip along the full height of the panel’s “back” edge. Keep all content and controls clear of the graphic.

**Density:** Low. One contour or one set of 2–4 annotation lines per panel; the rest negative space.

**Line weight:** 0.5–1px.

**Opacity:** 6–12% white or neutral. Must stay below `--border` and below any active or focus states in the panel.

**Amber:** Optional: one line or the endpoint of a contour in `--accent-muted` (10–15% opacity) to echo the main accent—e.g. the “live” end of a routing line. One accent per panel.

**Musical idea:** The room’s margin—notes, routing, a quiet reference at the edge of attention.

**Implementation note:** Background layer on the side-panel container (e.g. `::before` or a dedicated div with SVG). SVG can be a single path (contour) or a small set of lines; use CSS variables for stroke so it respects the palette. No animation.

---

## Summary table

| Zone              | Density   | Line (effective) | Opacity (drawing) | Amber                          |
|-------------------|-----------|-------------------|--------------------|---------------------------------|
| Intent Builder    | Very low  | 0.5–1px           | 6–12%              | None or one tiny accent         |
| Mixer / Timeline  | Low       | 0.5–1px           | 8–14%              | Optional, one per panel         |
| Empty states      | Very low  | 0.5–1px (+ soft fill) | 5–10% (lines), 3–6% (fill) | Optional, one per state   |
| Modal / overlay   | Minimal   | 0.5–1px           | 6–10%              | Optional, one per overlay       |
| Side panel        | Low       | 0.5–1px           | 6–12%              | Optional, one per panel         |

---

## Don’ts (recap)

- Do not imitate a specific audio plugin or DAW.
- No fake hardware renders, glossy chrome, or glassmorphism.
- No generic SaaS abstract blobs or futuristic sci-fi clichés.
- No purple; no gradients as a stylistic crutch; no bright illustration colors.
- Illustrations are not hero art, not wallpaper, not concept-art spectacle—they support the interface and the feeling of a studio, exact and calm.

---

*Art direction brief for implementation. Tokens and class names align with `src/index.css` and `docs/FRONTEND_DESIGN_BRIEF.md`.*
