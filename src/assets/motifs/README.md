# KmiDi / iDAW — SVG Motifs

Original diagrammatic motifs for studio-grade UI. Dark-studio restraint, etched linework, technical-draft clarity. Use behind or beside working UI; never overpower text or controls.

**Palette:** Neutral strokes `rgba(255,255,255,0.06–0.12)`; restrained amber accent `rgba(201,162,39,0.25–0.35)`. No purple, neon, or glossy effects.

---

## 1. Emotion Wheel Support (`emotion-wheel-support.svg`)

- **Name:** Emotion wheel support motif  
- **Intended placement:** Behind or beside the emotion/mood wheel (IntentBuilder, mood selection).  
- **Visual rationale:** Radial calibration ring, harmonic arc segments, and cardinal/ordinal ticks suggest a calibrated control surface (like a rotary or feel selector) without copying any product. One small amber arc segment gives a single accent. Keeps the wheel area feeling instrument-like and engineered.

**Usage:** Background image or inline SVG; low opacity (e.g. 0.4–0.6) so it stays subordinate to the wheel and labels.

---

## 2. Arrangement / Timeline Support (`arrangement-timeline-support.svg`)

- **Name:** Arrangement timeline support motif  
- **Intended placement:** Along the top or bottom of the timeline/arrangement strip, or as a subtle band behind the track list.  
- **Visual rationale:** Horizontal grid, bar-length ticks, and a single waveform-inspired curve suggest time base and level without literal meters. One amber tick marks a reference point. Reads as studio notation, not decoration.

**Usage:** Repeat along timeline axis if needed, or single instance; keep opacity low so grid and ticks don’t compete with actual timeline UI.

---

## 3. Signal-Flow / Routing Support (`signal-flow-routing.svg`)

- **Name:** Signal-flow routing motif  
- **Intended placement:** Routing/send/aux panels, or near signal path controls.  
- **Visual rationale:** One input node, trunk, and three branches to output nodes read as abstract signal flow. Thin routing paths and small node circles feel like a schematic. Center branch in restrained amber suggests primary path.

**Usage:** Single motif per routing section; works as corner or sidebar accent.

---

## 4. Empty-State Etched (`empty-state-etched.svg`)

- **Name:** Empty state etched motif  
- **Intended placement:** Empty states (no tracks, no selection, no results)—centered or offset in the content area.  
- **Visual rationale:** One calm curve (level/wave abstraction), baseline, and vertical calibration ticks read as “meter ready” or “signal expected.” Minimal and calm; avoids cartoon or blob. Single small circle is a reference mark, not an icon.

**Usage:** Center in empty-state container with low opacity; combine with short copy (e.g. “Add a track” or “No items yet”).

---

## 5. Panel-Corner / Section-Divider Accent (`panel-corner-divider.svg`)

- **Name:** Panel corner section-divider motif  
- **Intended placement:** Panel corners (e.g. top-left or bottom-right of a card/section) or as a section divider between two areas.  
- **Visual rationale:** L-shaped bracket with etched inner lines and short calibration ticks reads as engraved instrument marking or chassis corner. Small amber dot at the vertex gives a single accent. Works in any corner by rotating the SVG (e.g. `transform: rotate(90deg)` for other corners).

**Usage:** One instance per corner or divider; scale to panel size; keep opacity subtle so it doesn’t dominate the panel border.

---

## Implementation notes

- **Embedding:** Use `<img src="…/emotion-wheel-support.svg" alt="" aria-hidden="true">` or inline the SVG for CSS control (e.g. `stroke: var(--border)` or `opacity`).
- **Scaling:** All SVGs use `viewBox`; scale via `width`/`height` or CSS. Keep aspect ratio for best legibility.
- **Opacity:** Apply `opacity: 0.4–0.7` on the container so motifs sit behind content.
- **Contrast:** If used on lighter surfaces, ensure stroke values remain readable (design tokens: `--border`, `--border-hover`, `--border-strong`).
