# KmiDi / iDAW — SVG Motifs

Original diagrammatic motifs for studio-grade UI. Dark-studio restraint, etched linework, technical-draft clarity. Use behind or beside working UI; never overpower text or controls.

**Note:** The **emotion wheel support** motif is the only KmiDi-project–specific SVG in this folder. Panel-corner and signal-flow motifs are Subwooder project native and live in that repo.

**Palette:** Neutral strokes `rgba(255,255,255,0.06–0.12)`; restrained amber accent `rgba(201,162,39,0.25–0.35)`. No purple, neon, or glossy effects.

---

## 1. Emotion Wheel Support (`emotion-wheel-support.svg`) — KmiDi

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

## 3. Empty-State Etched (`empty-state-etched.svg`)

- **Name:** Empty state etched motif  
- **Intended placement:** Empty states (no tracks, no selection, no results)—centered or offset in the content area.  
- **Visual rationale:** One calm curve (level/wave abstraction), baseline, and vertical calibration ticks read as “meter ready” or “signal expected.” Minimal and calm; avoids cartoon or blob. Single small circle is a reference mark, not an icon.

**Usage:** Center in empty-state container with low opacity; combine with short copy (e.g. “Add a track” or “No items yet”).

---

## Implementation notes

- **Embedding:** Use `<img src="…/emotion-wheel-support.svg" alt="" aria-hidden="true">` or inline the SVG for CSS control (e.g. `stroke: var(--border)` or `opacity`).
- **Scaling:** All SVGs use `viewBox`; scale via `width`/`height` or CSS. Keep aspect ratio for best legibility.
- **Opacity:** Apply `opacity: 0.4–0.7` on the container so motifs sit behind content.
- **Contrast:** If used on lighter surfaces, ensure stroke values remain readable (design tokens: `--border`, `--border-hover`, `--border-strong`).
