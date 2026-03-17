# KmiDi contrast depth report

## Overall result

**READY**

## What feels deeper now

- **Background and shell** — The page background is slightly darker; the main frame reads clearly as a layer above it, with a stronger border and shadow so the instrument sits in space.
- **Surface ladder** — A new “inset” tier (darkest) is used for wells: mixer strips, timeline beats, VU track, log list, lyric/ghost output, form inputs, and progress track. Panels and raised areas read one step up; overlays (hover, modals) read above that.
- **Tabs and chips** — Inactive vs hover vs active is clearer; the tab bar has a darker well so the selected tab reads as the active surface. Emotion quick-picks now have the same selected treatment (border + background) as genre and technique.
- **Buttons** — Outline, CTA, primary, and Go buttons have a visible “pressed” state so you feel the click.
- **Mixer / Timeline / Master** — Strips, beats, and master VU sit in the inset tier with consistent borders; Transport and tempo bar read as control surfaces.
- **Intent Builder** — Song-details form, structure blocks, instrument rack, and tempo bar use the same ladder; error and modal surfaces are slightly stronger so they read at a glance.
- **Borders** — Default, hover, and strong borders are a bit stronger (no new hues), so panel edges and control boundaries read without relying on accent.

## What was redrawn

- **CSS variables** — Added `--surface-inset`; deepened root and surface; raised `--surface-raised` and `--surface-overlay`; increased `--border`, `--border-hover`, `--border-strong`.
- **Shell** — `.km-frame` border and shadow; page background darkened.
- **Tabs** — `.km-toggle` uses inset; tab hover and active states; tab `:active` for press.
- **Panels** — `.panel` and `.gen-panel` use raised + light top edge; inner content (mixer, timeline, log, ghost/emotion output, lyric status/textarea) use inset.
- **Mixer, Timeline, VU, meter** — Strips, `.beat`, `.vu-track`, `.meter` use inset and tokenized borders; range track uses inset.
- **Buttons** — `:active` for outline, CTA, primary-action, gen-go; tempo-tap already had active.
- **Chips** — Genre/technique/emotion-quick selected and `:active`; template-card `:active`; preset/mode active and `:active`.
- **Intent Builder** — gen-panel, gen-tempo-bar (raised), gf-input/textarea/select, stl-editor, stl-ed-num, vinyl-disc, inst-knob/inst-name to tokens; gen-error and lyric-error backgrounds slightly deeper; modal overlay and modal border/shadow.
- **Lyric / SpectoCloud / Ghost** — Badge, status, textarea, output cards, quickstart cards, customizer summary to inset or tokens; error card background.
- **Form elements** — Global input/select/textarea and Intent-specific inputs use `--surface-inset` and border tokens.

## What still feels too flat

- None. The ladder (inset → surface → raised → overlay) and state contrast (rest, hover, active, selected) are applied across the app. If you want more separation in a specific area (e.g. structure blocks, mood nodes), we can add a second pass there without changing the palette.

## Final note

The room keeps the same warm amber accent and neutral family; nothing is brighter for the sake of it. What changed is tonal depth: darker wells for content and controls, clearer steps between background and panels, and stronger but still restrained borders. Active and selected states are easier to see at a glance, and buttons give a clear pressed state. The result is more grounded and legible for long sessions while staying calm and studio-like.
