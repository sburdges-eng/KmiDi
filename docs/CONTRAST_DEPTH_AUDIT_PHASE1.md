# KmiDi contrast depth — Phase 1 tone audit

## Most important contrast problems

1. **Surface ladder too compressed** — Root `#0c0c0d`, `--surface` `#141416`, `--surface-raised` `#1a1a1d`, `--surface-overlay` `#222226` are only ~2–3% apart in luminance. Panels, gen-panel, mixer-strip, and content areas blend; hierarchy is hard to read at a glance.

2. **Borders too weak** — `--border` at 0.07 is barely visible; `--border-hover` 0.14 and `--border-strong` 0.18 are still subtle. Panel and card edges don’t separate clearly from the shell.

3. **Tab and chip states understated** — Tab hover uses `rgba(255,255,255,0.04)`; inactive vs hover vs active is too close. Emotion-quick selected state is only font-weight with no background/border change; genre/technique selected use accent-muted but could read more clearly.

4. **No explicit pressed/active state** — Buttons have hover and focus only; no `:active` depth, so interaction feels flat.

5. **Studio areas don’t sit in a “well”** — Mixer strips, timeline ruler, master VU, and log list use the same surface-raised/overlay as the panel chrome. There’s no darker floor for Transport/Mixer/Timeline/Master content, so they don’t feel physically nested.

6. **gen-tempo-bar uses `--surface`** — Same as the frame floor; the tempo bar doesn’t read as a distinct control bar.

7. **Scattered hardcoded darks** — Meter `rgba(2,6,12,0.7)`, VU track `#0c1422`, form inputs `rgba(11,18,33,0.85)` / `rgba(8,12,22,0.8)`, log-list `rgba(0,0,0,0.2)` are off the token ladder and inconsistent.

8. **Modal overlay at 0.55** — Could be slightly deeper so the modal reads more clearly above the app.

9. **Empty/loading/output areas** — Ghost output, lyric status, log list, emotion output all use surface-overlay or ad-hoc darks; no shared “inset” tier for content wells.

10. **Amber doing too much** — Selected and active states rely heavily on accent because neutrals don’t carry enough structure; strengthening the neutral ladder will let accent stay for true actions and focus.

---

## Areas that need deeper tonal separation

| Area | Need |
|------|------|
| **Global** | New `--surface-inset` (darker than surface) for wells; widen steps between surface / raised / overlay; strengthen border tokens. |
| **Shell** | km-frame vs page background: slightly deeper root or clearer surface so frame reads as one level up. |
| **Tabs** | Inactive → hover → active: clearer background steps; tab bar border strong enough to read as segmented control. |
| **Panels** | .panel / .gen-panel: use raised; inner content (mixer, timeline, log, ghost output) use inset so they sit “inside” the panel. |
| **Mixer / Timeline / Master** | Strips, ruler beats, VU track, meter track: use inset + slightly stronger border so they read as a clear tier below panel. |
| **Buttons** | outline / cta / primary: add `:active` (pressed) state with darker background. |
| **Chips** | genre, technique, emotion-quick, template-card, preset/mode: selected state with border + background step; emotion-quick selected to match (border + background). |
| **Intent Builder** | gen-panel, gen-tempo-bar (use raised), stl-block, gf-input, stl-editor: align to surface ladder; modal overlay deeper. |
| **Lyric / SpectoCloud / Ghost** | Output areas, badges, log-list: use inset where they are content wells. |
| **Errors / semantic** | gen-error, lyric-error: keep danger color; slightly deeper background for separation. |

---

## Order of edits by visual impact

1. **CSS variables** — Add `--surface-inset`; deepen root/surface and raise raised/overlay; increase `--border`, `--border-hover`, `--border-strong`. Single change, system-wide impact.

2. **Shell and main panels** — km-frame, .panel, .gen-panel use new ladder; give inner content (mixer-strip, timeline, log-list, ghost-output, emotion-output, lyric-status) `--surface-inset` and consistent borders.

3. **Tabs and chips** — Tab hover/active; genre/technique/emotion-quick selected (including emotion-quick background+border); template-card selected; preset/mode active. Makes mode and choice obvious.

4. **Buttons** — outline, cta, primary-action-btn, gen-go-btn: add `:active` state. Improves tactile feedback.

5. **Mixer, Timeline, VU, meter** — Strips, .beat, vu-track, .meter use inset and stronger border; tempo bar use raised so it reads as control bar.

6. **Intent Builder** — gen-panel, gen-tempo-bar, stl-editor, gf-input, stl-block, modal overlay. Clearer section grouping and form depth.

7. **Lyric / SpectoCloud / Ghost** — Output cards, badges, log-list to inset; error blocks slightly deeper.

8. **Range and form controls** — Tokenize input/select/textarea and range track to use design tokens (e.g. inset for track, accent for thumb unchanged).

---

*Next: Phase 2 (surface depth), Phase 3 (state contrast), Phase 4 (studio depth) implemented in `src/index.css`.*
