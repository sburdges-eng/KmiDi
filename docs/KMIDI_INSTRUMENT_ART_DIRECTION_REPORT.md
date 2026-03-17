# KmiDi instrument-art-direction report

## Overall result

The UI was refined component-area by area to feel more like an **original premium music instrument environment**: same KmiDi palette and warm amber accent, deeper darks, clearer surface separation, and more confident active/selected states. No layout or product was copied; no gradients, purple, rainbow, or glow-heavy effects were added. Changes are token- and CSS-only; existing APIs and class names were preserved, and `npx tsc --noEmit` and `npm run build` both pass.

---

## What feels more musical now

- **Mode selection (tabs)** — Tabs read as instrument mode selection: stronger inactive (muted) vs active (amber, bold) contrast; subtle inset on the toggle strip so it feels like a physical control surface.
- **Titlebar** — Clear material strip under the header with a defined border and slightly bolder subtitle so the current mode (Mix / Inspire / Create / Compose) reads as the main readout.
- **Transport and tempo** — Play/Stop/Record and CTA buttons have clearer resting/hover/active states and a bit more weight; tempo label uses tabular numerals and a tighter rhythm.
- **Mixer** — Strips have a light inset shadow; labels are uppercase with letter-spacing; meters use a restrained green–amber gradient and faster response so levels feel precise.
- **Timeline** — Ruler beats are tighter (2px gap); major beats use amber-muted border and background for bar-one emphasis; meta line (bars, duration) is slightly bolder and tabular.
- **Master VU** — Shorter track height, stronger percentage readout (larger, bold), and semantic gradient (green–amber–red) without visual noise.
- **Intent Builder** — Song Details labels are uppercase with consistent spacing; fields have a bit more padding and focus feedback; Mood (vinyl) and Arrangement panels have clearer panel depth and hierarchy.
- **Emotion (Side B)** — Feeling/Strength/Color selects and the combined output readout have clearer label hierarchy and a contained output block so the wheel feels authored and intentional.
- **Vinyl mood ring (Intent)** — Disc has a light inner shadow; center label is bolder; mood nodes have clearer default/hover/active contrast and a light border shadow so the ring feels measured.

---

## What was deepened

- **Surfaces** — Root and surface tokens are darker (`#050506` page, `#0a0a0c` inset, `#0f0f12` base, `#16161a` raised, `#1e1e23` overlay) for more depth and less flatness.
- **Borders** — Slightly lower default opacity and clearer steps (8% / 14% / 22%) so panels and controls separate without feeling harsh.
- **Text muted** — Muted tier darkened (`#5e5e5c`) for better hierarchy and less glare.
- **Frame** — Main shell has a soft inner highlight, stronger outer shadow, and a dark outer ring so the instrument feels grounded and machined rather than flat.
- **Panels** — Raised panels use a light top highlight and a soft drop shadow for depth.
- **Tempo bar** — Slightly larger padding and a defined border/shadow so the BPM + TAP + slider read as one instrument-grade block.
- **Progress** — Generation progress bar is a bit taller and uses an ease-out transition so waiting feels intentional and calm rather than spinner-led.

---

## What imagery was introduced

- **None** — No new icons, illustrations, or decorative imagery were added. The only “imagery” is existing diagrammatic support: the vinyl groove (repeating radial gradient) and the mood ring nodes. Ring depth and node state contrast were improved so the wheel feels more intentional, not more decorative.

---

## What still feels too software-like

- **Activity Feed** — The log list is still a simple scroll list; a future pass could give it a calmer, more “session log” feel (e.g. muted timestamps, clearer line separation) without changing behavior.
- **Quick Create / Sound Palette** — Template and genre/mood chips are improved by shared tokens but could be tuned further for a more “hardware preset” feel (e.g. card depth, selected state).
- **Structure timeline (Intent)** — Bar blocks and the + add control are clearer, but the strip could benefit from optional subtle bar-line ticks or a clearer “grid” readout if the product evolves.
- **Modals** — Length-warning and similar modals are clean and on-palette but still feel like standard dialogs; copy could be tightened to a single short, musical line where appropriate.

---

## Final note

The interface is now **darker, more separated, and more stateful** while staying within the KmiDi family: warm amber accent, no extra color accents, and no novelty or dashboard feel. The highest-impact refinements were applied first (tokens, shell, tabs, panels, transport, mixer, timeline, VU, Intent panels, emotion/vinyl, error and progress). All changes are in `src/index.css`; no new dependencies and no breaking changes to component APIs or class names. Validation: `npx tsc --noEmit` and `npm run build` both succeed.
