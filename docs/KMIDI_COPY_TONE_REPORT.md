# KmiDi copy-tone report

## Overall result

**READY**

Visible interface language has been rewritten so the product speaks to musicians and composers. Copy is calm, exact, musical, and human. No visual system, APIs, or class names were changed. `npx tsc --noEmit` and `npm run build` both pass.

---

## What sounds more musical now

- **Shell and mode titles** — "Mix", "Inspire", "Create", "Compose" and "Master" read as studio modes, not dashboard sections. "Reset" and "Session" replace longer, more technical labels.
- **Intent Builder** — "Describe the piece", "Set the mood", "Build", "Try an example", and "Break a rule (optional)" support composition flow. Loading states use "Preparing…", "Building…", "Done", "Stopping…" instead of system-style status codes.
- **Emotion / mood** — "Choose a mood", "intensity", "shade", "Set a feeling to begin" keep the wheel focused and musical. Offline fallback: "KmiDi: Shaped around [mood]." instead of "Drafted approach aligned to… path."
- **Lyrics** — "Your words first", "Loaded.", "Saved. N lines.", "No lyrics yet.", and badge "You" / "Generated" / "—" feel like a writing room. Errors: "Can't reach the studio", "Couldn't save", "Couldn't read that file."
- **SpectoCloud** — "Visualize", "Turn your piece into a visual", "Ready. Build an image or animation below.", "No piece yet. Generate one in Compose, then return here.", "Building…", "Build image" / "Build animation", "Done." / "Saved." — no "Error:" prefix, no marketing tone.
- **Quick start & Sound Palette** — "Pick a starting point", "Start from this", "Start from mood / lyrics / conversation", "Shape the sound", "Your choices:" — short, invitational, no pre-configured/optimized wording.
- **Session empty state** — "Nothing yet. Ask something or set a mood." (CSS) turns emptiness into a quiet pause.
- **Modals and errors** — "Arrangement is too long. Shorten it to continue." and "Something went wrong. Try again." are brief and stabilizing.

---

## What was rewritten

- **App.tsx** — Initial session line, offline KmiDi reply, active titles (Mix/Inspire/Create/Compose), Reset button, panel titles (Master, Mood, Ask, Session, Starters), Quick Start log line.
- **IntentBuilder.tsx** — Demo button, main CTA (Generate → Build), job status strings (COMPILING… → Preparing…, GENERATING… → Building…, COMPLETE → Done, FAILED → cleared, CANCELLING… → Stopping…), engine fallback error, Song Details label and placeholder, groove/narrative placeholders, rule-break toggle and justification placeholder, Remove/Add instrument, length modal, vinyl center text kept "Set the mood".
- **EmotionWheel.tsx** — Select placeholders (pick a mood → choose a mood, how strong? → intensity, add nuance → shade), empty output ("Choose a feeling to start" → "Set a feeling to begin").
- **LyricPanel.tsx** — Title and subtitle, source badge text, all status and error messages, button labels (Load file, Save, Reload), textarea placeholder.
- **SpectoCloudPanel.tsx** — Header and description, no-audio error, simple description and notices, error card (removed "Error:" prefix), button and hint copy, output card (Done. / Saved.).
- **QuickStartPanel.tsx** — Section heading and description, selected template heading, primary button and hint, three action cards (Start from mood/lyrics/conversation and their descriptions).
- **MusicCustomizer.tsx** — Section heading and description, Genre/Mood/Techniques headers and hints, summary label ("Your Customization:" → "Your choices:").
- **GhostWriter.tsx** — Placeholder ("Pick a mood above to seed the lyrics" → "Choose a mood above"). "Spark a lyric" and "Waiting for a spark." unchanged.
- **Interrogator.tsx** — Placeholder and submit button ("Ask KmiDi" → "Ask").
- **index.css** — `.log-list:empty::before` content.

---

## What still sounds too technical

- **None** for this pass. Remaining terms (e.g. "Bars", "Reps", "Key", "BPM", "Vol", "Pan", "Transport", section names like "intro"/"verse"/"chorus") are standard studio vocabulary and were left as-is so the product still reads as a serious instrument.

---

## Final note

The product now speaks like a writing room and a studio instrument: short, grounded phrases; "piece" and "build" / "shape" where appropriate; errors that explain the issue without alarming ("Can't reach the studio", "Couldn't save"); loading that feels intentional ("Preparing…", "Building…"); and empty states that feel like a pause ("Nothing yet. Ask something or set a mood.", "Set a feeling to begin"). The voice is consistent across buttons, status lines, errors, modals, and panel headings, and stays within the existing compact layout.
