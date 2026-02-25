# Pull Request: compile → main

Use this to open a PR from branch **compile** into **main** on [sburdges-eng/KmiDi](https://github.com/sburdges-eng/KmiDi).

---

## Open the PR

**Link:** [Create PR: compile → main](https://github.com/sburdges-eng/KmiDi/compare/main...compile)

---

## Suggested title

```
Merge compile into main: sync with main, CMake fixes, docs (Magenta ref, path audit, migration)
```

---

## Suggested description

```markdown
## Summary
- Merges `origin/main` into `compile` and resolves conflicts (keeps compile layout: `external/JUCE` at repo root; removes duplicate `KmiDi/external/JUCE` from main).
- CMake: local JUCE path when `external/JUCE` exists; optional plugin icons (warning instead of fatal).
- Docs: external references (Magenta, JUCE), path audit (`EXTERNAL_PATH_REFERENCES.md`), migration scripts for moving data into KmiDi-compile.

## Magenta
- [Magenta](https://github.com/magenta/magenta) is referenced in `docs/EXTERNAL_REFERENCES.md`. That repo is archived (Jan 2026); we use Magenta datasets (Groove MIDI, MAESTRO, NSynth) via `magentadata` URLs. Current Magenta work: [Magenta org](https://github.com/magenta).

## External / GitHub references (valuable data)
- **Alain Riou (aRI0U):** [deep-music-generation](https://github.com/aRI0U/deep-music-generation), [music-source-separation](https://github.com/aRI0U/music-source-separation).
- **Sony CSL Paris:** [PESTO](https://github.com/SonyCSLParis/pesto) (pitch estimation), [music2latent](https://github.com/SonyCSLParis/music2latent), [DrumGAN](https://github.com/SonyCSLParis/DrumGAN), [codicodec](https://github.com/SonyCSLParis/codicodec). See `docs/EXTERNAL_REFERENCES.md` and `docs/GITHUB_DATA_REFERENCES.md`.
- **Magenta:** Datasets (Groove MIDI, MAESTRO, NSynth) via magentadata; repo archived; Magenta org for current work.

## Testing
- Merge conflict resolution verified (tauri.conf, package.json, src/hooks, JUCE layout).
- Push to `origin/compile` succeeded.
```

---

## Note: PR to magenta/magenta

The [magenta/magenta](https://github.com/magenta/magenta) repository is **archived** (read-only as of Jan 2026). Opening a PR there is not supported. KmiDi references Magenta datasets and the project in `docs/EXTERNAL_REFERENCES.md`; for active Magenta work see the [Magenta GitHub Organization](https://github.com/magenta).
