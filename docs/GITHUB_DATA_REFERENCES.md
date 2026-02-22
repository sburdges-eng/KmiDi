# Valuable Data & Repos from GitHub (PR reference)

Curated list of GitHub repos and data sources that are valuable for KmiDi (music/MIDI/ML). Use this for PR descriptions, integration planning, or citations.

---

## Alain Riou (aRI0U)

- **Profile:** [github.com/aRI0U](https://github.com/aRI0U)  
- **Context:** Assistant Researcher at Sony CSL Paris; music + ML (self-supervised learning, pitch, source separation).

| Repo | License | Stars | Value |
|------|---------|-------|--------|
| [deep-music-generation](https://github.com/aRI0U/deep-music-generation) | (check repo) | 1 | Deep learning music generation (encoder, model, tonality). |
| [music-source-separation](https://github.com/aRI0U/music-source-separation) | (check repo) | 8 | DNN source separation with phase features; MUSDB18; PyTorch. |

---

## Sony CSL Paris

- **Org:** [github.com/SonyCSLParis](https://github.com/SonyCSLParis)  
- **Site:** [csl.sony.fr](https://csl.sony.fr/)

| Repo | License | Stars | Value |
|------|---------|-------|--------|
| [pesto](https://github.com/SonyCSLParis/pesto) | LGPL-3.0 | 274 | Real-time pitch estimation (PESTO); `pip install pesto-pitch`; ONNX/JIT export. |
| [music2latent](https://github.com/SonyCSLParis/music2latent) | (check repo) | 241 | Audio ↔ latent representations. |
| [music-inpainting-ts](https://github.com/SonyCSLParis/music-inpainting-ts) | (check repo) | 138 | Web UIs for AI-assisted music creation (TypeScript). |
| [DrumGAN](https://github.com/SonyCSLParis/DrumGAN) | (check repo) | 125 | Drum synthesis with timbral conditioning (GANs). |
| [codicodec](https://github.com/SonyCSLParis/codicodec) | (check repo) | 79 | Audio encode/decode (continuous/discrete). |
| [audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | GPL-3.0 | 41 | Audio evaluation metrics. |
| [Stem-JEPA](https://github.com/SonyCSLParis/Stem-JEPA) | LGPL-3.0 | (check repo) | Stem compatibility in latent space (ISMIR 2024); arrangement validation, stem suggestions. See `docs/research/STEM_JEPA_INTEGRATION.md`. |

---

## Magenta

| Repo / data | Note | Value |
|-------------|------|--------|
| [magenta/magenta](https://github.com/magenta/magenta) | Archived Jan 2026. | Groove MIDI, MAESTRO, NSynth via `magentadata` URLs. |
| [Magenta org](https://github.com/magenta) | Current work. | Individual repos for new models/tools. |

---

## JUCE

| Repo | Value |
|------|--------|
| [juce-framework/JUCE](https://github.com/juce-framework/JUCE) | Audio/plugin framework; used in `external/JUCE`. |

---

## PR snippet (copy into PR description)

```markdown
## External / GitHub references
- **Alain Riou (aRI0U):** [deep-music-generation](https://github.com/aRI0U/deep-music-generation), [music-source-separation](https://github.com/aRI0U/music-source-separation).
- **Sony CSL Paris:** [PESTO](https://github.com/SonyCSLParis/pesto) (pitch estimation), [music2latent](https://github.com/SonyCSLParis/music2latent), [DrumGAN](https://github.com/SonyCSLParis/DrumGAN), [codicodec](https://github.com/SonyCSLParis/codicodec). See `docs/EXTERNAL_REFERENCES.md` and `docs/GITHUB_DATA_REFERENCES.md`.
- **Magenta:** Datasets (Groove MIDI, MAESTRO, NSynth) via magentadata; repo archived, see Magenta org for current work.
```
