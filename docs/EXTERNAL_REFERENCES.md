# External References & Data Sources

References to external projects, datasets, and APIs used by KmiDi.

---

## Magenta (Music & Art with ML)

- **Repo:** [github.com/magenta/magenta](https://github.com/magenta/magenta)  
  *Note: This repository was [archived](https://github.com/magenta/magenta) by the owner (Jan 2026). It is read-only; the project has moved to individual repos under the [Magenta GitHub Organization](https://github.com/magenta).*
- **What we use:**  
  - **Groove MIDI Dataset** – expressive drum performances (e.g. `magentadata/datasets/groove/groove-v1.0.0-midionly.zip`) for groove training.  
  - **MAESTRO**, **NSynth**, and other datasets served from `storage.googleapis.com/magentadata/` (see `scripts/prepare_datasets.py` / `scripts/utilities/prepare_datasets.py`).
- **Docs / current work:** [magenta.tensorflow.org](https://magenta.tensorflow.org), [Magenta.js](https://github.com/magenta/magenta-js) for browser models.

---

## Alain Riou (aRI0U) & Sony CSL Paris

**Alain Riou** ([aRI0U](https://github.com/aRI0U) on GitHub) is an Assistant Researcher at [Sony CSL Paris](https://csl.sony.fr/) focused on music and machine learning (self-supervised learning, pitch estimation, source separation). His work targets ML tools for composition (melodies, arrangements) that stay lightweight and user-friendly.

### aRI0U (personal repos)

| Repo | Description | Value for KmiDi |
|------|-------------|-----------------|
| [aRI0U/deep-music-generation](https://github.com/aRI0U/deep-music-generation) | Music generation with deep learning (downloader, encoder, model, tonality). | Ideas for melody/arrangement generation; encoder/tonality modules. |
| [aRI0U/music-source-separation](https://github.com/aRI0U/music-source-separation) | PyTorch implementation of “Improving DNN-based Music Source Separation using Phase Features” (Muth et al.). Uses MUSDB18. | Source separation for stems; phase features; training pipeline. |

### Sony CSL Paris (organization)

| Repo | Description | Value for KmiDi |
|------|-------------|-----------------|
| [SonyCSLParis/pesto](https://github.com/SonyCSLParis/pesto) | **PESTO** – Pitch Estimation with Self-supervised Transposition-equivariant Objective. Real-time pitch estimation; Python API; ONNX/JIT export. | Real-time F0 for voice/instruments; lightweight; `pip install pesto-pitch`. |
| [SonyCSLParis/music2latent](https://github.com/SonyCSLParis/music2latent) | Encode/decode audio to/from compressed latent representations. | Latent audio representations for generation/editing. |
| [SonyCSLParis/music-inpainting-ts](https://github.com/SonyCSLParis/music-inpainting-ts) | Web interfaces for AI-assisted interactive music creation (TypeScript). | UI/UX patterns for inpainting and interactive creation. |
| [SonyCSLParis/DrumGAN](https://github.com/SonyCSLParis/DrumGAN) | Drum sound synthesis with perceptual timbral conditioning (GANs). | Drum timbre generation; conditioning. |
| [SonyCSLParis/codicodec](https://github.com/SonyCSLParis/codicodec) | Encode/decode audio to/from continuous and discrete compressed representations. | Audio codec for low-bitrate or latent pipelines. |
| [SonyCSLParis/audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | Audio evaluation metrics. | Evaluation for separation, generation, or pitch. |

**Citations (PESTO):** [ISMIR 2023](https://arxiv.org/abs/2309.02265), [PESTO v2 2025](https://arxiv.org/abs/2508.01488). See repo README for BibTeX.

---

## Other references

- **JUCE:** [github.com/juce-framework/JUCE](https://github.com/juce-framework/JUCE) – audio/plugin framework (see `external/JUCE`).
- **Path and dependency audit:** `docs/EXTERNAL_PATH_REFERENCES.md` (paths outside KmiDi-compile).
- **Valuable GitHub data (PR-ready list):** `docs/GITHUB_DATA_REFERENCES.md`.
