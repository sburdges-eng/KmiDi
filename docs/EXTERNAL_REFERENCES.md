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
# External References and Resources

This document catalogs external repositories, papers, and resources that are relevant to KmiDi's development and could provide valuable implementations or insights.

**Last Updated:** 2026-01-28

## GitHub Repositories

### Self-Supervised Learning for Music

#### Stem-JEPA (Alain Riou - aRI0U)
**Repository:** https://github.com/SonyCSLParis/Stem-JEPA  
**Fork:** https://github.com/aRI0U/Stem-JEPA  
**Author:** Alain Riou (aRI0U)  
**Institution:** Sony Computer Science Laboratories Paris, Télécom Paris  
**License:** Check repository  
**Status:** Active (as of 2024)

**Description:**  
Joint-Embedding Predictive Architecture for Musical Stem Compatibility Estimation. Uses self-supervised learning to determine when different musical parts (stems) are compatible and work well together.

**Key Features:**
- Self-supervised learning (no labels required)
- Stem compatibility estimation
- Stem retrieval and arrangement
- Genre and key estimation
- Temporal alignment

**Relevance to KmiDi:**
- Validates arrangement quality
- Suggests compatible instruments
- Learns from unlabeled audio
- Enhances emotion-driven generation

**Technologies:**
- PyTorch, PyTorch Lightning
- Hydra configuration
- Dora job scheduling
- EVAR evaluation framework

**Integration Status:** Planning (see `docs/research/STEM_JEPA_INTEGRATION.md`)

**References:**
- Paper (ISMIR 2024): https://arxiv.org/abs/2408.02514
- Author Profile: https://github.com/aRI0U

---

### Related JEPA Implementations

#### I-JEPA (Image JEPA)
**Repository:** https://github.com/facebookresearch/ijepa  
**Author:** Meta AI Research (FAIR)  
**Description:** Joint-Embedding Predictive Architecture for images

#### V-JEPA (Video JEPA)
**Repository:** https://github.com/facebookresearch/jepa  
**Author:** Meta AI Research (FAIR)  
**Description:** Joint-Embedding Predictive Architecture for video

#### Awesome JEPA
**Repository:** https://github.com/gauravfs-14/awesome-jepa  
**Description:** Curated list of JEPA papers and implementations

---

## Research Papers

### Music AI and Generation

1. **Stem-JEPA: A Joint-Embedding Predictive Architecture for Musical Stem Compatibility Estimation**
   - Authors: Riou, A., Lattner, S., Hadjeres, G., Anslow, M., Peeters, G.
   - Venue: ISMIR 2024
   - arXiv: https://arxiv.org/abs/2408.02514
   - Topics: Self-supervised learning, stem compatibility, music arrangement

2. **A Path Towards Autonomous Machine Intelligence**
   - Author: Yann LeCun (Meta AI)
   - Topics: JEPA paradigm, predictive learning
   - Note: Foundational paper for JEPA approach

---

## Author Profiles

### Alain Riou (aRI0U)
**GitHub:** https://github.com/aRI0U  
**Institution:** Sony CSL Paris, Télécom Paris  
**Focus:** Self-supervised learning for music, stem compatibility, music information retrieval

**Notable Work:**
- Stem-JEPA (ISMIR 2024)
- Research in music AI and generation

**Potential Collaboration:**
- Stem compatibility for KmiDi arrangements
- Self-supervised learning approaches
- Music AI techniques

---

## Tools and Frameworks

### Audio Processing

#### EVAR (Evaluation of Audio Representations)
**Repository:** https://github.com/nttcslab/eval-audio-repr  
**Description:** Framework for evaluating learned audio representations  
**Use Case:** Evaluate JEPA models on music tasks

#### Hydra
**Repository:** https://github.com/facebookresearch/hydra  
**Description:** Configuration management for Python applications  
**Use Case:** Used by Stem-JEPA for configuration

#### Dora
**Repository:** https://github.com/facebookresearch/dora  
**Description:** Experiment management and job scheduling  
**Use Case:** Used by Stem-JEPA for training orchestration

---

## Integration Roadmap

### Immediate (Current)
- [x] Research Stem-JEPA capabilities
- [x] Document integration opportunities
- [x] Create stub modules in KmiDi

### Short-term (Next 1-3 months)
- [ ] Clone and test Stem-JEPA locally
- [ ] Create adapter for KmiDi audio format
- [ ] Proof-of-concept integration
- [ ] Performance benchmarking

### Medium-term (3-6 months)
- [ ] Integrate with arrangement engines
- [ ] Add to learning system
- [ ] User testing
- [ ] Optimization for real-time use

### Long-term (6+ months)
- [ ] Custom model training
- [ ] Emotion-aware compatibility
- [ ] Real-time stem suggestions
- [ ] Full production deployment

---

## Usage in KmiDi

### Current Integration Points

1. **Learning Module**  
   Location: `music_brain/learning/stem_compatibility.py`  
   Status: Stub implementation  
   Purpose: Interface for JEPA-based compatibility

2. **Research Documentation**  
   Location: `docs/research/STEM_JEPA_INTEGRATION.md`  
   Status: Complete  
   Purpose: Integration planning and architecture

### Future Integration Points

1. **Arrangement Engine**  
   Location: `music_brain/kelly_companion/engines/arrangement_engine.py`  
   Purpose: Validate arrangement quality

2. **Intelligence Layer**  
   Location: `music_brain/intelligence/`  
   Purpose: AI-powered stem suggestions

3. **Training Pipeline**  
   Location: `music_brain/penta_core/ml/training/`  
   Purpose: Self-supervised learning

---

## Contributing

To add new external resources:

1. Add entry to appropriate section
2. Include:
   - Repository/paper URL
   - Author/institution
   - Brief description
   - Relevance to KmiDi
   - Integration status
3. Update integration roadmap if applicable
4. Cross-reference in research documents

---

## Related Documentation

- [Research Directory README](research/README.md)
- [Stem-JEPA Integration Plan](research/STEM_JEPA_INTEGRATION.md)
- [KmiDi Architecture](ARCHITECTURE.md)
- [Learning Module](../music_brain/learning/)

---

## License Considerations

When integrating external code:
- Review license compatibility
- Maintain proper attribution
- Document dependencies
- Follow academic citation practices

For research implementations:
- Cite original papers
- Acknowledge authors
- Follow publication guidelines
- Maintain research integrity

---

**Maintained by:** KmiDi Development Team  
**Contact:** Open an issue in the main repository for questions or suggestions
