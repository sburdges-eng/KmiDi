# Research Directory

This directory contains research documents and integration plans for external technologies and academic work that could enhance KmiDi.

## Current Research

### Stem-JEPA Integration
**Document:** [STEM_JEPA_INTEGRATION.md](STEM_JEPA_INTEGRATION.md)  
**Author:** Alain Riou (aRI0U) - Sony CSL Paris  
**Status:** Planning Phase

Research on integrating Joint-Embedding Predictive Architecture (JEPA) for musical stem compatibility estimation into KmiDi's generation and arrangement systems.

**Key Benefits:**
- Validate arrangement quality through learned compatibility
- Intelligent stem suggestions for completing arrangements
- Self-supervised learning from unlabeled audio
- Emotion-aware compatibility checking

**References:**
- GitHub: https://github.com/SonyCSLParis/Stem-JEPA
- Paper: https://arxiv.org/abs/2408.02514
- Conference: ISMIR 2024

### Multimodal Representation & JEPA Ecosystem (2026)
**Document:** [MULTIMODAL_REPRESENTATIONS_2026.md](MULTIMODAL_REPRESENTATIONS_2026.md)  
**Author:** Internal synthesis (KmiDi)  
**Status:** Planning / Design

Survey of 2026-era multimodal representation learning (alignment, canonicalization, continual learning, representation analysis) and practical audio/symbolic tooling for KmiDi. Captures concrete integration paths around Perch audio embeddings, REMI-BPE tokenization, Lhotse+DataLad manifests, BNNS + Audio Workgroups, a shared C ABI for the DSP/latent engine, and MIDI 2.0 Property Exchange + UMP for live affect control.

### KmiDi 90-Day Demo Roadmap (2026 Q2)
**Document:** [KMIDI_90_DAY_DEMO_ROADMAP_2026.md](KMIDI_90_DAY_DEMO_ROADMAP_2026.md)  
**Author:** User briefing captured in repo  
**Status:** Proposed

Execution-focused roadmap for a demo-ready local stack: canonical emotion and intent contracts, a lightweight JEPA audio encoder, and a local AU helper that maps intent to DSP parameters under strict latency constraints.

### KmiDi Platform Watchlist (2026)
**Document:** [KMIDI_PLATFORM_WATCHLIST_2026.md](KMIDI_PLATFORM_WATCHLIST_2026.md)  
**Author:** User briefings captured in repo  
**Status:** Informational / planning input

Consolidated notes covering multi-agent baton handoffs, Tauri updater rollout, Core ML/ExecuTorch export and attestation concerns, stateful KV-cache loops, symbolic tokenizer defaults, expressive MIDI datasets, and hardware/controller watch items.

## Integration Status

| Technology | Status | Documentation | Next Steps |
|-----------|---------|---------------|-----------|
| Stem-JEPA | Planning | STEM_JEPA_INTEGRATION.md | Proof-of-concept |
| Multimodal JEPA & tooling | Planning | MULTIMODAL_REPRESENTATIONS_2026.md | Perch+REMI-BPE prototype, JEPA manifest + RT/PE pipelines |
| 90-day demo slice | Proposed | KMIDI_90_DAY_DEMO_ROADMAP_2026.md | Canonical schema, short-window JEPA, local AU helper |
| Platform/runtime watchlist | Informational | KMIDI_PLATFORM_WATCHLIST_2026.md | Verify sources, promote adopted items into canonical schemas/docs |

## References from Pulse recovery

Recovered ChatGPT Pulse entries (see [../PULSE_RECOVERY_ENTRIES.md](../PULSE_RECOVERY_ENTRIES.md)) surface the following for benchmarks, audit, and datasets:

- **AI music benchmarks & audit (#010068):** NIST draft standards, MuSpike benchmark, I‑O audit architecture, improved audio metrics (MAD) for reproducibility and model audits. Use when defining evaluation or guardrails.
- **MIDI tooling (#010051):** Controllable MIDI models (MIDI GPT), tuning practices; Autotroph noted as proprietary/internal with no public trace.
- **MIDI datasets & models (#010067):** Large MIDI dataset releases and new symbolic music models. Cross-reference when curating datasets or reviewing literature.

## Contributing Research

To add new research integration proposals:

1. Create a new markdown document in this directory
2. Use the template structure from existing documents
3. Include:
   - Executive summary
   - Technical details
   - Integration opportunities
   - Implementation plan
   - References
4. Update this README with the new research entry

## Research Template

```markdown
# [Technology Name] Integration Research

**Author:** [Original Author/Institution]
**Research Date:** [Date]
**Status:** [Planning/Prototype/Production]
**Repository:** [GitHub URL if applicable]

## Executive Summary
[Brief overview of the technology and its value for KmiDi]

## Technical Details
[Architecture, capabilities, requirements]

## Integration Opportunities
[How it fits into KmiDi's architecture]

## Implementation Plan
[Phased approach to integration]

## References
[Papers, repos, documentation]
```

## Research Areas of Interest

Areas where external research could benefit KmiDi:

1. **Music Information Retrieval (MIR)**
   - Audio analysis and feature extraction
   - Genre and style classification
   - Music structure analysis

2. **Machine Learning for Music**
   - Generative models for composition
   - Style transfer and transformation
   - Self-supervised learning approaches

3. **Audio Processing**
   - Real-time effects and synthesis
   - Audio quality enhancement
   - Spatial audio and mixing

4. **Human-Computer Interaction (HCI)**
   - Novel music interfaces
   - Gesture and motion control
   - Accessibility features

5. **Music Theory and Cognition**
   - Computational music theory
   - Emotional response modeling
   - Music perception research

## Contact

For questions about research integrations, please open an issue in the main repository.
