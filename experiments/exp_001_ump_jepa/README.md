# UMP → JEPA expressive conditioning (experiment)

Scaffold for windowed UMP control aggregation and control embedding used to condition JEPA context encoder. See [docs/UMP_JEPA_EXPRESSIVE_CONDITIONING.md](../../docs/UMP_JEPA_EXPRESSIVE_CONDITIONING.md).

- **ump_aggregate.py** — Perceptual quantization and window aggregation (mean, std, slope per CC).
- **control_embedding.py** — Embedding layer for aggregated control state (for injection as bias into context encoder).

Not part of core; validate in experiments before promoting.
