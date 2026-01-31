# Experiments

Experimental code lives here. **Promote to core only after validation.** (Governance: EXPERIMENT LAW.)

## Naming

Use: `exp_NNN_short_description` (e.g. `exp_001_emotion_encoder`, `exp_002_groove_ablation`).

- `NNN` = zero-padded number (001, 002, …).
- `short_description` = lowercase, underscores.

## Layout

```
experiments/
  exp_001_description/
    README.md       # goal, setup, results summary
    config.yaml     # optional
    ...             # code / notebooks
  exp_002_other/
    ...
```

Do not scatter experimental code across `src/` or `KmiDi_CANON/`. Keep experiments isolated; merge into core only after review.
