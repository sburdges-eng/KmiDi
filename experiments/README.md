# experiments/

Isolated scratch space for hypotheses that have **not yet been validated** for core promotion.

## Naming convention

Every experiment lives in its own directory named:

```
exp_NNN_short_description/
```

- `NNN` — zero-padded three-digit sequence number (`001`, `002`, …)
- `short_description` — lowercase, underscores, ≤ 40 chars

Examples: `exp_001_ump_jepa/`, `exp_002_wavjepa_emotion/`

## Rules

1. **Isolation** — no imports between experiments and core: experimental code must not be imported by anything in `music_brain/`, `src/`, `src-tauri/`, or any other core path, and experiments must not import from those core paths either.
2. **Self-contained** — each experiment folder must have its own `README.md` explaining the hypothesis, datasets, and how to run.
3. **Promotion gate** — to move code into core, open a PR marked `[promote]` and get at least one review confirming the experiment passed its stated acceptance criteria.
4. **No secrets / no large binaries** — use `.gitignore` inside the experiment folder for any data, model weights, or generated outputs.

## Current experiments

| # | Directory | Status | Description |
|---|-----------|--------|-------------|
| 001 | [exp_001_ump_jepa](exp_001_ump_jepa/) | 🔬 active | UMP → JEPA expressive conditioning scaffold |
| 002 | [exp_002_wavjepa_emotion](exp_002_wavjepa_emotion/) | 🔬 active | WavJEPA emotional separability probe |
| 003 | [exp_003_jepa_transcriber_probe](exp_003_jepa_transcriber_probe/) | 📋 stub | JEPA-to-MIDI transcriber probe (design only) |
| 004 | [exp_004_moe_bridge](exp_004_moe_bridge/) | 📋 stub | Frozen JEPA encoder + MoE bridge (design only) |

## Reference materials

NotebookLM mind maps (exported visuals for research/planning):

- [NotebookLM Mind Map.png](NotebookLM%20Mind%20Map.png)
- [NotebookLM Mind Map (1).png](NotebookLM%20Mind%20Map%20(1).png)
- [NotebookLM Mind Map (2).png](NotebookLM%20Mind%20Map%20(2).png)
- [NotebookLM Mind Map (3).png](NotebookLM%20Mind%20Map%20(3).png)

Ref: `.cursor/rules/engineering-governance.mdc`
