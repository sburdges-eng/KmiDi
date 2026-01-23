# 🍎 MLX-Only Workflow (Experimental)

This folder defines a **single-MLX workflow** for running the entire KmiDi model lifecycle on a Mac (M4, 16GB RAM). It is designed as an **experiment** alongside the primary online training vision.

## Goals

- Run **end-to-end training, fine-tuning, and inference** using MLX.
- Prioritize **M4 Metal performance** with a strict 16GB memory budget.
- Keep the workflow **self-contained**, so MLX can run the entire project pipeline alone.

## What This Covers

- **Spectocloud** vision model (spectrogram → point cloud)
- **MIDI generator** (token generation with emotion conditioning)
- **Lightweight evaluation + export** for on-device inference

## Quick Start (M4 16GB)

```bash
# 1. Create/activate env
python3 -m venv venv
source venv/bin/activate

# 2. Install MLX stack
pip install mlx mlx-lm numpy scipy librosa pyyaml

# 3. Use the MLX workflow config
cat training/mlx_session/mlx_workflow.yaml

# 4. Run the MLX workflow runner (create your own wrapper)
# Example stub command:
# python scripts/mlx/run_mlx_workflow.py --config training/mlx_session/mlx_workflow.yaml
```

## Memory-First Settings (16GB)

- **Spectocloud**
  - batch_size: **4**
  - grad_accum_steps: **4** (effective batch = 16)
  - mixed precision: **fp16**
- **MIDI generator**
  - batch_size: **6**
  - sequence length: **512**
  - gradient checkpointing: **on**

## Notes

- This MLX workflow is intentionally **self-contained** and does **not** rely on the CUDA session.
- Keep the **online training** path as the primary production vision. MLX is experimental.
- Use `mlx_workflow.yaml` as the source of truth for resource limits and scheduling.

