# macOS Metal/ANE setup (Apple M4)

Use this when running locally on the M4 MacBook Pro for inference, DSP, and light training.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.macos-metal.txt
```

PyTorch’s MPS wheels are on the default index (no CUDA index needed). If MPS issues appear, try nightly wheels:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Device selection helper (PyTorch)

```python
import torch

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # unlikely on Apple Silicon, but keeps code portable
        return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()
print("Using", device)
```

For training, prefer `dtype=torch.float16` or `bfloat16` to stay within 16 GB RAM.

## MLX quickstart

```python
import mlx.core as mx
import mlx.nn as nn

model = nn.Linear(16, 4)
x = mx.random.normal((8, 16))
y = model(x)
print(y)
```

## CoreML / ONNX

- Convert: PyTorch → ONNX → CoreML via `coremltools.convert`.
- Inference: use `onnxruntime-silicon` (Metal EP) or CoreML (`coremltools` with `compute_units="ALL"`).

## Audio DSP

- Use `torchaudio` + `librosa` + `soundfile` for feature work.
- For real-time paths, prefer C++/Rust/Swift with Metal or Accelerate/vDSP; Python is fine for offline/nearline.

## When to use the Linux devcontainer

- Heavy CUDA training or deployment targets that require NVIDIA GPUs.
- Otherwise, run natively on macOS to access Metal/ANE; containers on macOS cannot access Metal/ANE.***
