from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class DeviceInfo:
    backend: str
    device: str
    precision: str
    reason: str
    available: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch_info() -> Optional[DeviceInfo]:
    try:
        import torch
    except ImportError:
        return None

    mps_available = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_available and torch.backends.mps.is_available())
    cuda_ok = torch.cuda.is_available()

    if cuda_ok:
        return DeviceInfo(
            backend="torch-cuda",
            device="cuda",
            precision="float16",
            reason="CUDA available",
            available=True,
            details={"torch": torch.__version__},
        )

    if mps_ok:
        return DeviceInfo(
            backend="torch-mps",
            device="mps",
            precision="float16",
            reason="Metal (MPS) available",
            available=True,
            details={"torch": torch.__version__},
        )

    return DeviceInfo(
        backend="torch-cpu",
        device="cpu",
        precision="float32",
        reason="Torch found, GPU backends unavailable",
        available=True,
        details={"torch": torch.__version__},
    )


def _mlx_info() -> Optional[DeviceInfo]:
    try:
        import mlx.core as mx  # type: ignore[import-untyped]
    except ImportError:
        return None

    return DeviceInfo(
        backend="mlx",
        device="mlx:0",
        precision="float16",
        reason="MLX available",
        available=True,
        details={"mlx": getattr(mx, '__version__', 'unknown')},
    )


def select_device(preference: Optional[str] = None) -> DeviceInfo:
    """
    Select a compute backend with sane defaults:
    - prefer explicit preference if available
    - otherwise prefer CUDA > MPS > MLX > CPU
    """
    pref = (preference or "").lower()

    torch_info = _torch_info()
    mlx_info = _mlx_info()

    if pref in {"cuda", "gpu"} and torch_info and torch_info.device == "cuda":
        return torch_info

    if pref in {"mps", "metal"} and torch_info and torch_info.device == "mps":
        return torch_info

    if pref == "mlx" and mlx_info:
        return mlx_info

    if torch_info:
        return torch_info

    if mlx_info:
        return mlx_info

    return DeviceInfo(
        backend="cpu",
        device="cpu",
        precision="float32",
        reason="Fallback to CPU; torch/MLX missing",
        available=True,
        details={},
    )


def suggest_settings(info: DeviceInfo) -> Dict[str, Any]:
    """
    Provide simple dtype/batch size suggestions based on backend.
    """
    if info.backend.startswith("torch-cuda"):
        return {"dtype": "float16", "batch_size_hint": "medium", "notes": "CUDA detected"}
    if info.backend.startswith("torch-mps"):
        return {
            "dtype": "float16",
            "batch_size_hint": "small",
            "notes": "Apple Metal (MPS); prefer fp16 to conserve unified memory",
        }
    if info.backend.startswith("mlx"):
        return {
            "dtype": "float16",
            "batch_size_hint": "small",
            "notes": "MLX backend on Apple Silicon; favors fp16",
        }
    return {"dtype": "float32", "batch_size_hint": "small", "notes": "CPU fallback"}
