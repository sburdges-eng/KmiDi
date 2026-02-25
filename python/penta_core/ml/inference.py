"""
Core Inference Module - Base inference classes and results.

Provides:
- InferenceResult dataclass for standard inference outputs
- BaseInferenceEngine abstract class for inference implementations
- Model loading utilities
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class InferenceStatus(Enum):
    """Status of an inference operation."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class InferenceResult:
    """
    Result from an inference operation.

    Attributes:
        outputs: Dictionary of output tensors/arrays
        latency_ms: Inference latency in milliseconds
        status: Status of the inference operation
        model_name: Name of the model used
        metadata: Additional metadata about the inference
        error: Error message if status is ERROR
    """
    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    latency_ms: float = 0.0
    status: InferenceStatus = InferenceStatus.SUCCESS
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if inference was successful."""
        return self.status == InferenceStatus.SUCCESS

    def get_output(self, name: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get a specific output by name."""
        return self.outputs.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "outputs": {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in self.outputs.items()},
            "latency_ms": self.latency_ms,
            "status": self.status.value,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "error": self.error,
        }


class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.

    Implementations should handle model loading, inference execution,
    and resource management.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        use_fallback: bool = True
    ) -> InferenceResult:
        """
        Run inference on inputs.

        Args:
            inputs: Dictionary of input tensors
            use_fallback: Whether to use fallback on error

        Returns:
            InferenceResult with outputs or error
        """
        pass

    def unload(self) -> None:
        """Unload the model and free resources."""
        self._model = None
        self._loaded = False

    def warmup(self, warmup_inputs: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        Warmup the model with sample inputs.

        Args:
            warmup_inputs: Optional inputs for warmup, or use defaults

        Returns:
            True if warmup successful
        """
        if not self._loaded:
            return False

        try:
            if warmup_inputs is None:
                # Create default warmup inputs based on model
                warmup_inputs = self._create_default_warmup_inputs()

            if warmup_inputs:
                self.infer(warmup_inputs, use_fallback=False)
            return True
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            return False

    def _create_default_warmup_inputs(self) -> Dict[str, np.ndarray]:
        """Create default warmup inputs. Override in subclasses."""
        return {}


class SimpleInferenceEngine(BaseInferenceEngine):
    """
    Simple inference engine for basic model execution.

    Supports PyTorch, ONNX, and numpy-based models.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "auto"
    ):
        super().__init__(model_path)
        self.backend = backend
        self._session = None

    def load(self) -> bool:
        """Load model from path."""
        if not self.model_path:
            logger.error("No model path specified")
            return False

        path = Path(self.model_path)
        if not path.exists():
            logger.error(f"Model path does not exist: {path}")
            return False

        try:
            if path.suffix == ".onnx":
                return self._load_onnx(path)
            elif path.suffix in (".pt", ".pth"):
                return self._load_torch(path)
            elif path.suffix == ".npy":
                return self._load_numpy(path)
            else:
                logger.error(f"Unsupported model format: {path.suffix}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_onnx(self, path: Path) -> bool:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(path))
            self._loaded = True
            self.backend = "onnx"
            return True
        except ImportError:
            logger.error("onnxruntime not installed")
            return False

    def _load_torch(self, path: Path) -> bool:
        """Load PyTorch model."""
        try:
            import torch
            self._model = torch.jit.load(str(path))
            self._model.eval()
            self._loaded = True
            self.backend = "torch"
            return True
        except ImportError:
            logger.error("torch not installed")
            return False

    def _load_numpy(self, path: Path) -> bool:
        """Load numpy weights (for simple models)."""
        try:
            self._model = np.load(str(path), allow_pickle=True).item()
            self._loaded = True
            self.backend = "numpy"
            return True
        except Exception as e:
            logger.error(f"Failed to load numpy model: {e}")
            return False

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        use_fallback: bool = True
    ) -> InferenceResult:
        """Run inference."""
        if not self._loaded:
            return InferenceResult(
                status=InferenceStatus.ERROR,
                error="Model not loaded"
            )

        start_time = time.perf_counter()

        try:
            if self.backend == "onnx":
                outputs = self._infer_onnx(inputs)
            elif self.backend == "torch":
                outputs = self._infer_torch(inputs)
            elif self.backend == "numpy":
                outputs = self._infer_numpy(inputs)
            else:
                return InferenceResult(
                    status=InferenceStatus.ERROR,
                    error=f"Unknown backend: {self.backend}"
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return InferenceResult(
                outputs=outputs,
                latency_ms=latency_ms,
                status=InferenceStatus.SUCCESS,
                model_name=str(self.model_path)
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Inference failed: {e}")

            if use_fallback:
                return self._create_fallback_result(inputs, latency_ms, str(e))

            return InferenceResult(
                status=InferenceStatus.ERROR,
                error=str(e),
                latency_ms=latency_ms
            )

    def _infer_onnx(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run ONNX inference."""
        input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]

        # Map inputs to session input names
        feed_dict = {}
        for name in input_names:
            if name in inputs:
                feed_dict[name] = inputs[name]
            else:
                # Try to find matching input
                for k, v in inputs.items():
                    if k not in feed_dict.values():
                        feed_dict[name] = v
                        break

        results = self._session.run(output_names, feed_dict)

        return {name: result for name, result in zip(output_names, results)}

    def _infer_torch(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run PyTorch inference."""
        import torch

        # Convert numpy to torch tensors
        torch_inputs = [torch.from_numpy(v) for v in inputs.values()]

        with torch.no_grad():
            outputs = self._model(*torch_inputs)

        # Handle single output or tuple
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        return {f"output_{i}": out.numpy() for i, out in enumerate(outputs)}

    def _infer_numpy(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run numpy-based inference (simple linear models)."""
        if callable(self._model):
            result = self._model(inputs)
            if isinstance(result, dict):
                return result
            return {"output": result}

        # Assume it's a weights dict for simple linear model
        x = list(inputs.values())[0]
        if "weight" in self._model and "bias" in self._model:
            output = np.dot(x, self._model["weight"]) + self._model["bias"]
            return {"output": output}

        return {"output": x}

    def _create_fallback_result(
        self,
        inputs: Dict[str, np.ndarray],
        latency_ms: float,
        error: str
    ) -> InferenceResult:
        """Create fallback result on error."""
        # Create zero outputs matching input shapes
        outputs = {}
        for name, arr in inputs.items():
            outputs[f"output_{name}"] = np.zeros_like(arr)

        return InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            status=InferenceStatus.ERROR,
            error=f"Fallback used: {error}",
            metadata={"fallback": True}
        )


def load_model(
    model_path: str,
    backend: str = "auto"
) -> Optional[BaseInferenceEngine]:
    """
    Load a model and return an inference engine.

    Args:
        model_path: Path to the model file
        backend: Backend to use ("auto", "onnx", "torch", "numpy")

    Returns:
        Inference engine if successful, None otherwise
    """
    engine = SimpleInferenceEngine(model_path, backend)
    if engine.load():
        return engine
    return None


__all__ = [
    "InferenceStatus",
    "InferenceResult",
    "BaseInferenceEngine",
    "SimpleInferenceEngine",
    "load_model",
    # Aliases for compatibility with __init__.py expectations
    "InferenceEngine",
    "create_engine",
]

# Aliases for compatibility
InferenceEngine = SimpleInferenceEngine

def create_engine(model_path: str, backend: str = "auto") -> Optional[BaseInferenceEngine]:
    """Create an inference engine for the given model path."""
    return load_model(model_path, backend)
