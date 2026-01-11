"""
Base classes for generative models.

Provides common interfaces and utilities for all generative models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np


@dataclass
class GenerativeConfig:
    """Configuration for generative models."""
    
    # Device settings
    device: str = "auto"  # "auto", "mps", "cuda", "cpu"
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    
    # Model settings
    model_path: Optional[str] = None
    model_name: str = "default"
    
    # Generation settings
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    seed: Optional[int] = None
    
    # Audio settings
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    
    # Performance
    batch_size: int = 1
    use_cache: bool = True
    
    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


class GenerativeModel(ABC):
    """
    Abstract base class for all generative models.
    
    Provides a common interface for loading, generating, and managing
    generative models across different architectures.
    """
    
    def __init__(self, config: Optional[GenerativeConfig] = None):
        """Initialize the generative model."""
        self.config = config or GenerativeConfig()
        self._device = self.config.get_device()
        self._model = None
        self._is_loaded = False
    
    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """Load model weights from path."""
        pass
    
    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate output from the model."""
        pass
    
    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._is_loaded = False
    
    def to_device(self, device: str) -> "GenerativeModel":
        """Move model to specified device."""
        self._device = device
        if self._model is not None:
            try:
                self._model = self._model.to(device)
            except AttributeError:
                pass  # Model may not be a PyTorch module
        return self
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        seed = seed or self.config.seed
        if seed is not None:
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                if self._device == "cuda":
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass


@dataclass
class GenerationResult:
    """Result from a generative model."""
    
    # The generated output (audio array, MIDI, etc.)
    output: Any
    
    # Metadata about the generation
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    model_name: str = ""
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Optional additional outputs
    intermediate_outputs: List[Any] = field(default_factory=list)
    latents: Optional[np.ndarray] = None
    
    def to_audio_array(self) -> np.ndarray:
        """Convert output to numpy audio array."""
        if isinstance(self.output, np.ndarray):
            return self.output
        elif hasattr(self.output, "numpy"):
            return self.output.numpy()
        else:
            raise ValueError(f"Cannot convert {type(self.output)} to audio array")
    
    def save(self, path: str, format: str = "wav") -> str:
        """Save generated audio to file."""
        try:
            import soundfile as sf
            audio = self.to_audio_array()
            sf.write(path, audio.T if audio.shape[0] <= 2 else audio, self.sample_rate)
            return path
        except ImportError:
            raise ImportError("soundfile required for saving audio: pip install soundfile")
