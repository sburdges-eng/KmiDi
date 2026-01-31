"""
Image generation engine stub.

Provides ImageGenerationEngine for orchestrator when full pipeline (e.g. Stable Diffusion)
is not installed or configured. Replace with real implementation when needed.
"""

from pathlib import Path
from typing import Any, Dict, Optional


class ImageGenerationEngine:
    """Stub image generation engine. Load pipeline when available."""

    def __init__(self, model_dir: str = "./stable_diffusion_v1_5"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None

    def _load_pipeline(self) -> None:
        """Load image generation pipeline. Stub: no-op; real impl would load SD etc."""
        # Optional: try import diffusers etc. and set self._pipeline
        pass
