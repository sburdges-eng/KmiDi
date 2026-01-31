"""
Image generation engine — reimplementation for orchestrator.

Provides ImageGenerationEngine: optional Stable Diffusion (diffusers) pipeline;
when not installed or configured, returns stubbed result so orchestrator continues.
Contract: generate() returns dict with status in {stubbed, completed, failed},
details, and when completed: output_path and optionally image_data_base64.

Deps (optional): diffusers, torch, transformers. Install when image gen needed.
Env: KMI_DI_IMAGE_MODEL_PATH or STABLE_DIFFUSION_MODEL_PATH — model dir or hub id
     (e.g. runwayml/stable-diffusion-v1-5). Default: stub; check mode runs without.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import threading
import base64
import io

try:
    from diffusers import StableDiffusionPipeline  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionPipeline = None
    torch = None
    logging.getLogger(__name__).warning(
        "diffusers/torch not found. Image generation will be stubbed."
    )


class ImageGenerationEngine:
    """Image generation engine. Loads SD pipeline when available; otherwise stub."""

    def __init__(
        self,
        model_dir: str = "./stable_diffusion_v1_5",
        device: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None
        self._device = device
        self.lock = threading.Lock()

    def _load_pipeline(self) -> None:
        """Load image generation pipeline. No-op if diffusers unavailable or load fails."""
        if not DIFFUSERS_AVAILABLE:
            logging.getLogger(__name__).info(
                "Image pipeline not loaded: diffusers/torch not installed."
            )
            return
        if self._pipeline is not None:
            return
        import os
        hub_id = os.environ.get(
            "KMI_DI_IMAGE_MODEL_PATH",
            os.environ.get("STABLE_DIFFUSION_MODEL_PATH", "runwayml/stable-diffusion-v1-5"),
        )
        try:
            logging.getLogger(__name__).info("Loading image pipeline: %s", hub_id)
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                hub_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline = self._pipeline.to(dev)
            logging.getLogger(__name__).info("Image pipeline loaded on %s.", dev)
        except Exception as e:
            logging.getLogger(__name__).warning("Image pipeline load failed: %s", e)
            self._pipeline = None

    def generate(
        self,
        prompt: str,
        style_constraints: str = "",
        output_path: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        assume_locked: bool = False,
        lock_timeout: Optional[float] = 300.0,
    ) -> Dict[str, Any]:
        """Generate image from prompt. Returns dict: status, details; when completed, output_path, optionally image_data_base64.

        Args:
            prompt: Text description for image generation.
            style_constraints: Optional style hints (appended to prompt if used).
            output_path: Optional path to save image; otherwise under model_dir.
            width, height: Output size.
            num_inference_steps: Diffusion steps.
            assume_locked: If True, skip lock (caller holds lock).
            lock_timeout: Lock timeout in seconds; None = block forever, 0 = non-blocking.
        """
        if not assume_locked:
            if lock_timeout is not None:
                acquired = self.lock.acquire(timeout=lock_timeout)
            else:
                acquired = self.lock.acquire()
            if not acquired:
                return {
                    "status": "timeout",
                    "details": "Could not acquire image generation lock.",
                    "prompt": prompt,
                }
        try:
            if DIFFUSERS_AVAILABLE and self._pipeline is None:
                try:
                    self._load_pipeline()
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Image pipeline load failed; stub: %s", e
                    )
                    self._pipeline = None

            if not DIFFUSERS_AVAILABLE or self._pipeline is None:
                return {
                    "status": "stubbed",
                    "prompt": prompt,
                    "details": "Image pipeline not loaded (diffusers missing or model failed).",
                }

            full_prompt = f"{prompt}. {style_constraints}".strip() if style_constraints else prompt
            out_path = Path(output_path) if output_path else self.model_dir / "generated_image.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with torch.no_grad():
                    image = self._pipeline(
                        full_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                    ).images[0]
                image.save(str(out_path))
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                image_data_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return {
                    "status": "completed",
                    "prompt": prompt,
                    "output_path": str(out_path),
                    "image_data_base64": image_data_base64,
                    "details": "Image generated successfully.",
                }
            except Exception as e:
                logging.getLogger(__name__).exception("Image generation failed: %s", e)
                return {
                    "status": "failed",
                    "prompt": prompt,
                    "details": str(e),
                }
        finally:
            if not assume_locked:
                try:
                    self.lock.release()
                except RuntimeError:
                    pass
