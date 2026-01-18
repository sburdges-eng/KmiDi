"""
ImageGenerationJob
------------------
Local Stable Diffusion 1.5 (Metal/MPS) invocation for image generation.

Responsibilities:
 - Accept prompts + style rules from BrainController
 - Run SD 1.5 locally (no network, no telemetry)
 - Return image paths + metadata

Constraints:
 - SDXL excluded (memory)
 - No silent retries / prompt mutation
 - Fail loudly when resources unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import contextlib
import os


def _try_import_diffusers():
    try:
        from diffusers import StableDiffusionPipeline
        import torch

        return StableDiffusionPipeline, torch
    except Exception:
        return None, None


@dataclass
class ImageGenerationConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: Path = Path.home() / "Music" / "iDAW_Output" / "images"
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: Optional[int] = 42
    use_mps: bool = True  # Apple Silicon


class ImageGenerationJob:
    def __init__(self, config: Optional[ImageGenerationConfig] = None):
        self.config = config or ImageGenerationConfig()
        self._pipeline = None
        self._sd_cls, self._torch = _try_import_diffusers()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self._pipeline or not self._sd_cls:
            return
        torch_device = "mps" if (self.config.use_mps and self._torch and self._torch.backends.mps.is_available()) else "cpu"
        self._pipeline = self._sd_cls.from_pretrained(
            self.config.model_id,
            torch_dtype=self._torch.float16 if torch_device == "mps" else self._torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self._pipeline = self._pipeline.to(torch_device)
        if torch_device == "mps":
            # Disable TF32 for consistency on Apple Silicon
            self._torch.backends.cuda.matmul.allow_tf32 = False if self._torch.backends.cuda.is_available() else False

    def unload(self):
        self._pipeline = None

    def generate(
        self,
        prompt: str,
        style_rules: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
    ) -> List[Path]:
        """
        Generate images from prompt + style rules.
        """
        if not self._sd_cls:
            raise ImportError("diffusers[torch] is required for ImageGenerationJob.")
        self._load()

        full_prompt = self._compose_prompt(prompt, style_rules or {})
        generator = None
        if self.config.seed is not None and self._torch:
            generator = self._torch.Generator(device=self._pipeline.device).manual_seed(self.config.seed)

        outputs = self._pipeline(
            prompt=full_prompt,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        saved_paths: List[Path] = []
        for idx, image in enumerate(outputs.images):
            out_path = self.config.output_dir / f"sd15_{self.config.seed or 0}_{idx}.png"
            image.save(out_path)
            saved_paths.append(out_path)

        return saved_paths

    def generate_from_intent(
        self,
        intent: Any,
        prompt: str,
        style_rules: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
    ) -> List[Path]:
        """
        Generate images using prompt hints provided by BrainController.
        """
        return self.generate(prompt=prompt, style_rules=style_rules, num_images=num_images)

    @staticmethod
    def _compose_prompt(prompt: str, style_rules: Dict[str, Any]) -> str:
        if not style_rules:
            return prompt
        style_fragments = []
        for key, value in style_rules.items():
            if value:
                style_fragments.append(f"{key}: {value}")
        style_text = ", ".join(style_fragments)
        return f"{prompt}. Style: {style_text}" if style_text else prompt


# Convenience factory
def create_image_job(config: Optional[ImageGenerationConfig] = None) -> ImageGenerationJob:
    return ImageGenerationJob(config=config)

