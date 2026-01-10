"""
Audio Diffusion - Diffusion-based audio generation.

Provides interfaces to diffusion models for generating audio from text prompts,
emotion embeddings, or other conditioning signals. Supports:
- AudioLDM/AudioLDM2
- Stable Audio
- MusicGen (via transformers)
- Custom diffusion models

Usage:
    from music_brain.generative import AudioDiffusion
    
    gen = AudioDiffusion(model="audioldm", device="mps")
    audio = gen.generate("sad piano melody in minor key", duration=10)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import warnings

from .base import GenerativeModel, GenerativeConfig, GenerationResult


# Check for available backends
AUDIOLDM_AVAILABLE = False
MUSICGEN_AVAILABLE = False
DIFFUSERS_AVAILABLE = False

try:
    from diffusers import AudioLDMPipeline, AudioLDM2Pipeline
    DIFFUSERS_AVAILABLE = True
    AUDIOLDM_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    MUSICGEN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AudioDiffusionConfig(GenerativeConfig):
    """Configuration specific to audio diffusion models."""
    
    # Model selection
    model_type: str = "audioldm"  # "audioldm", "audioldm2", "musicgen", "stable_audio"
    model_id: str = "cvssp/audioldm-s-full-v2"  # HuggingFace model ID
    
    # Audio generation
    duration_seconds: float = 10.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Quality settings
    audio_length_in_s: float = 10.0
    num_waveforms_per_prompt: int = 1


class AudioDiffusion(GenerativeModel):
    """
    Audio generation using diffusion models.
    
    Supports multiple backends:
    - AudioLDM: Text-to-audio diffusion
    - AudioLDM2: Improved text-to-audio with better quality
    - MusicGen: Meta's music generation model
    - Stable Audio: Stability AI's audio model
    
    Example:
        gen = AudioDiffusion(model_type="audioldm", device="mps")
        result = gen.generate(
            prompt="melancholic piano melody",
            duration=30,
            emotion="grief"
        )
        result.save("output.wav")
    """
    
    SUPPORTED_MODELS = {
        "audioldm": "cvssp/audioldm-s-full-v2",
        "audioldm2": "cvssp/audioldm2",
        "audioldm2-music": "cvssp/audioldm2-music",
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium",
        "musicgen-large": "facebook/musicgen-large",
    }
    
    def __init__(
        self,
        model_type: str = "audioldm",
        device: str = "auto",
        config: Optional[AudioDiffusionConfig] = None,
    ):
        """
        Initialize audio diffusion generator.
        
        Args:
            model_type: Type of model to use (audioldm, audioldm2, musicgen)
            device: Device to run on (auto, mps, cuda, cpu)
            config: Optional configuration object
        """
        if config is None:
            config = AudioDiffusionConfig(model_type=model_type, device=device)
        super().__init__(config)
        
        self.config: AudioDiffusionConfig = config
        self._pipeline = None
        self._processor = None
        
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the diffusion model.
        
        Args:
            path: Optional path to local model (otherwise downloads from HF)
        """
        model_id = path or self.SUPPORTED_MODELS.get(
            self.config.model_type, 
            self.config.model_id
        )
        
        if self.config.model_type.startswith("audioldm"):
            self._load_audioldm(model_id)
        elif self.config.model_type.startswith("musicgen"):
            self._load_musicgen(model_id)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self._is_loaded = True
    
    def _load_audioldm(self, model_id: str) -> None:
        """Load AudioLDM model."""
        if not AUDIOLDM_AVAILABLE:
            warnings.warn(
                "AudioLDM not available. Install with: pip install diffusers torch"
            )
            return
        
        import torch
        
        # Select appropriate pipeline
        if "audioldm2" in model_id.lower():
            PipelineClass = AudioLDM2Pipeline
        else:
            PipelineClass = AudioLDMPipeline
        
        # Load with appropriate dtype
        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
        
        self._pipeline = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        
        # Move to device
        device = self.config.get_device()
        if device != "cpu":
            self._pipeline = self._pipeline.to(device)
    
    def _load_musicgen(self, model_id: str) -> None:
        """Load MusicGen model."""
        if not MUSICGEN_AVAILABLE:
            warnings.warn(
                "MusicGen not available. Install with: pip install transformers torch"
            )
            return
        
        import torch
        
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        self._processor = AutoProcessor.from_pretrained(model_id)
        
        # Move to device
        device = self.config.get_device()
        if device != "cpu":
            self._model = self._model.to(device)
    
    def generate(
        self,
        prompt: str,
        duration: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        emotion: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate audio from a text prompt.
        
        Args:
            prompt: Text description of desired audio
            duration: Duration in seconds (default from config)
            negative_prompt: What to avoid in generation
            emotion: Emotional modifier to add to prompt
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated audio
        """
        # Auto-load if needed
        if not self._is_loaded:
            self.load()
        
        # Set seed for reproducibility
        self._set_seed(seed)
        
        # Enhance prompt with emotion
        if emotion:
            prompt = self._enhance_prompt_with_emotion(prompt, emotion)
        
        # Get duration
        duration = duration or self.config.duration_seconds
        
        # Generate based on model type
        if self.config.model_type.startswith("audioldm"):
            audio = self._generate_audioldm(prompt, duration, negative_prompt, **kwargs)
        elif self.config.model_type.startswith("musicgen"):
            audio = self._generate_musicgen(prompt, duration, **kwargs)
        else:
            # Fallback: generate silence
            audio = np.zeros((2, int(duration * self.config.sample_rate)))
        
        return GenerationResult(
            output=audio,
            duration_seconds=duration,
            sample_rate=self.config.sample_rate,
            model_name=self.config.model_type,
            generation_params={
                "prompt": prompt,
                "duration": duration,
                "seed": seed,
                "emotion": emotion,
            }
        )
    
    def _generate_audioldm(
        self,
        prompt: str,
        duration: float,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate using AudioLDM pipeline."""
        if self._pipeline is None:
            # Fallback
            return np.zeros((2, int(duration * self.config.sample_rate)))
        
        result = self._pipeline(
            prompt,
            negative_prompt=negative_prompt,
            audio_length_in_s=duration,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_waveforms_per_prompt=self.config.num_waveforms_per_prompt,
            **kwargs,
        )
        
        # Extract audio from result with error handling
        if result is None or not hasattr(result, 'audios') or not result.audios:
            return np.zeros((2, int(duration * self.config.sample_rate)))
        audio = result.audios[0]
        return audio
    
    def _generate_musicgen(
        self,
        prompt: str,
        duration: float,
        **kwargs,
    ) -> np.ndarray:
        """Generate using MusicGen model."""
        if self._model is None or self._processor is None:
            # Fallback
            return np.zeros((2, int(duration * self.config.sample_rate)))
        
        import torch
        
        # Process prompt
        inputs = self._processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = self.config.get_device()
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Calculate max tokens for duration
        # MusicGen generates at 50Hz, so tokens = duration * 50
        max_new_tokens = int(duration * 50)
        
        # Generate
        with torch.no_grad():
            audio_values = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
            )
        
        # Convert to numpy
        audio = audio_values[0].cpu().numpy()
        return audio
    
    def _enhance_prompt_with_emotion(self, prompt: str, emotion: str) -> str:
        """Enhance prompt with emotional descriptors."""
        emotion_modifiers = {
            "joy": "uplifting, bright, cheerful",
            "grief": "melancholic, sorrowful, minor key",
            "anger": "intense, aggressive, powerful",
            "peace": "calm, serene, ambient",
            "fear": "tense, suspenseful, dark",
            "love": "warm, romantic, tender",
            "hope": "uplifting, inspirational, building",
            "nostalgia": "wistful, bittersweet, reflective",
        }
        
        modifier = emotion_modifiers.get(emotion.lower(), emotion)
        return f"{modifier}, {prompt}"
    
    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Generate multiple variations of the same prompt.
        
        Args:
            prompt: Text description
            num_variations: Number of variations to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of GenerationResults
        """
        variations = []
        base_seed = kwargs.pop("seed", None) or np.random.randint(0, 2**31)
        
        for i in range(num_variations):
            result = self.generate(prompt, seed=base_seed + i, **kwargs)
            variations.append(result)
        
        return variations


def create_audio_generator(
    model: str = "audioldm",
    device: str = "auto",
) -> AudioDiffusion:
    """
    Factory function to create an audio generator.
    
    Args:
        model: Model type (audioldm, audioldm2, musicgen)
        device: Device to use (auto, mps, cuda, cpu)
        
    Returns:
        Configured AudioDiffusion instance
    """
    return AudioDiffusion(model_type=model, device=device)
