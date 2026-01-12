"""
Tier 2: LoRA-based fine-tuning for Mac-friendly model adaptation.

LoRA (Low-Rank Adaptation) reduces fine-tuning parameters by 10-100x:
  - Memory: 16GB → 4-8GB
  - Training time: 16h → 2-4h
  - Quality loss: ~2% typical

Usage:
    from music_brain.tier2 import Tier2LORAfinetuner

    finetuner = Tier2LORAfinetuner(device="mps", lora_rank=8)
    finetuner.finetune_on_dataset(midi_paths, emotion_paths, epochs=10)
    finetuner.inference_with_lora(emotion)
"""

from .lora_finetuner import Tier2LORAfinetuner

__all__ = ["Tier2LORAfinetuner"]

__version__ = "1.0.0"
