"""JEPA model and training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class JEPAConfig:
    """Base configuration shared across JEPA variants."""

    latent_dim: int = 256
    mask_ratio: float = 0.6
    mask_block_size: int = 4
    num_mask_blocks: int = 4
    device: str = "cpu"


@dataclass
class AudioJEPAConfig(JEPAConfig):
    """Audio-JEPA encoder configuration."""

    n_mels: int = 128
    max_frames: int = 512
    sample_rate: int = 22050
    hop_length: int = 512
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    tier: str = "medium"  # small | medium | large


@dataclass
class ChordJEPAConfig(JEPAConfig):
    """Chord-JEPA configuration."""

    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    seq_len: int = 64
    num_chords: int = 170
    dropout: float = 0.1


@dataclass
class StemJEPAConfig(JEPAConfig):
    """Stem-JEPA compatibility estimation configuration."""

    n_mels: int = 128
    stem_embedding_dim: int = 128
    compatibility_heads: int = 4
    max_stems: int = 8


@dataclass
class TrainingConfig:
    """Training hyperparameters for SageMaker and local runs."""

    epochs: int = 120
    batch_size: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    save_every: int = 5
    early_stop_patience: int = 15
    mixed_precision: bool = True
    gradient_clip: float = 1.0

    emotion_loss_weight: float = 0.0
    emotion_probe_checkpoint: str = ""

    # SageMaker
    instance_type: str = "ml.g5.xlarge"
    s3_output_uri: Optional[str] = None
    s3_data_uri: Optional[str] = None
    experiment_name: Optional[str] = None
