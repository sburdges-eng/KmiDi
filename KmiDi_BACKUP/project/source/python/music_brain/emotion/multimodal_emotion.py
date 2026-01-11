"""
Multimodal Audio+Text Emotion Model

Fuses audio spectrogram features with text emotion features to produce
a unified emotional state prediction. Handles cases where only one
modality is available through modality dropout and attention gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math


@dataclass
class MultimodalEmotionOutput:
    """Output from the multimodal emotion model."""
    valence: float
    arousal: float
    base_emotion: str
    base_emotion_probs: Dict[str, float]
    intensity_tier: int
    preset: str
    confidence: float
    modality_weights: Dict[str, float]  # How much each modality contributed


class AudioEncoder(nn.Module):
    """
    CNN encoder for mel spectrogram features.

    Takes (batch, 1, n_mels, time_frames) and outputs (batch, embed_dim).
    """

    def __init__(self, n_mels: int = 64, embed_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # (1, 64, T) -> (32, 32, T/2)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # (32, 32, T/2) -> (64, 16, T/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # (64, 16, T/4) -> (128, 8, T/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # (128, 8, T/8) -> (256, 4, T/16)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, time_frames)
        Returns:
            (batch, embed_dim)
        """
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):
    """
    Encoder for pre-parsed text emotion features.

    Takes structured emotion features (valence, arousal, base_emotion one-hot,
    intensity, modifiers) and projects to embedding space.
    """

    def __init__(
        self,
        n_base_emotions: int = 6,
        n_modifiers: int = 5,
        embed_dim: int = 256,
    ):
        super().__init__()

        # Input: valence(1) + arousal(1) + base_emotion(6) + intensity(1) +
        #        blend_present(1) + modifiers(5) = 15
        input_dim = 2 + n_base_emotions + 1 + 1 + n_modifiers

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        self.n_base_emotions = n_base_emotions
        self.n_modifiers = n_modifiers
        self.embed_dim = embed_dim

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch, input_dim) structured text emotion features
        Returns:
            (batch, embed_dim)
        """
        return self.encoder(text_features)


class CrossModalAttention(nn.Module):
    """
    Cross-attention between audio and text modalities.

    Allows each modality to attend to the other for richer fusion.
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attend between modalities.

        Args:
            audio_embed: (batch, embed_dim)
            text_embed: (batch, embed_dim)
            audio_mask: (batch,) bool mask, True = valid
            text_mask: (batch,) bool mask, True = valid

        Returns:
            Tuple of (audio_enhanced, text_enhanced) embeddings
        """
        # Add sequence dimension for attention
        audio_seq = audio_embed.unsqueeze(1)  # (batch, 1, embed)
        text_seq = text_embed.unsqueeze(1)    # (batch, 1, embed)

        # Audio attends to text
        if text_mask is not None:
            key_padding_mask = ~text_mask.unsqueeze(1)  # (batch, 1)
        else:
            key_padding_mask = None

        audio_enhanced, _ = self.audio_to_text(
            audio_seq, text_seq, text_seq,
            key_padding_mask=key_padding_mask,
        )
        audio_enhanced = self.norm1(audio_seq + audio_enhanced)

        # Text attends to audio
        if audio_mask is not None:
            key_padding_mask = ~audio_mask.unsqueeze(1)
        else:
            key_padding_mask = None

        text_enhanced, _ = self.text_to_audio(
            text_seq, audio_seq, audio_seq,
            key_padding_mask=key_padding_mask,
        )
        text_enhanced = self.norm2(text_seq + text_enhanced)

        return audio_enhanced.squeeze(1), text_enhanced.squeeze(1)


class ModalityGating(nn.Module):
    """
    Learn to weight modalities based on confidence and availability.

    Handles missing modalities gracefully through learned gating.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # Gate for audio modality
        self.audio_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Gate for text modality
        self.text_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        audio_available: torch.Tensor,
        text_available: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gated fusion of modalities.

        Args:
            audio_embed: (batch, embed_dim)
            text_embed: (batch, embed_dim)
            audio_available: (batch,) bool indicating if audio is present
            text_available: (batch,) bool indicating if text is present

        Returns:
            (fused_embed, audio_weight, text_weight)
        """
        # Compute raw gates
        audio_weight = self.audio_gate(audio_embed).squeeze(-1)  # (batch,)
        text_weight = self.text_gate(text_embed).squeeze(-1)     # (batch,)

        # Zero out unavailable modalities
        audio_weight = audio_weight * audio_available.float()
        text_weight = text_weight * text_available.float()

        # Normalize weights
        total = audio_weight + text_weight + 1e-8
        audio_weight = audio_weight / total
        text_weight = text_weight / total

        # Weighted sum
        fused = (
            audio_embed * audio_weight.unsqueeze(-1) +
            text_embed * text_weight.unsqueeze(-1)
        )

        return fused, audio_weight, text_weight


class MultimodalEmotionModel(nn.Module):
    """
    Complete multimodal emotion recognition model.

    Fuses audio spectrogram and text emotion features to predict:
    - Valence (-1 to 1)
    - Arousal (0 to 1)
    - Base emotion (6 classes)
    - Preset (9 classes)
    - Intensity tier (6 classes)
    """

    BASE_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"]
    PRESETS = [
        "grief", "anxiety", "nostalgia", "calm", "uplifting",
        "aggressive", "dark", "tension_building", "neutral"
    ]

    def __init__(
        self,
        n_mels: int = 64,
        embed_dim: int = 256,
        n_heads: int = 4,
        modality_dropout: float = 0.2,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout

        # Encoders
        self.audio_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(embed_dim, n_heads)

        # Modality gating
        self.gating = ModalityGating(embed_dim)

        # Shared feature refinement
        self.shared_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.valence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.base_emotion_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.BASE_EMOTIONS)),
        )

        self.preset_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.PRESETS)),
        )

        self.intensity_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        audio_available: Optional[torch.Tensor] = None,
        text_available: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional modalities.

        Args:
            audio: (batch, 1, n_mels, time) mel spectrogram, or None
            text_features: (batch, 15) parsed text features, or None
            audio_available: (batch,) bool mask for audio availability
            text_available: (batch,) bool mask for text availability

        Returns:
            Dict with valence, arousal, base_emotion, preset, intensity, weights
        """
        batch_size = audio.shape[0] if audio is not None else text_features.shape[0]
        device = audio.device if audio is not None else text_features.device

        # Default availability masks
        if audio_available is None:
            audio_available = torch.ones(batch_size, dtype=torch.bool, device=device)
        if text_available is None:
            text_available = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Apply modality dropout during training
        if self.training and self.modality_dropout > 0:
            audio_drop = torch.rand(batch_size, device=device) < self.modality_dropout
            text_drop = torch.rand(batch_size, device=device) < self.modality_dropout
            # Ensure at least one modality is available
            both_drop = audio_drop & text_drop
            audio_drop = audio_drop & ~both_drop
            audio_available = audio_available & ~audio_drop
            text_available = text_available & ~text_drop

        # Encode modalities
        if audio is not None:
            audio_embed = self.audio_encoder(audio)
        else:
            audio_embed = torch.zeros(batch_size, self.embed_dim, device=device)
            audio_available = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if text_features is not None:
            text_embed = self.text_encoder(text_features)
        else:
            text_embed = torch.zeros(batch_size, self.embed_dim, device=device)
            text_available = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Cross-modal attention (only when both available)
        both_available = audio_available & text_available
        if both_available.any():
            audio_enhanced, text_enhanced = self.cross_attention(
                audio_embed, text_embed,
                audio_available, text_available,
            )
            # Blend enhanced embeddings where both available
            audio_embed = torch.where(
                both_available.unsqueeze(-1),
                audio_enhanced,
                audio_embed,
            )
            text_embed = torch.where(
                both_available.unsqueeze(-1),
                text_enhanced,
                text_embed,
            )

        # Gated fusion
        fused, audio_weight, text_weight = self.gating(
            audio_embed, text_embed,
            audio_available, text_available,
        )

        # Shared refinement
        features = self.shared_layers(fused)

        # Task heads
        valence = self.valence_head(features).squeeze(-1)
        arousal = self.arousal_head(features).squeeze(-1)
        base_emotion = self.base_emotion_head(features)
        preset = self.preset_head(features)
        intensity = self.intensity_head(features)

        return {
            "valence": valence,
            "arousal": arousal,
            "base_emotion": base_emotion,
            "preset": preset,
            "intensity": intensity,
            "audio_weight": audio_weight,
            "text_weight": text_weight,
            "features": features,
        }

    def predict(
        self,
        audio: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> List[MultimodalEmotionOutput]:
        """
        Predict emotions with human-readable output.

        Args:
            audio: (batch, 1, n_mels, time) or None
            text_features: (batch, 15) or None

        Returns:
            List of MultimodalEmotionOutput for each batch item
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio, text_features)

        batch_size = outputs["valence"].shape[0]
        results = []

        for i in range(batch_size):
            # Get predictions
            valence = outputs["valence"][i].item()
            arousal = outputs["arousal"][i].item()

            base_probs = F.softmax(outputs["base_emotion"][i], dim=-1)
            base_idx = base_probs.argmax().item()
            base_emotion = self.BASE_EMOTIONS[base_idx]

            preset_probs = F.softmax(outputs["preset"][i], dim=-1)
            preset_idx = preset_probs.argmax().item()
            preset = self.PRESETS[preset_idx]

            intensity_probs = F.softmax(outputs["intensity"][i], dim=-1)
            intensity_tier = intensity_probs.argmax().item() + 1

            # Confidence based on top probability
            confidence = max(
                base_probs.max().item(),
                preset_probs.max().item(),
            )

            results.append(MultimodalEmotionOutput(
                valence=valence,
                arousal=arousal,
                base_emotion=base_emotion,
                base_emotion_probs={
                    e: base_probs[j].item()
                    for j, e in enumerate(self.BASE_EMOTIONS)
                },
                intensity_tier=intensity_tier,
                preset=preset,
                confidence=confidence,
                modality_weights={
                    "audio": outputs["audio_weight"][i].item(),
                    "text": outputs["text_weight"][i].item(),
                },
            ))

        return results


def prepare_text_features(parsed_emotion) -> torch.Tensor:
    """
    Convert ParsedEmotion to tensor features for the text encoder.

    Args:
        parsed_emotion: ParsedEmotion from TextEmotionParser

    Returns:
        (15,) tensor of features
    """
    BASE_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"]
    MODIFIERS = ["ptsd_intrusion", "dissociation", "misdirection", "suppressed", "cathartic_release"]

    features = []

    # Valence and arousal
    features.append(parsed_emotion.valence)
    features.append(parsed_emotion.arousal)

    # Base emotion one-hot
    base_onehot = [0.0] * 6
    if parsed_emotion.base_emotion.upper() in BASE_EMOTIONS:
        idx = BASE_EMOTIONS.index(parsed_emotion.base_emotion.upper())
        base_onehot[idx] = 1.0
    features.extend(base_onehot)

    # Intensity (normalized to 0-1)
    features.append(parsed_emotion.intensity_tier / 6.0)

    # Blend present
    features.append(1.0 if parsed_emotion.blend_components else 0.0)

    # Modifiers one-hot
    for mod in MODIFIERS:
        features.append(1.0 if mod in parsed_emotion.modifiers else 0.0)

    return torch.tensor(features, dtype=torch.float32)


class MultimodalEmotionTrainer:
    """Training utilities for the multimodal model."""

    def __init__(
        self,
        model: MultimodalEmotionModel,
        learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def train_step(
        self,
        audio: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            audio: (batch, 1, n_mels, time) or None
            text_features: (batch, 15) or None
            targets: Dict with valence, arousal, base_emotion, preset, intensity

        Returns:
            Dict of losses
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        if audio is not None:
            audio = audio.to(self.device)
        if text_features is not None:
            text_features = text_features.to(self.device)

        targets = {k: v.to(self.device) for k, v in targets.items()}

        # Forward
        outputs = self.model(audio, text_features)

        # Compute losses
        losses = {}
        total_loss = 0.0

        if "valence" in targets:
            loss = self.mse_loss(outputs["valence"], targets["valence"])
            losses["valence"] = loss.item()
            total_loss += loss

        if "arousal" in targets:
            loss = self.mse_loss(outputs["arousal"], targets["arousal"])
            losses["arousal"] = loss.item()
            total_loss += loss

        if "base_emotion" in targets:
            loss = self.ce_loss(outputs["base_emotion"], targets["base_emotion"])
            losses["base_emotion"] = loss.item()
            total_loss += loss

        if "preset" in targets:
            loss = self.ce_loss(outputs["preset"], targets["preset"])
            losses["preset"] = loss.item()
            total_loss += loss

        if "intensity" in targets:
            loss = self.ce_loss(outputs["intensity"], targets["intensity"])
            losses["intensity"] = loss.item()
            total_loss += loss

        # Backward
        total_loss.backward()
        self.optimizer.step()

        losses["total"] = total_loss.item()
        return losses


if __name__ == "__main__":
    print("Multimodal Emotion Model Demo")
    print("=" * 50)

    # Create model
    model = MultimodalEmotionModel(n_mels=64, embed_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Test with audio only
    print("\n--- Audio Only ---")
    audio = torch.randn(2, 1, 64, 128)  # batch=2, 64 mels, 128 time frames
    results = model.predict(audio=audio)
    for i, r in enumerate(results):
        print(f"  Sample {i}: {r.base_emotion} (V={r.valence:.2f}, A={r.arousal:.2f})")
        print(f"    Weights: audio={r.modality_weights['audio']:.2f}, text={r.modality_weights['text']:.2f}")

    # Test with text only
    print("\n--- Text Only ---")
    text_features = torch.randn(2, 15)  # batch=2, 15 features
    results = model.predict(text_features=text_features)
    for i, r in enumerate(results):
        print(f"  Sample {i}: {r.base_emotion} (V={r.valence:.2f}, A={r.arousal:.2f})")
        print(f"    Weights: audio={r.modality_weights['audio']:.2f}, text={r.modality_weights['text']:.2f}")

    # Test with both
    print("\n--- Both Modalities ---")
    results = model.predict(audio=audio, text_features=text_features)
    for i, r in enumerate(results):
        print(f"  Sample {i}: {r.base_emotion} (V={r.valence:.2f}, A={r.arousal:.2f})")
        print(f"    Weights: audio={r.modality_weights['audio']:.2f}, text={r.modality_weights['text']:.2f}")

    print("\nModel structure:")
    print(f"  Audio encoder: {sum(p.numel() for p in model.audio_encoder.parameters()):,} params")
    print(f"  Text encoder: {sum(p.numel() for p in model.text_encoder.parameters()):,} params")
    print(f"  Cross attention: {sum(p.numel() for p in model.cross_attention.parameters()):,} params")
    print(f"  Gating: {sum(p.numel() for p in model.gating.parameters()):,} params")
