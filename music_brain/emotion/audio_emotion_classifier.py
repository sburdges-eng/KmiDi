"""
Audio-based Emotion Classification for KmiDi.

Integrates the ML Training Suite's trained audio classifiers with
KmiDi's emotion processing pipeline.

Usage:
    from music_brain.emotion.audio_emotion_classifier import AudioEmotionClassifier

    classifier = AudioEmotionClassifier()
    result = classifier.classify("audio.wav")
    # Returns: {"emotion": "happy", "confidence": 0.95, "valence": 0.8, "arousal": 0.7}
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# Emotion to Valence/Arousal mapping (Russell's circumplex model)
EMOTION_VA_MAP = {
    # 7-class emotions
    "angry": {"valence": -0.6, "arousal": 0.8},
    "disgust": {"valence": -0.7, "arousal": 0.3},
    "fear": {"valence": -0.6, "arousal": 0.7},
    "happy": {"valence": 0.8, "arousal": 0.6},
    "neutral": {"valence": 0.0, "arousal": 0.3},
    "sad": {"valence": -0.7, "arousal": -0.3},
    "surprise": {"valence": 0.3, "arousal": 0.8},
    # Voice types (no direct VA mapping, but useful for voice processing)
    "Alto": {"valence": 0.0, "arousal": 0.4},
    "Bass": {"valence": 0.0, "arousal": 0.3},
    "Soprano": {"valence": 0.0, "arousal": 0.6},
    "Tenor": {"valence": 0.0, "arousal": 0.5},
}


@dataclass
class AudioEmotionResult:
    """Result from audio emotion classification."""
    emotion: str
    confidence: float
    valence: float
    arousal: float
    top_predictions: List[Dict]
    model_type: str

    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "valence": self.valence,
            "arousal": self.arousal,
            "top_predictions": self.top_predictions,
            "model_type": self.model_type,
        }


class ResidualBlock(nn.Module):
    """Basic residual block (matches ML Training Suite architecture)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (named 'skip' to match training checkpoint)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out


class AudioClassifierResNet(nn.Module):
    """
    ResNet-style audio classifier (matches ML Training Suite architecture).

    This architecture matches the models trained in the ML Training Suite
    for voice type and emotion classification.
    """

    def __init__(self, num_classes: int = 7, base_channels: int = 64, dropout: float = 0.5):
        super().__init__()

        # Stem (named 'stem' to match training checkpoint)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # FC as Sequential to match checkpoint (fc.0=Flatten, fc.1=Dropout, fc.2=Linear)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, num_classes),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer."""
        layers = []
        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class AudioEmotionClassifier:
    """
    Audio emotion classifier that integrates with KmiDi's emotion system.

    Uses trained models from the ML Training Suite to classify emotions
    from audio files and convert them to valence/arousal coordinates.
    """

    CLASS_MAPPINGS = {
        "voice_type": ["Alto", "Bass", "Soprano", "Tenor"],
        "emotion_7": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    }

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "emotion_7",
        device: str = "auto",
    ):
        """
        Initialize the audio emotion classifier.

        Args:
            checkpoint_path: Path to model checkpoint. If None, uses default.
            model_type: "emotion_7" or "voice_type"
            device: "auto", "mps", "cuda", or "cpu"
        """
        # Set device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_type = model_type
        self.class_names = self.CLASS_MAPPINGS.get(model_type, [])
        self.num_classes = len(self.class_names)

        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_default_checkpoint(model_type)

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None

        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_model()

    def _find_default_checkpoint(self, model_type: str) -> Optional[str]:
        """Find the default checkpoint for a model type."""
        # Search in common locations
        # Path from emotion/audio_emotion_classifier.py -> music_brain -> KmiDi -> models
        search_paths = [
            # KmiDi/models/checkpoints
            Path(__file__).parent.parent.parent / "models" / "checkpoints",

            # parent/models/checkpoints
            Path(__file__).parent.parent.parent.parent / "models" / "checkpoints",
            Path.home() / "models" / "audio_classifiers",
            # ML Training Suite (via environment variable)
            Path(
                os.environ.get("KELLY_MODELS_PATH", "models")
            ) / "checkpoints",
        ]

        model_dirs = {
            "emotion_7": ["emotion_balanced_7class", "emotion_7class"],
            "voice_type": ["voice_type_classifier_macos", "voice_type_classifier"],
        }

        for base_path in search_paths:
            for dir_name in model_dirs.get(model_type, []):
                checkpoint = base_path / dir_name / "best.pt"
                if checkpoint.exists():
                    return str(checkpoint)

        return None

    def _load_model(self):
        """Load the model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Determine num_classes from checkpoint
        fc_keys = [k for k in state_dict.keys() if "fc" in k and "weight" in k]
        if fc_keys:
            self.num_classes = state_dict[fc_keys[-1]].shape[0]

        # Update class names if auto-detected
        if self.num_classes == 7 and not self.class_names:
            self.class_names = self.CLASS_MAPPINGS["emotion_7"]
            self.model_type = "emotion_7"
        elif self.num_classes == 4 and not self.class_names:
            self.class_names = self.CLASS_MAPPINGS["voice_type"]
            self.model_type = "voice_type"

        # Create and load model
        self.model = AudioClassifierResNet(
            num_classes=self.num_classes,
            base_channels=64,
            dropout=0.0,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for inference.

        Uses torchaudio transforms to match the training pipeline from
        the ML Training Suite.

        Args:
            audio_path: Path to audio file

        Returns:
            Mel spectrogram tensor ready for inference
        """
        import torchaudio

        # Try different loading methods
        waveform = None
        sr = None

        # Try librosa first (best compatibility)
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=5.0)
            waveform = torch.from_numpy(y).float().unsqueeze(0)
        except Exception:
            pass

        # Fallback to torchaudio
        if waveform is None:
            try:
                waveform, sr = torchaudio.load(audio_path)
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # Resample if needed
                if sr != 22050:
                    resampler = torchaudio.transforms.Resample(sr, 22050)
                    waveform = resampler(waveform)
                    sr = 22050
            except Exception as e:
                raise RuntimeError(f"Could not load audio file: {audio_path}") from e

        # Normalize (peak normalization)
        peak = torch.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak

        # Pad or trim to exact duration (5 seconds)
        target_length = int(22050 * 5.0)
        current_length = waveform.shape[1]
        if current_length > target_length:
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            padding = torch.zeros(1, target_length - current_length)
            waveform = torch.cat([waveform, padding], dim=1)

        # Extract mel spectrogram using torchaudio (matches training pipeline)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)

        # Convert to dB and normalize to [0, 1] (matches training)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

        # Already has shape [1, n_mels, time], add batch dim
        mel_tensor = mel_spec.unsqueeze(0)  # [1, 1, 128, time]
        return mel_tensor.to(self.device)

    @torch.no_grad()
    def classify(self, audio_path: str, top_k: int = 3) -> AudioEmotionResult:
        """
        Classify emotion from audio file.

        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return

        Returns:
            AudioEmotionResult with emotion, confidence, and VA coordinates
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide a valid checkpoint path.")

        mel_spec = self.preprocess_audio(audio_path)

        logits = self.model(mel_spec)
        probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            predictions.append({
                "emotion": class_name,
                "confidence": float(prob),
            })

        # Get top prediction
        top_emotion = predictions[0]["emotion"]
        top_confidence = predictions[0]["confidence"]

        # Map to valence/arousal
        va = EMOTION_VA_MAP.get(top_emotion, {"valence": 0.0, "arousal": 0.5})

        return AudioEmotionResult(
            emotion=top_emotion,
            confidence=top_confidence,
            valence=va["valence"],
            arousal=va["arousal"],
            top_predictions=predictions,
            model_type=self.model_type,
        )

    @torch.no_grad()
    def classify_batch(self, audio_paths: List[str], top_k: int = 3) -> List[AudioEmotionResult]:
        """Classify multiple audio files."""
        results = []
        for path in audio_paths:
            try:
                result = self.classify(path, top_k=top_k)
                results.append(result)
            except Exception as e:
                # Return a neutral result on error
                results.append(AudioEmotionResult(
                    emotion="neutral",
                    confidence=0.0,
                    valence=0.0,
                    arousal=0.3,
                    top_predictions=[{"emotion": "error", "confidence": 0.0, "error": str(e)}],
                    model_type=self.model_type,
                ))
        return results

    def get_valence_arousal(self, audio_path: str) -> Dict[str, float]:
        """
        Get valence/arousal coordinates from audio.

        This is the primary interface for integration with KmiDi's
        emotion-to-music mapping system.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with "valence" and "arousal" keys
        """
        result = self.classify(audio_path)
        return {
            "valence": result.valence,
            "arousal": result.arousal,
            "emotion": result.emotion,
            "confidence": result.confidence,
        }

    def is_available(self) -> bool:
        """Check if the classifier is properly loaded and available."""
        return self.model is not None

    def __repr__(self):
        status = "loaded" if self.model else "not loaded"
        return (
            f"AudioEmotionClassifier(model_type={self.model_type}, "
            f"device={self.device}, status={status})"
        )


# Convenience function for quick classification
def classify_audio_emotion(audio_path: str) -> Dict:
    """
    Quick emotion classification from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Dict with emotion, confidence, valence, arousal
    """
    classifier = AudioEmotionClassifier()
    if classifier.is_available():
        return classifier.classify(audio_path).to_dict()
    else:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "valence": 0.0,
            "arousal": 0.3,
            "error": "Model not available",
        }
