"""
Custom Music-Aware Loss Functions for Neural Network Training.

Provides specialized loss functions that incorporate music theory knowledge:
- Harmonic loss (penalizes dissonance)
- Contour loss (for melodic shape preservation)
- Emotion-aware loss (valence/arousal weighted)
- Temporal coherence loss (for sequences)
- Perceptual loss (based on human perception)
- Voice leading loss (smooth transitions)
- Groove loss (timing pattern consistency)

Usage:
    from penta_core.ml.losses import (
        HarmonicLoss, EmotionAwareLoss, TemporalCoherenceLoss
    )

    # In training loop
    harmonic_loss = HarmonicLoss(weight=0.3)
    emotion_loss = EmotionAwareLoss(valence_weight=0.5, arousal_weight=0.5)

    # Combine losses
    total_loss = base_loss + harmonic_loss(pred, target) + emotion_loss(pred, target, metadata)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Music Theory Constants
# =============================================================================

# Consonance/dissonance scores for intervals (semitones)
# Higher score = more consonant
INTERVAL_CONSONANCE = {
    0: 1.0,   # Unison (perfect)
    1: 0.1,   # Minor 2nd (very dissonant)
    2: 0.3,   # Major 2nd (dissonant)
    3: 0.5,   # Minor 3rd (consonant)
    4: 0.7,   # Major 3rd (consonant)
    5: 0.6,   # Perfect 4th (consonant)
    6: 0.2,   # Tritone (very dissonant)
    7: 0.9,   # Perfect 5th (very consonant)
    8: 0.7,   # Minor 6th (consonant)
    9: 0.7,   # Major 6th (consonant)
    10: 0.4,  # Minor 7th (dissonant)
    11: 0.3,  # Major 7th (dissonant)
    12: 1.0,  # Octave (perfect)
}

# Voice leading penalties (semitones)
# Lower is better for smooth voice leading
VOICE_LEADING_PENALTY = {
    0: 0.0,   # No movement (good)
    1: 0.2,   # Semitone (good)
    2: 0.3,   # Whole tone (good)
    3: 0.5,   # Minor 3rd (ok)
    4: 0.6,   # Major 3rd (ok)
    5: 0.8,   # Perfect 4th (larger leap)
    6: 1.0,   # Tritone (awkward)
    7: 0.9,   # Perfect 5th (larger leap)
    8: 1.1,   # Minor 6th (large leap)
    9: 1.2,   # Major 6th (large leap)
    10: 1.3,  # Minor 7th (very large)
    11: 1.4,  # Major 7th (very large)
    12: 1.5,  # Octave (very large)
}


# =============================================================================
# Base Loss Classes
# =============================================================================


class MusicTheoryLoss:
    """Base class for music theory-aware losses."""

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        """
        Initialize loss.

        Args:
            weight: Weight multiplier for this loss
            reduction: How to reduce batch losses ("mean", "sum", "none")
        """
        self.weight = weight
        self.reduction = reduction

    def __call__(self, *args, **kwargs):
        """Compute loss (to be implemented by subclasses)."""
        raise NotImplementedError

    def _reduce(self, losses: np.ndarray) -> Union[float, np.ndarray]:
        """Apply reduction to losses."""
        if self.reduction == "mean":
            return np.mean(losses)
        elif self.reduction == "sum":
            return np.sum(losses)
        else:  # "none"
            return losses


# =============================================================================
# Harmonic Losses
# =============================================================================


class HarmonicLoss(MusicTheoryLoss):
    """
    Loss that penalizes harmonically unstable predictions.

    Encourages consonant intervals and penalizes dissonance.
    """

    def __init__(
        self,
        weight: float = 1.0,
        consonance_target: float = 0.7,
        reduction: str = "mean",
    ):
        """
        Initialize harmonic loss.

        Args:
            weight: Weight for this loss
            consonance_target: Target consonance score (0-1)
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.consonance_target = consonance_target

    def __call__(
        self,
        predicted_notes: np.ndarray,
        target_notes: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute harmonic loss.

        Args:
            predicted_notes: Predicted note pitches (batch, seq_len, pitches)
            target_notes: Optional target notes for comparison
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        losses = []

        # Ensure 2D shape (batch, notes)
        if predicted_notes.ndim == 1:
            predicted_notes = predicted_notes.reshape(1, -1)

        for batch in predicted_notes:
            # Get active notes (assuming threshold or one-hot)
            active_pitches = self._get_active_pitches(batch)

            if len(active_pitches) < 2:
                # Single note or silence - no harmonic loss
                losses.append(0.0)
                continue

            # Compute average consonance
            consonance = self._compute_consonance(active_pitches)

            # Loss is deviation from target consonance
            loss = abs(consonance - self.consonance_target)
            losses.append(loss)

        losses = np.array(losses)
        return self.weight * self._reduce(losses)

    def _get_active_pitches(self, notes: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Extract active pitch indices from note array."""
        if notes.ndim == 1:
            # One-hot or activation vector
            active = np.where(notes > threshold)[0]
        else:
            # Multi-dimensional - take argmax
            active = [np.argmax(notes)]

        return list(active)

    def _compute_consonance(self, pitches: List[int]) -> float:
        """
        Compute average consonance score for a set of pitches.

        Args:
            pitches: List of MIDI pitch values

        Returns:
            Consonance score (0-1, higher = more consonant)
        """
        if len(pitches) < 2:
            return 1.0

        consonance_scores = []

        # Check all pairs of pitches
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                interval = abs(pitches[j] - pitches[i]) % 12
                score = INTERVAL_CONSONANCE.get(interval, 0.5)
                consonance_scores.append(score)

        return np.mean(consonance_scores)


class ChordProgressionLoss(MusicTheoryLoss):
    """
    Loss for chord progression prediction.

    Penalizes unlikely chord transitions based on music theory.
    """

    def __init__(
        self,
        weight: float = 1.0,
        transition_matrix: Optional[np.ndarray] = None,
        reduction: str = "mean",
    ):
        """
        Initialize chord progression loss.

        Args:
            weight: Weight for this loss
            transition_matrix: (num_chords, num_chords) transition probabilities
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.transition_matrix = transition_matrix

        if self.transition_matrix is None:
            # Use default music theory transition probabilities
            self.transition_matrix = self._get_default_transitions()

    def __call__(
        self,
        predicted_chords: np.ndarray,
        target_chords: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute chord progression loss.

        Args:
            predicted_chords: Predicted chord sequence (batch, seq_len, num_chords)
            target_chords: Optional target chord sequence
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        losses = []

        # Get chord indices
        if predicted_chords.ndim == 3:
            # Soft probabilities - use argmax
            pred_indices = np.argmax(predicted_chords, axis=-1)
        else:
            pred_indices = predicted_chords.astype(int)

        for sequence in pred_indices:
            # Compute transition loss for this sequence
            seq_loss = 0.0
            for i in range(len(sequence) - 1):
                from_chord = sequence[i]
                to_chord = sequence[i + 1]

                # Get transition probability
                prob = self.transition_matrix[from_chord, to_chord]

                # Negative log likelihood
                seq_loss += -np.log(prob + 1e-8)

            losses.append(seq_loss / max(1, len(sequence) - 1))

        losses = np.array(losses)
        return self.weight * self._reduce(losses)

    def _get_default_transitions(self, num_chords: int = 48) -> np.ndarray:
        """
        Get default chord transition matrix based on music theory.

        Simplified version - real implementation would use actual theory.
        """
        # Initialize with small uniform probability
        matrix = np.ones((num_chords, num_chords)) * 0.01

        # Common progressions (I-IV-V-I pattern)
        for i in range(num_chords):
            # Higher probability for V-I (dominant to tonic)
            matrix[i, (i + 7) % num_chords] = 0.3

            # Higher probability for IV-I (subdominant to tonic)
            matrix[i, (i + 5) % num_chords] = 0.2

            # Same chord repetition
            matrix[i, i] = 0.15

        # Normalize rows to sum to 1
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        return matrix


class VoiceLeadingLoss(MusicTheoryLoss):
    """
    Loss that encourages smooth voice leading.

    Penalizes large melodic leaps and parallel fifths/octaves.
    """

    def __init__(
        self,
        weight: float = 1.0,
        max_leap: int = 7,  # semitones
        penalize_parallels: bool = True,
        reduction: str = "mean",
    ):
        """
        Initialize voice leading loss.

        Args:
            weight: Weight for this loss
            max_leap: Maximum acceptable leap (semitones)
            penalize_parallels: If True, penalize parallel fifths/octaves
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.max_leap = max_leap
        self.penalize_parallels = penalize_parallels

    def __call__(
        self,
        predicted_melody: np.ndarray,
        target_melody: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute voice leading loss.

        Args:
            predicted_melody: Predicted melody (batch, seq_len) in MIDI pitch
            target_melody: Optional target melody
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        losses = []

        # Ensure 2D
        if predicted_melody.ndim == 1:
            predicted_melody = predicted_melody.reshape(1, -1)

        for melody in predicted_melody:
            if len(melody) < 2:
                losses.append(0.0)
                continue

            # Compute intervallic penalties
            intervals = np.diff(melody.astype(int))
            penalties = []

            for interval in intervals:
                interval_size = abs(interval)

                # Get penalty from lookup table
                penalty = VOICE_LEADING_PENALTY.get(
                    min(interval_size, 12),
                    1.5,  # Very large leap
                )

                # Extra penalty for leaps larger than max_leap
                if interval_size > self.max_leap:
                    penalty *= 1.5

                penalties.append(penalty)

            # Parallel motion penalty
            if self.penalize_parallels and len(intervals) >= 2:
                for i in range(len(intervals) - 1):
                    int1 = intervals[i]
                    int2 = intervals[i + 1]

                    # Check for parallel perfect intervals (same direction)
                    if abs(int1) in [5, 7, 12] and int1 == int2:
                        penalties.append(2.0)  # Heavy penalty

            losses.append(np.mean(penalties) if penalties else 0.0)

        losses = np.array(losses)
        return self.weight * self._reduce(losses)


# =============================================================================
# Emotion-Aware Losses
# =============================================================================


class EmotionAwareLoss(MusicTheoryLoss):
    """
    Loss that incorporates emotional metadata.

    Weights predictions based on valence/arousal space.
    """

    def __init__(
        self,
        weight: float = 1.0,
        valence_weight: float = 0.5,
        arousal_weight: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Initialize emotion-aware loss.

        Args:
            weight: Overall weight for this loss
            valence_weight: Weight for valence dimension
            arousal_weight: Weight for arousal dimension
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight

    def __call__(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> float:
        """
        Compute emotion-aware loss.

        Args:
            predicted: Predicted values
            target: Target values
            metadata: Dict with 'valence' and 'arousal' keys
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        # Base MSE loss
        base_loss = np.mean((predicted - target) ** 2, axis=-1)

        if metadata is None or ("valence" not in metadata and "arousal" not in metadata):
            # No metadata - return base loss
            return self.weight * self._reduce(base_loss)

        # Get emotion values
        valence = metadata.get("valence", 0.0)  # -1 to 1
        arousal = metadata.get("arousal", 0.0)  # -1 to 1

        # Compute emotion weights
        # Higher arousal = more sensitive to errors
        arousal_factor = 1.0 + abs(arousal) * self.arousal_weight

        # Extreme valence = more sensitive
        valence_factor = 1.0 + abs(valence) * self.valence_weight

        # Combine weights
        emotion_weight = arousal_factor * valence_factor

        # Apply emotion weighting
        weighted_loss = base_loss * emotion_weight

        return self.weight * self._reduce(weighted_loss)


class ContrastiveLoss(MusicTheoryLoss):
    """
    Contrastive loss for emotion/style classification.

    Pulls similar samples together, pushes dissimilar samples apart.
    """

    def __init__(
        self,
        weight: float = 1.0,
        margin: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Initialize contrastive loss.

        Args:
            weight: Weight for this loss
            margin: Margin for dissimilar pairs
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.margin = margin

    def __call__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Compute contrastive loss.

        Args:
            embeddings: Embedding vectors (batch, embedding_dim)
            labels: Class labels (batch,)
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        losses = []
        n = len(embeddings)

        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance
                dist = np.linalg.norm(embeddings[i] - embeddings[j])

                # Same class: minimize distance
                if labels[i] == labels[j]:
                    loss = dist ** 2
                # Different class: maximize distance up to margin
                else:
                    loss = max(0, self.margin - dist) ** 2

                losses.append(loss)

        if not losses:
            return 0.0

        losses = np.array(losses)
        return self.weight * self._reduce(losses)


# =============================================================================
# Temporal Losses
# =============================================================================


class TemporalCoherenceLoss(MusicTheoryLoss):
    """
    Loss that encourages temporal coherence in sequences.

    Penalizes abrupt changes and encourages smooth transitions.
    """

    def __init__(
        self,
        weight: float = 1.0,
        smoothness_weight: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Initialize temporal coherence loss.

        Args:
            weight: Weight for this loss
            smoothness_weight: Weight for smoothness penalty
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.smoothness_weight = smoothness_weight

    def __call__(
        self,
        predicted_sequence: np.ndarray,
        target_sequence: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute temporal coherence loss.

        Args:
            predicted_sequence: Predicted sequence (batch, seq_len, features)
            target_sequence: Optional target sequence
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        losses = []

        # Ensure 3D
        if predicted_sequence.ndim == 2:
            predicted_sequence = predicted_sequence.reshape(1, *predicted_sequence.shape)

        for sequence in predicted_sequence:
            if len(sequence) < 2:
                losses.append(0.0)
                continue

            # Compute first-order differences (velocity)
            first_diff = np.diff(sequence, axis=0)
            velocity_penalty = np.mean(first_diff ** 2)

            # Compute second-order differences (acceleration)
            second_diff = np.diff(first_diff, axis=0)
            acceleration_penalty = np.mean(second_diff ** 2)

            # Combined loss
            loss = velocity_penalty + self.smoothness_weight * acceleration_penalty
            losses.append(loss)

        losses = np.array(losses)
        return self.weight * self._reduce(losses)


class RhythmicRegularityLoss(MusicTheoryLoss):
    """
    Loss that encourages rhythmic regularity.

    Penalizes irregular timing patterns.
    """

    def __init__(
        self,
        weight: float = 1.0,
        target_periodicity: Optional[float] = None,
        reduction: str = "mean",
    ):
        """
        Initialize rhythmic regularity loss.

        Args:
            weight: Weight for this loss
            target_periodicity: Target period in time steps (None = auto-detect)
            reduction: Reduction method
        """
        super().__init__(weight, reduction)
        self.target_periodicity = target_periodicity

    def __call__(
        self,
        timing_data: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Compute rhythmic regularity loss.

        Args:
            timing_data: Onset times (batch, num_onsets)
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        losses = []

        # Ensure 2D
        if timing_data.ndim == 1:
            timing_data = timing_data.reshape(1, -1)

        for onsets in timing_data:
            if len(onsets) < 3:
                losses.append(0.0)
                continue

            # Compute inter-onset intervals (IOIs)
            iois = np.diff(onsets)

            if self.target_periodicity is None:
                # Auto-detect periodicity (median IOI)
                target = np.median(iois)
            else:
                target = self.target_periodicity

            # Penalize deviation from target period
            deviations = (iois - target) ** 2
            loss = np.mean(deviations)

            losses.append(loss)

        losses = np.array(losses)
        return self.weight * self._reduce(losses)


# =============================================================================
# Perceptual Losses
# =============================================================================


class SpectralConvergenceLoss(MusicTheoryLoss):
    """
    Spectral convergence loss for audio generation.

    Measures similarity in frequency domain.
    """

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        """Initialize spectral convergence loss."""
        super().__init__(weight, reduction)

    def __call__(
        self,
        predicted_spec: np.ndarray,
        target_spec: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Compute spectral convergence.

        Args:
            predicted_spec: Predicted spectrogram
            target_spec: Target spectrogram
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        # Frobenius norm of difference / Frobenius norm of target
        numerator = np.linalg.norm(target_spec - predicted_spec, ord="fro", axis=(-2, -1))
        denominator = np.linalg.norm(target_spec, ord="fro", axis=(-2, -1))

        losses = numerator / (denominator + 1e-8)
        return self.weight * self._reduce(losses)


class LogSTFTMagnitudeLoss(MusicTheoryLoss):
    """
    Log-magnitude STFT loss for audio generation.

    More perceptually relevant than raw magnitude.
    """

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        """Initialize log-STFT magnitude loss."""
        super().__init__(weight, reduction)

    def __call__(
        self,
        predicted_spec: np.ndarray,
        target_spec: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Compute log-magnitude STFT loss.

        Args:
            predicted_spec: Predicted spectrogram
            target_spec: Target spectrogram
            **kwargs: Additional parameters

        Returns:
            Loss value
        """
        # Log magnitude
        log_pred = np.log(np.abs(predicted_spec) + 1e-8)
        log_target = np.log(np.abs(target_spec) + 1e-8)

        # L1 distance
        losses = np.mean(np.abs(log_pred - log_target), axis=(-2, -1))

        return self.weight * self._reduce(losses)


# =============================================================================
# Combined Loss
# =============================================================================


class CombinedMusicLoss:
    """
    Combines multiple music-aware losses.

    Provides easy interface for multi-objective training.
    """

    def __init__(self, losses: Dict[str, Tuple[MusicTheoryLoss, float]]):
        """
        Initialize combined loss.

        Args:
            losses: Dict of {name: (loss_fn, weight)} pairs
        """
        self.losses = losses

    def __call__(self, *args, **kwargs) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        total_loss = 0.0
        individual_losses = {}

        for name, (loss_fn, weight) in self.losses.items():
            try:
                loss_value = loss_fn(*args, **kwargs)
                weighted_loss = weight * loss_value
                total_loss += weighted_loss
                individual_losses[name] = float(loss_value)
            except Exception as e:
                logger.warning(f"Loss {name} failed: {e}")
                individual_losses[name] = 0.0

        return total_loss, individual_losses


# =============================================================================
# PyTorch Wrappers (if torch available)
# =============================================================================


try:
    import torch
    import torch.nn as nn

    class TorchHarmonicLoss(nn.Module):
        """PyTorch wrapper for HarmonicLoss."""

        def __init__(self, weight: float = 1.0, consonance_target: float = 0.7):
            super().__init__()
            self.loss_fn = HarmonicLoss(weight, consonance_target)

        def forward(self, predicted, target=None):
            # Convert to numpy
            pred_np = predicted.detach().cpu().numpy()
            loss = self.loss_fn(pred_np, None if target is None else target.detach().cpu().numpy())
            return torch.tensor(loss, device=predicted.device, dtype=predicted.dtype)

    class TorchEmotionAwareLoss(nn.Module):
        """PyTorch wrapper for EmotionAwareLoss."""

        def __init__(self, weight: float = 1.0, valence_weight: float = 0.5, arousal_weight: float = 0.5):
            super().__init__()
            self.loss_fn = EmotionAwareLoss(weight, valence_weight, arousal_weight)

        def forward(self, predicted, target, metadata=None):
            pred_np = predicted.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            loss = self.loss_fn(pred_np, target_np, metadata)
            return torch.tensor(loss, device=predicted.device, dtype=predicted.dtype)

except ImportError:
    # PyTorch not available
    pass
