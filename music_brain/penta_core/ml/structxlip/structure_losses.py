"""
StructXLIP alignment losses.
Aligns Audio Edge Maps (onsets, flux) with structural text embeddings (e.g. "drop at beat 3").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def global_structure_loss(
    audio_edge_features: torch.Tensor,
    text_structure_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Global structure-text alignment loss (Contrastive InfoNCE).

    Args:
        audio_edge_features: (batch_size, embed_dim) - pooled feature from Audio Edge Maps.
        text_structure_features: (batch_size, embed_dim) - text embedding of structural prompt.
        temperature: Softmax temperature for contrastive loss.

    Returns:
        Scalar contrastive loss.
    """
    # L2 normalize
    audio_norm = F.normalize(audio_edge_features, p=2, dim=-1)
    text_norm = F.normalize(text_structure_features, p=2, dim=-1)

    # Cosine similarity logits
    logits = torch.matmul(audio_norm, text_norm.transpose(0, 1)) / temperature

    # Target is diagonal (batch index)
    batch_size = logits.shape[0]
    targets = torch.arange(batch_size, device=logits.device)

    # Symmetric loss
    loss_a2t = F.cross_entropy(logits, targets)
    loss_t2a = F.cross_entropy(logits.transpose(0, 1), targets)

    return (loss_a2t + loss_t2a) / 2


def local_structure_loss(
    audio_edge_sequence: torch.Tensor,
    text_chunk_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Local (temporal chunk) edge–text alignment loss.

    Aligns temporal chunks of audio edge features with corresponding
    text descriptions. Each chunk in the sequence is matched to the
    text embedding at the same temporal position.

    Args:
        audio_edge_sequence: (batch, num_chunks, embed_dim) — edge features per chunk.
        text_chunk_features: (batch, num_chunks, embed_dim) — text embeddings per chunk.
        temperature: Softmax temperature.

    Returns:
        Scalar loss (mean over batch and chunks).
    """
    audio_norm = F.normalize(audio_edge_sequence, p=2, dim=-1)
    text_norm = F.normalize(text_chunk_features, p=2, dim=-1)

    # Per-chunk cosine similarity: (batch, num_chunks)
    similarity = (audio_norm * text_norm).sum(dim=-1) / temperature

    # Target: maximize similarity at each position (sigmoid BCE with target=1)
    loss = F.binary_cross_entropy_with_logits(
        similarity,
        torch.ones_like(similarity),
    )
    return loss


def consistency_edge_loss(
    audio_edge_sequence: torch.Tensor,
    audio_main_sequence: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Consistency loss between edge proxies and continuous audio encoder.

    Encourages the edge-based representation to stay close to the main
    audio encoder's representation, acting as a regularizer to prevent
    the edge proxy from drifting into a degenerate space.

    Args:
        audio_edge_sequence: (batch, seq_len, embed_dim) — from edge map encoder.
        audio_main_sequence: (batch, seq_len, embed_dim) — from main audio encoder.
        margin: Minimum cosine similarity before penalty applies.

    Returns:
        Scalar loss.
    """
    edge_norm = F.normalize(audio_edge_sequence, p=2, dim=-1)
    main_norm = F.normalize(audio_main_sequence, p=2, dim=-1)

    # Cosine similarity per position: (batch, seq_len)
    cos_sim = (edge_norm * main_norm).sum(dim=-1)

    # Hinge-style: penalize when similarity drops below margin
    loss = F.relu(margin - cos_sim).mean()
    return loss
