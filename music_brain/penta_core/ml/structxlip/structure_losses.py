"""
StructXLIP alignment losses.
Aligns Audio Edge Maps (onsets, flux) with structural text embeddings (e.g. "drop at beat 3").
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
) -> torch.Tensor:
    """
    Local (temporal chunk) edge-local text loss.
    (Placeholder for future implementation).
    """
    logger.warning("local_structure_loss not yet implemented, returning tensor(0.0)")
    return torch.tensor(0.0, requires_grad=True)

def consistency_edge_loss(
    audio_edge_sequence: torch.Tensor,
    audio_main_sequence: torch.Tensor,
) -> torch.Tensor:
    """
    Consistency loss between edge proxies and continuous audio encoder representation.
    (Placeholder for future implementation).
    """
    logger.warning("consistency_edge_loss not yet implemented, returning tensor(0.0)")
    return torch.tensor(0.0, requires_grad=True)
