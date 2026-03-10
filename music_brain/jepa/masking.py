"""Multi-block masking for JEPA self-supervised learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class MaskResult:
    """Result of a masking operation."""

    masked: torch.Tensor
    mask: torch.Tensor
    target_indices: torch.Tensor


class MultiBlockMasking(nn.Module):
    """
    Multi-block masking strategy for JEPA training.

    Masks contiguous blocks of the latent sequence so the predictor
    must reconstruct them from context — forcing the encoder to learn
    globally coherent representations.
    """

    def __init__(
        self,
        mask_ratio: float = 0.6,
        block_size: int = 4,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.num_blocks = num_blocks

    def forward(self, z: torch.Tensor) -> MaskResult:
        """
        Apply multi-block masking to latent representations.

        Args:
            z: Latent tensor of shape (B, T, D) or (B, C, H, W)

        Returns:
            MaskResult with masked tensor, mask, and target indices
        """
        return _mask_latents_impl(
            z,
            mask_ratio=self.mask_ratio,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
        )


def mask_latents(
    z: torch.Tensor,
    mask_ratio: float = 0.6,
    block_size: int = 4,
    num_blocks: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply multi-block masking to latent representations.

    Compatible with the ``final_kel`` training scripts which expect
    ``(masked_tensor, mask)`` as the return tuple.

    Args:
        z: Latent tensor. Supports:
           - 3-D ``(B, T, D)``  – sequence latents
           - 4-D ``(B, C, H, W)`` – spatial latents (mask along W)

    Returns:
        Tuple of (masked_z, mask) where mask is a boolean tensor
        indicating which positions were zeroed out.
    """
    result = _mask_latents_impl(z, mask_ratio, block_size, num_blocks)
    return result.masked, result.mask


def _mask_latents_impl(
    z: torch.Tensor,
    mask_ratio: float = 0.6,
    block_size: int = 4,
    num_blocks: int = 4,
) -> MaskResult:
    is_4d = z.dim() == 4
    if is_4d:
        B, C, H, W = z.shape
        seq_len = W
    else:
        B, seq_len = z.shape[0], z.shape[1]

    num_to_mask = max(1, int(seq_len * mask_ratio))
    clamped_block = min(block_size, seq_len)

    mask = torch.zeros(B, seq_len, dtype=torch.bool, device=z.device)

    for b in range(B):
        masked_count = 0
        attempts = 0
        while masked_count < num_to_mask and attempts < num_blocks * 3:
            start = torch.randint(0, max(1, seq_len - clamped_block + 1), (1,)).item()
            end = min(start + clamped_block, seq_len)
            mask[b, start:end] = True
            masked_count = mask[b].sum().item()
            attempts += 1

    target_indices = mask.nonzero(as_tuple=False)

    if is_4d:
        expand = mask.unsqueeze(1).unsqueeze(2).expand_as(z)
        masked = z.clone()
        masked[expand] = 0.0
    else:
        expand = mask.unsqueeze(-1).expand_as(z)
        masked = z.clone()
        masked[expand] = 0.0

    return MaskResult(masked=masked, mask=mask, target_indices=target_indices)
