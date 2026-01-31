"""LoRA (Low-Rank Adaptation) implementation for audio models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear layer.

    LoRA decomposes weight updates into two low-rank matrices:
    W' = W + BA where B is (out_features, rank) and A is (rank, in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with kaiming, B with zeros (so initial output is zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: scaling * (x @ A^T @ B^T)"""
        return self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    def merge_weights(self):
        """Merge LoRA weights into the main linear layer."""
        with torch.no_grad():
            self.linear.weight.data += (
                self.lora.scaling * self.lora.lora_B @ self.lora.lora_A
            )
        return self.linear


class LoRAConv2d(nn.Module):
    """Conv2d layer with LoRA adaptation (applied to 1x1 convolutions)."""

    def __init__(
        self,
        conv: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = conv

        # LoRA for conv is applied as if it were a linear layer
        # Works best for 1x1 convolutions or by reshaping
        in_features = conv.in_channels
        out_features = conv.out_channels

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original conv output
        out = self.conv(x)

        # LoRA path: global average pool -> linear -> broadcast back
        # This is a simplified LoRA for conv that modulates channel-wise
        b, c, h, w = x.shape
        x_pooled = x.mean(dim=(2, 3))  # [B, C_in]
        lora_out = self.scaling * (self.dropout(x_pooled) @ self.lora_A.T @ self.lora_B.T)
        lora_out = lora_out.view(b, -1, 1, 1)  # [B, C_out, 1, 1]

        return out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.

    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout for LoRA layers
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all Linear layers in 'fc' modules.

    Returns:
        Model with LoRA layers applied
    """
    if target_modules is None:
        target_modules = ["fc"]

    modules_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should get LoRA
            if any(target in name for target in target_modules):
                modules_to_replace.append((name, module))

    # Replace modules
    for name, module in modules_to_replace:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        # Replace the module
        attr_name = parts[-1]
        if attr_name.isdigit():
            parent[int(attr_name)] = LoRALinear(module, rank, alpha, dropout)
        else:
            setattr(parent, attr_name, LoRALinear(module, rank, alpha, dropout))

    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get only the LoRA parameters for training."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights."""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.clone()
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu"):
    """Load LoRA weights into a model."""
    lora_state = torch.load(path, map_location=device)
    model_state = model.state_dict()

    for name, param in lora_state.items():
        if name in model_state:
            model_state[name] = param

    model.load_state_dict(model_state)
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base model for faster inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    return model


def count_lora_params(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)

    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / total,
    }
