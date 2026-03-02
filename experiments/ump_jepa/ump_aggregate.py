"""Windowed UMP control aggregation: perceptual quantization + mean/std/slope per CC."""

from __future__ import annotations

import math
from typing import Any

MAX32 = 0xFFFFFFFF
BINS = 16384  # 14-bit effective


def perceptual_quantize(v: int) -> int:
    """Map 32-bit CC value to perceptually uniform bin (sqrt scaling)."""
    if v <= 0:
        return 0
    if v >= MAX32:
        return BINS - 1
    x = v / MAX32
    x = math.sqrt(x)
    return int(x * (BINS - 1))


def aggregate_window(cc_events: list[dict[str, Any]]) -> dict[str, int] | None:
    """Compute mean, std, slope (perceptually quantized) for one CC over a window."""
    if not cc_events:
        return None
    try:
        values = [e["value32"] for e in cc_events]
    except KeyError:
        values = [e.get("value", e.get("value32", 0)) for e in cc_events]
    if not values:
        return None
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_val = math.sqrt(variance) if variance >= 0 else 0.0
    slope_val = values[-1] - values[0] if len(values) > 1 else 0
    return {
        "mean": perceptual_quantize(int(mean_val)),
        "std": perceptual_quantize(int(std_val)),
        "slope": perceptual_quantize(int(slope_val)),
    }
