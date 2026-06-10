"""
PID Flow — Partial Information Decomposition per layer for modality collapse detection.

Computes layer-wise how much of the representation is redundant, unique to each
modality (e.g. text vs audio/MIDI), or synergistic. Use as an offline eval hook:
if the report shows 0% audio-unique in late layers, the model has effectively
collapsed to text-only (deaf to the music).

Pipeline (simplified from "PID Flow" paper):
  1. Extract activations per layer for a batch of (modality_a, modality_b) pairs.
  2. Dimensionality reduction (PCA) to avoid undersampling.
  3. Rank-based Gaussianization so closed-form Gaussian PID applies.
  4. Closed-form PID: Redundant, Unique_A, Unique_B, Synergy (percentages).

Usage:
  from penta_core.ml.diagnostics.pid_flow import run_pid_flow_report
  report = run_pid_flow_report(model, dataloader, layer_names, device="mps")
  print(pid_report_to_string(report))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerPID:
    """PID breakdown for one layer (percentages)."""

    layer_name: str
    redundant_pct: float
    unique_a_pct: float  # e.g. text-unique
    unique_b_pct: float  # e.g. audio/MIDI-unique
    synergy_pct: float


def _gaussian_mi(cov_xy: np.ndarray, dim_x: int, dim_y: int) -> float:
    """Mutual information I(X;Y) for joint Gaussian (nats). cov_xy is (dim_x+dim_y, dim_x+dim_y)."""
    n = cov_xy.shape[0]
    if n == 0 or dim_x <= 0 or dim_y <= 0:
        return 0.0
    reg = 1e-5
    try:
        cov_reg = cov_xy + reg * np.eye(n)
        det_full = np.linalg.det(cov_reg)
        det_x = np.linalg.det(cov_reg[:dim_x, :dim_x])
        det_y = np.linalg.det(cov_reg[dim_x:, dim_x:])
        if det_full <= 0 or det_x <= 0 or det_y <= 0:
            return 0.0
        ratio = (det_x * det_y) / (det_full + 1e-15)
        if ratio <= 0:
            return 0.0
        return 0.5 * float(np.log(ratio))
    except Exception:
        return 0.0


def _rank_gaussianize(x: np.ndarray) -> np.ndarray:
    """Rank-based Gaussianization: replace values by normal quantiles of ranks."""
    out = np.zeros_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        col = x[:, j]
        order = np.argsort(col)
        ranks = np.empty_like(col, dtype=np.float64)
        ranks[order] = np.linspace(0.5, len(col) - 0.5, len(col))
        # Normal quantiles
        from scipy import stats

        out[:, j] = stats.norm.ppf(ranks / (len(col) + 1))
    return out


def compute_layer_pid(
    t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_components: int = 32,
    use_rank_gaussianize: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute Redundant, Unique_A, Unique_B, Synergy (as fractions 0..1) for one layer.

    t: (n_samples, dim_t) target (e.g. fused or decoder hidden state)
    a: (n_samples, dim_a) modality A (e.g. text branch)
    b: (n_samples, dim_b) modality B (e.g. audio branch)
    """
    n = t.shape[0]
    if n < 10:
        return 0.0, 0.0, 0.0, 0.0

    # Flatten and cap dimension for stability
    t_flat = t.reshape(n, -1).astype(np.float64)
    a_flat = a.reshape(n, -1).astype(np.float64)
    b_flat = b.reshape(n, -1).astype(np.float64)

    max_d = min(n_components, n // 2, t_flat.shape[1], a_flat.shape[1], b_flat.shape[1], 64)
    if max_d < 2:
        return 0.0, 0.0, 0.0, 0.0

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        # Fallback: no PCA, use first max_d dimensions
        t_flat = t_flat[:, :max_d]
        a_flat = a_flat[:, :max_d]
        b_flat = b_flat[:, :max_d]
    else:
        pca_t = PCA(n_components=max_d, random_state=42).fit(t_flat)
        t_flat = pca_t.transform(t_flat)
        a_flat = PCA(n_components=max_d, random_state=42).fit_transform(a_flat)
        b_flat = PCA(n_components=max_d, random_state=42).fit_transform(b_flat)

    if use_rank_gaussianize:
        t_flat = _rank_gaussianize(t_flat)
        a_flat = _rank_gaussianize(a_flat)
        b_flat = _rank_gaussianize(b_flat)

    dim_t, dim_a, dim_b = t_flat.shape[1], a_flat.shape[1], b_flat.shape[1]
    # Joint (T, A, B)
    joint = np.hstack([t_flat, a_flat, b_flat])
    cov = np.cov(joint, rowvar=False)
    if cov.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    i_ta = _gaussian_mi(np.cov(np.hstack([t_flat, a_flat]), rowvar=False), dim_t, dim_a)
    i_tb = _gaussian_mi(np.cov(np.hstack([t_flat, b_flat]), rowvar=False), dim_t, dim_b)
    i_tab = _gaussian_mi(cov, dim_t, dim_a + dim_b)

    # Practical PID approximation: Red = min(I(T;A), I(T;B)); then unique and synergy
    red = min(i_ta, i_tb)
    uniq_a = max(0.0, i_ta - red)
    uniq_b = max(0.0, i_tb - red)
    syn = max(0.0, i_tab - i_ta - i_tb + red)

    total = red + uniq_a + uniq_b + syn
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return red / total, uniq_a / total, uniq_b / total, syn / total


def run_pid_flow_report(
    model: Any,
    get_activations: Callable[[Any], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    layer_names: List[str],
    n_components: int = 32,
    use_rank_gaussianize: bool = True,
) -> List[LayerPID]:
    """
    Run PID for each layer and return a list of LayerPID.

    get_activations(model) must return a list of (t, a, b) arrays per layer,
    where t = target (e.g. fused or decoder hidden), a = modality A, b = modality B.
    layer_names[i] is the name for the i-th layer.
    """
    activations = get_activations(model)
    report: List[LayerPID] = []
    for i, name in enumerate(layer_names):
        if i >= len(activations):
            break
        t, a, b = activations[i]
        red, ua, ub, syn = compute_layer_pid(
            t,
            a,
            b,
            n_components=n_components,
            use_rank_gaussianize=use_rank_gaussianize,
        )
        report.append(
            LayerPID(
                layer_name=name,
                redundant_pct=red * 100.0,
                unique_a_pct=ua * 100.0,
                unique_b_pct=ub * 100.0,
                synergy_pct=syn * 100.0,
            )
        )
    return report


def pid_report_to_string(
    report: List[LayerPID],
    name_a: str = "Text",
    name_b: str = "Audio/MIDI",
) -> str:
    """Format PID report for terminal
    (e.g. 'Layer 12: 95% Text Unique, 2% MIDI Unique, 3% Synergy')."""
    lines = []
    for r in report:
        lines.append(
            f"{r.layer_name}: Redundant {r.redundant_pct:.1f}% | "
            f"{name_a} Unique {r.unique_a_pct:.1f}% | "
            f"{name_b} Unique {r.unique_b_pct:.1f}% | "
            f"Synergy {r.synergy_pct:.1f}%"
        )
    return "\n".join(lines)


def pid_report_to_dict(report: List[LayerPID]) -> List[Dict[str, Any]]:
    """Serialize report to JSON-suitable list of dicts."""
    return [
        {
            "layer_name": r.layer_name,
            "redundant_pct": r.redundant_pct,
            "unique_a_pct": r.unique_a_pct,
            "unique_b_pct": r.unique_b_pct,
            "synergy_pct": r.synergy_pct,
        }
        for r in report
    ]


def check_modality_collapse(
    report: List[LayerPID],
    audio_unique_threshold: float = 0.05,
    text_unique_threshold: float = 0.80,
    name_a: str = "Text",
    name_b: str = "Audio/MIDI",
) -> List[Tuple[str, str]]:
    """
    Check for modality collapse: layers where audio-unique is very low and text-unique is very high.

    Returns a list of (layer_name, message) warnings. Use as a "Check Engine Light" for training:
    if any layer triggers, consider halting or adjusting cross-attention / loss weights.
    """
    warnings: List[Tuple[str, str]] = []
    for r in report:
        ua_frac = r.unique_a_pct / 100.0
        ub_frac = r.unique_b_pct / 100.0
        if ub_frac < audio_unique_threshold and ua_frac > text_unique_threshold:
            msg = (
                f"CRITICAL WARNING - Modality collapse suspected. "
                f"{name_a} Unique {r.unique_a_pct:.0f}%, "
                f"{name_b} Unique {r.unique_b_pct:.0f}%, Synergy {r.synergy_pct:.0f}%."
            )
            warnings.append((r.layer_name, msg))
    return warnings
