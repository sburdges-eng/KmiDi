"""
Latent canonicalization: orthogonal Procrustes + mean shift to align two embedding spaces.

Use when upgrading multimodal encoders (e.g. to a noise-robust Modest-Align architecture):
instead of re-embedding all vec:* sidecar databases, fit a map from ~500 anchor pairs
and apply it at query time so new queries match the old index (O(1) per query).

Reference: "Canonicalizing Multimodal Contrastive Representation Learning" — independently
trained contrastive models can be aligned by a shared orthogonal transform plus mean shift,
learnable from a small anchor set in one modality.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fit_orthogonal_map(
    anchor_old: np.ndarray,
    anchor_new: np.ndarray,
    center: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit orthogonal Procrustes (and optional mean shift) from new to old space.

    Solves for R (orthogonal) and optionally centers so that:
      anchor_new @ R + shift ≈ anchor_old
    (or equivalently: (anchor_new - center_new) @ R + center_old ≈ anchor_old).

    Args:
        anchor_old: (N, D) embeddings from the "old" (legacy) model.
        anchor_new: (N, D) embeddings from the "new" model for the same N items.
        center: If True, center both matrices before solving Procrustes and return centers.

    Returns:
        R: (D, D) orthogonal matrix (new -> old).
        center_old: (D,) or None; subtract from old-space queries when applying inverse.
        center_new: (D,) or None; subtract from new embeddings before applying R.
    """
    anchor_old = np.asarray(anchor_old, dtype=np.float64)
    anchor_new = np.asarray(anchor_new, dtype=np.float64)
    if anchor_old.shape[0] != anchor_new.shape[0]:
        raise ValueError("anchor_old and anchor_new must have the same number of rows")
    if anchor_old.shape[1] != anchor_new.shape[1]:
        raise ValueError("anchor_old and anchor_new must have the same dimension D")
    n, d = anchor_old.shape

    center_old = np.mean(anchor_old, axis=0) if center else None
    center_new = np.mean(anchor_new, axis=0) if center else None
    A = anchor_new - (center_new if center else 0.0)
    B = anchor_old - (center_old if center else 0.0)

    # Orthogonal Procrustes: min ||A @ R - B||_F s.t. R^T R = I  =>  A @ R ≈ B (new @ R ≈ old)
    try:
        from scipy.linalg import orthogonal_procrustes
        R, _ = orthogonal_procrustes(A, B)
        # So (anchor_new_centered) @ R ≈ (anchor_old_centered); R maps new -> old
    except ImportError:
        # Fallback: SVD of A^T B; R = U V^T so that A R best approximates B
        M = A.T @ B
        U, _, Vt = np.linalg.svd(M, full_matrices=True)
        R = U @ Vt

    return R, center_old, center_new


def apply_map(
    embedding_new: np.ndarray,
    R: np.ndarray,
    center_old: Optional[np.ndarray],
    center_new: Optional[np.ndarray],
    normalize: bool = False,
) -> np.ndarray:
    """
    Map embeddings from new space to old space for backward-compatible retrieval.

    embedding_old ≈ (embedding_new - center_new) @ R + center_old

    If centers were not used when fitting, pass None for both.

    When normalize is True, the result is L2-normalized per row so that
    dot-product / cosine similarity retrieval matches the legacy index (unit vectors).
    """
    x = np.asarray(embedding_new, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if center_new is not None:
        x = x - center_new
    out = x @ R
    if center_old is not None:
        out = out + center_old
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(norms > 0, out / norms, out)
    return out.squeeze() if embedding_new.ndim == 1 else out


def save_map(path: str, R: np.ndarray, center_old: Optional[np.ndarray], center_new: Optional[np.ndarray]) -> None:
    """Save R and centers to .npz for later use."""
    np.savez(
        path,
        R=R,
        center_old=center_old if center_old is not None else np.array([]),
        center_new=center_new if center_new is not None else np.array([]),
    )


def load_map(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load R and centers from .npz."""
    data = np.load(path, allow_pickle=False)
    R = data["R"]
    co = data["center_old"]
    cn = data["center_new"]
    center_old = co if co.size > 0 else None
    center_new = cn if cn.size > 0 else None
    return R, center_old, center_new
