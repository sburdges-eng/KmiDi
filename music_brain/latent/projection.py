"""LinearProjection — small affine bridge between embedding spaces.

A weights+bias pair plus apply() / save / load. Used wherever the music
brain stack needs to glue two embedding spaces of different dimensions —
e.g. conditioning a 256-d intent representation into a 384-d audio
encoder's input space, or attaching a low-rank adapter on top of a
pre-trained backbone.

This is *just* the inference-side primitive. Training a projection
requires gradient infra not in scope here — load pre-trained weights via
:meth:`from_json` / :meth:`load`, or hand-set ``weight``/``bias`` after
construction to plug in trained values.

Glorot uniform initialisation by default with a caller-supplied seed,
so the same seed produces identical untrained weights — useful for
deterministic-baseline comparisons and CI sanity checks.

Stdlib + numpy.

Goal-list ties: Conditioning bridge projection layer (direct),
Soft prompt or adapter injection (small linear adapters are this shape).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np


class LinearProjection:
    """``y = x @ W + b`` for embedding-space bridging.

    Parameters
    ----------
    in_dim, out_dim:
        Input and output dimensions; both must be > 0.
    seed:
        Optional integer seed for the Glorot-uniform initialiser. Same
        seed → same starting weights.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: Optional[int] = None) -> None:
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must both be > 0")
        self._in_dim = int(in_dim)
        self._out_dim = int(out_dim)
        rng = np.random.default_rng(seed)
        # Glorot uniform: U[-limit, +limit] with limit = sqrt(6 / (in + out)).
        limit = float(np.sqrt(6.0 / (in_dim + out_dim)))
        self.weight: np.ndarray = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(
            np.float32
        )
        self.bias: np.ndarray = np.zeros(out_dim, dtype=np.float32)

    # ----------------------------------------------------------------- meta
    @property
    def in_dim(self) -> int:
        return self._in_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    # --------------------------------------------------------------- apply
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Project ``x`` through the affine transform.

        Accepts a single 1-D vector of length ``in_dim`` or a 2-D batch of
        shape ``(N, in_dim)``. Returns matching rank: 1-D → 1-D, 2-D → 2-D.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            if x.shape[0] != self._in_dim:
                raise ValueError(
                    f"input dim {x.shape[0]} != expected {self._in_dim}"
                )
            return (x @ self.weight + self.bias).astype(np.float32)
        if x.ndim == 2:
            if x.shape[1] != self._in_dim:
                raise ValueError(
                    f"input dim {x.shape[1]} != expected {self._in_dim}"
                )
            return (x @ self.weight + self.bias).astype(np.float32)
        raise ValueError(f"expected 1-D or 2-D input; got shape {x.shape}")

    # -------------------------------------------------- serialisation (JSON)
    def to_json(self) -> str:
        return json.dumps(
            {
                "in_dim": self._in_dim,
                "out_dim": self._out_dim,
                "weight": self.weight.tolist(),
                "bias": self.bias.tolist(),
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "LinearProjection":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e
        try:
            obj = cls(in_dim=int(data["in_dim"]), out_dim=int(data["out_dim"]), seed=0)
            obj.weight = np.asarray(data["weight"], dtype=np.float32)
            obj.bias = np.asarray(data["bias"], dtype=np.float32)
        except (KeyError, TypeError) as e:
            raise ValueError(f"missing or malformed field: {e}") from e
        return obj

    def save(self, path: Union[Path, str]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> "LinearProjection":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
