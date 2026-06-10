"""Deterministic record/replay harness for seeded generation runs.

Closes the **Deterministic record/replay** spec item documented as a gap
in ``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``, and reinforces
``docs/ARCHITECTURAL_CONTRACTS.md`` §12 (Clear export determinism) and
§17 (deterministic vs probabilistic boundary).

Two complementary tools:

  1. ``DeterministicContext`` — seeds Python's stdlib RNG, NumPy's RNG,
     and (when present) PyTorch's RNG for CPU + CUDA. Restores prior
     state on exit. Use this as a ``with``-block around any probabilistic
     subsystem whose output must be byte-identical for a fixed
     ``(inputs, seed, model_hash)`` tuple.

  2. ``ReplayLog`` — appends a JSON record per generation with the
     ``run_id``, ``seed``, input hash, model manifest hash, output hash,
     and timestamp. Two runs with the same input/seed/model triplet that
     produce different output hashes is a determinism regression — the
     log lets CI compare across versions.

Scope:
- Stdlib only by default; numpy / torch wiring is best-effort and
  guarded by import-availability.
- Single-process. Cross-process / cross-host reproducibility additionally
  requires consistent toolchain pinning (already covered by
  ``Cargo.lock`` / ``uv.lock`` / ``package-lock.json``).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


def _hash_payload(payload: bytes | str | dict | list) -> str:
    """Stable short hex hash of an input payload."""
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class DeterministicContext:
    """Context manager that seeds available RNGs and restores prior state.

    Args:
        seed: 64-bit integer seed. The same seed is passed to every
            available backend (random, numpy, torch CPU + CUDA).

    On exit, every backend's RNG is restored to the snapshot captured at
    entry. This makes the harness safe to use around probabilistic code
    without contaminating the surrounding test or process state.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._py_state: Optional[Any] = None
        self._np_state: Optional[Any] = None
        self._torch_cpu_state: Optional[Any] = None
        self._torch_cuda_state: Optional[Any] = None

    # ------------------------------------------------------------------
    # Backend probing
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import(name: str) -> Optional[Any]:
        try:
            return __import__(name)
        except Exception:  # noqa: BLE001 — optional dep, missing is fine
            return None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "DeterministicContext":
        # Python stdlib.
        self._py_state = random.getstate()
        random.seed(self.seed)

        # NumPy (legacy global RNG path, kept for parity with model code).
        np = self._try_import("numpy")
        if np is not None:
            self._np_state = np.random.get_state()
            np.random.seed(self.seed)

        # PyTorch CPU + CUDA, best-effort.
        torch = self._try_import("torch")
        if torch is not None:
            try:
                self._torch_cpu_state = torch.random.get_rng_state()
                torch.manual_seed(self.seed)
            except Exception:  # noqa: BLE001
                self._torch_cpu_state = None
            cuda = getattr(torch, "cuda", None)
            if cuda is not None:
                try:
                    if cuda.is_available():
                        self._torch_cuda_state = cuda.get_rng_state_all()
                        cuda.manual_seed_all(self.seed)
                except Exception:  # noqa: BLE001
                    self._torch_cuda_state = None
        return self

    def __exit__(self, *_exc) -> None:
        if self._py_state is not None:
            random.setstate(self._py_state)
        np = self._try_import("numpy")
        if np is not None and self._np_state is not None:
            try:
                np.random.set_state(self._np_state)
            except Exception:  # noqa: BLE001
                pass
        torch = self._try_import("torch")
        if torch is not None:
            try:
                if self._torch_cpu_state is not None:
                    torch.random.set_rng_state(self._torch_cpu_state)
            except Exception:  # noqa: BLE001
                pass
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and self._torch_cuda_state is not None:
                try:
                    cuda.set_rng_state_all(self._torch_cuda_state)
                except Exception:  # noqa: BLE001
                    pass


@dataclass(frozen=True)
class ReplayRecord:
    """A single deterministic-replay log entry."""

    run_id: str
    seed: int
    input_hash: str
    model_hash: str
    output_hash: str
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "seed": self.seed,
                "input_hash": self.input_hash,
                "model_hash": self.model_hash,
                "output_hash": self.output_hash,
                "timestamp": self.timestamp,
                "extra": self.extra,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> "ReplayRecord":
        obj = json.loads(line)
        return cls(
            run_id=str(obj["run_id"]),
            seed=int(obj["seed"]),
            input_hash=str(obj["input_hash"]),
            model_hash=str(obj["model_hash"]),
            output_hash=str(obj["output_hash"]),
            timestamp=str(obj["timestamp"]),
            extra=dict(obj.get("extra", {})),
        )


class ReplayLog:
    """Append-only JSONL log of (input, seed, model) -> output_hash records.

    Used by CI to detect determinism regressions: if a record exists in
    the historical log with the same (input_hash, seed, model_hash) but
    a different output_hash, deterministic-export has regressed.

    Format: one JSON object per line (JSONL). Stable across encoders;
    no leading whitespace.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: ReplayRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def __iter__(self) -> Iterable[ReplayRecord]:
        if not self._path.exists():
            return iter([])
        return self._iter_records()

    def _iter_records(self) -> Iterable[ReplayRecord]:
        with self._path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                yield ReplayRecord.from_json(raw)

    def find_conflicts(self, record: ReplayRecord) -> list[ReplayRecord]:
        """Return prior records with same (input, seed, model) but different output."""
        out: list[ReplayRecord] = []
        for prior in self:
            if (
                prior.input_hash == record.input_hash
                and prior.seed == record.seed
                and prior.model_hash == record.model_hash
                and prior.output_hash != record.output_hash
            ):
                out.append(prior)
        return out


__all__ = [
    "DeterministicContext",
    "ReplayLog",
    "ReplayRecord",
    "_hash_payload",
]
