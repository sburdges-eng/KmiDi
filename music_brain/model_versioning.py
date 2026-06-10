"""Model versioning policy + manifest for KMiDi checkpoints.

Closes the **Model versioning policy** spec item documented as a gap in
``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``. Also reinforces
**Checkpoint rollback** by giving the rollback logic a typed compatibility
check rather than ad-hoc filename parsing.

The policy treats checkpoint compatibility as semantic versioning over
two axes:

  - **schema_version**: major bumps when the model architecture changes
    in a way that breaks loading older weights (new tensors, removed
    layers, changed shapes). Minor bumps for additive compatible
    extensions.
  - **arch_hash**: short stable hash over the architecture definition
    (layer config) — catches "same schema_version but the wrong code
    checked it in". Mismatch is always blocking, even at the same
    schema_version.

The training data is tracked via ``data_hash`` for reproducibility but
does NOT participate in compatibility — two checkpoints can be
compatible-for-inference even if trained on different data.

Manifests serialise as small JSON files alongside the checkpoint, so
roll-forward and roll-back can be decided by inspecting the manifest
without loading the (potentially large) weights.

Scope:
- In-process, stdlib only (dataclasses + json + hashlib).
- Loader/consumer policy is opt-in: callers ask
  ``is_compatible(consumer, candidate)``; nothing is enforced
  globally.
- Hash format is the first 12 hex chars of a SHA-256 — short enough to
  be human-checkable, wide enough to make collisions accidental rather
  than adversarial.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def short_hash(payload: str | bytes) -> str:
    """First 12 hex chars of SHA-256 over ``payload``."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class ModelManifest:
    """Versioning metadata for a single checkpoint.

    Fields:
        name: human-readable model name (e.g. "AudioJEPA-tiny").
        schema_version: (major, minor). Major bumps break weight loading.
        arch_hash: short_hash() over the architecture definition. Loaders
            with a different arch_hash MUST refuse this checkpoint even
            at the same schema_version.
        params_hash: short_hash() over the checkpoint file contents.
            Used for cache keying and download-integrity checks. Not part
            of compatibility — two compatible checkpoints can have
            different params_hash.
        data_hash: short_hash() of the training-data manifest. Tracked
            for reproducibility; never participates in compatibility.
        created_at: ISO-8601 timestamp string (caller-supplied).
        notes: free-form text. Useful for "do not roll back past this
            point" markers.
    """

    name: str
    schema_version: tuple[int, int]
    arch_hash: str
    params_hash: str
    data_hash: str
    created_at: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, data: str) -> "ModelManifest":
        obj = json.loads(data)
        sv = obj.get("schema_version", [0, 0])
        if isinstance(sv, list):
            sv = tuple(sv)
        if not (isinstance(sv, tuple) and len(sv) == 2 and all(isinstance(x, int) for x in sv)):
            raise ValueError(f"invalid schema_version: {obj.get('schema_version')}")
        return cls(
            name=str(obj["name"]),
            schema_version=sv,
            arch_hash=str(obj["arch_hash"]),
            params_hash=str(obj["params_hash"]),
            data_hash=str(obj["data_hash"]),
            created_at=str(obj["created_at"]),
            notes=str(obj.get("notes", "")),
            extra=dict(obj.get("extra", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        return cls.from_json(Path(path).read_text())


# ----------------------------------------------------------------------
# Compatibility policy
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CompatibilityVerdict:
    compatible: bool
    reason: str

    def __bool__(self) -> bool:
        return self.compatible


def is_compatible(consumer: ModelManifest, candidate: ModelManifest) -> CompatibilityVerdict:
    """Decide whether ``candidate`` is loadable by code expecting ``consumer``.

    Rules (in order):
      1. Name must match.
      2. arch_hash must match exactly (different architecture → block).
      3. Major schema_version must match.
      4. Candidate's minor schema_version must be >= consumer's
         (a newer minor schema is loadable as long as it is additive;
         an older minor is not, because the consumer may reference a
         field the older minor didn't have).
    """
    if consumer.name != candidate.name:
        return CompatibilityVerdict(
            False, f"name mismatch: {consumer.name!r} vs {candidate.name!r}"
        )
    if consumer.arch_hash != candidate.arch_hash:
        return CompatibilityVerdict(
            False, f"arch_hash mismatch: {consumer.arch_hash} vs {candidate.arch_hash}"
        )
    c_major, c_minor = consumer.schema_version
    k_major, k_minor = candidate.schema_version
    if c_major != k_major:
        return CompatibilityVerdict(False, f"schema_version major mismatch: {c_major} vs {k_major}")
    if k_minor < c_minor:
        return CompatibilityVerdict(
            False,
            f"candidate minor {k_minor} < consumer minor {c_minor} "
            "(would miss fields the consumer expects)",
        )
    return CompatibilityVerdict(True, "compatible")


def pick_rollback_target(
    consumer: ModelManifest, candidates: list[ModelManifest]
) -> Optional[ModelManifest]:
    """Pick the most-recent ``candidate`` compatible with ``consumer``.

    "Most recent" is determined by ``created_at`` string order, which is
    valid for ISO-8601 timestamps. Returns None if no candidate is
    compatible.
    """
    compatible = [c for c in candidates if is_compatible(consumer, c)]
    if not compatible:
        return None
    return max(compatible, key=lambda c: c.created_at)


__all__ = [
    "CompatibilityVerdict",
    "ModelManifest",
    "is_compatible",
    "pick_rollback_target",
    "short_hash",
]
