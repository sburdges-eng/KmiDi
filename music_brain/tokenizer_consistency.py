"""Tokenizer consistency validation for cross-process compatibility.

Closes the **Tokenizer consistency** spec item. Detects drift between
the tokenizer used at training time and the tokenizer loaded at
inference time, by comparing a fingerprint computed from the vocab +
merge rules + special-token map.

Two tokenizers can have the same vocab size but disagree on encoding —
e.g. a different BPE merge order, or a different special-token offset.
Either disagreement silently produces garbage outputs. This module
gives a single byte-level fingerprint to compare.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TokenizerFingerprint:
    """Hashable summary of a tokenizer's behavior contract.

    Fields:
        name: human-readable identifier (e.g. ``"bpe-32k"``).
        vocab_size: total number of token IDs.
        vocab_hash: short SHA-256 of the sorted ``(id, surface_form)``
            pairs. Catches swapped IDs.
        merges_hash: short SHA-256 of the BPE merge list in application
            order. None for non-BPE tokenizers (e.g. unigram, char).
        special_tokens: ordered mapping of role -> id (e.g.
            ``{"pad": 0, "bos": 1, "eos": 2}``).
        version: free-form version string. Bumping it forces a mismatch
            even when the other fields look identical, useful for
            "we patched something subtle, force a re-check" cases.
    """

    name: str
    vocab_size: int
    vocab_hash: str
    merges_hash: Optional[str]
    special_tokens: dict[str, int] = field(default_factory=dict)
    version: str = "1"

    def fingerprint(self) -> str:
        """Single short hash combining every consistency-relevant field."""
        payload = {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "vocab_hash": self.vocab_hash,
            "merges_hash": self.merges_hash,
            "special_tokens": dict(sorted(self.special_tokens.items())),
            "version": self.version,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, data: str) -> "TokenizerFingerprint":
        obj = json.loads(data)
        return cls(
            name=str(obj["name"]),
            vocab_size=int(obj["vocab_size"]),
            vocab_hash=str(obj["vocab_hash"]),
            merges_hash=obj.get("merges_hash"),
            special_tokens={k: int(v) for k, v in obj.get("special_tokens", {}).items()},
            version=str(obj.get("version", "1")),
        )

    @classmethod
    def load(cls, path: Path) -> "TokenizerFingerprint":
        return cls.from_json(Path(path).read_text())


def hash_vocab(vocab: dict[str, int]) -> str:
    """Hash the ``surface_form -> id`` mapping in sorted-id order."""
    pairs = sorted(vocab.items(), key=lambda kv: (kv[1], kv[0]))
    blob = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def hash_merges(merges: list[tuple[str, str]]) -> str:
    """Hash an ordered list of BPE merge pairs (order is significant)."""
    blob = json.dumps([list(m) for m in merges], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class ConsistencyReport:
    """Outcome of comparing two fingerprints."""

    consistent: bool
    mismatches: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.consistent


def compare(expected: TokenizerFingerprint, actual: TokenizerFingerprint) -> ConsistencyReport:
    """Diff two fingerprints field-by-field. Returns a ConsistencyReport."""
    diffs: list[str] = []
    if expected.name != actual.name:
        diffs.append(f"name: {expected.name!r} -> {actual.name!r}")
    if expected.vocab_size != actual.vocab_size:
        diffs.append(f"vocab_size: {expected.vocab_size} -> {actual.vocab_size}")
    if expected.vocab_hash != actual.vocab_hash:
        diffs.append(f"vocab_hash: {expected.vocab_hash} -> {actual.vocab_hash}")
    if expected.merges_hash != actual.merges_hash:
        diffs.append(f"merges_hash: {expected.merges_hash} -> {actual.merges_hash}")
    if expected.special_tokens != actual.special_tokens:
        diffs.append(f"special_tokens: {expected.special_tokens} -> {actual.special_tokens}")
    if expected.version != actual.version:
        diffs.append(f"version: {expected.version} -> {actual.version}")
    return ConsistencyReport(consistent=not diffs, mismatches=tuple(diffs))


__all__ = [
    "ConsistencyReport",
    "TokenizerFingerprint",
    "compare",
    "hash_merges",
    "hash_vocab",
]
