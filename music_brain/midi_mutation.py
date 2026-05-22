"""Transactional MIDI mutation engine with all-or-nothing semantics.

Closes the **Real-time MIDI mutation transactional engine** spec item
documented as a gap in ``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``.
Reinforces ``docs/ARCHITECTURAL_CONTRACTS.md`` §6 (Clear rollback
semantics).

This module sits on the *message thread* — not the audio thread. It
governs edit operations that change the canonical MIDI buffer before
the next atomic shadow-swap (see ``src/plugin/PluginProcessor.cpp``
:1011-1020). The audio thread never sees a partially-applied edit:
either every mutation in a transaction lands, or none of them do.

Design:
- Stdlib only (dataclasses + copy).
- The engine holds a list of ``MidiNote`` records identified by stable
  ``id``. Mutations are typed (AddNote / RemoveNote / SetPitch /
  SetVelocity / MoveNote) so a transaction script is auditable and
  serialisable.
- Transactions snapshot the buffer at begin(), apply mutations
  in-order, and either commit() (release snapshot) or rollback()
  (restore snapshot).
- Mutations validate against the current state at apply time. A
  validation failure inside a transaction does NOT auto-rollback —
  it raises ``MutationError`` and leaves the partial mutations in the
  pending buffer. The caller decides whether to rollback() (the usual
  path) or continue. This makes "best-effort with skip-on-error"
  scripts possible.
- ``with engine.transaction() as txn`` is the typical entry point:
  it auto-commits on clean exit and auto-rollbacks on exception.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------
# Domain model
# ----------------------------------------------------------------------

@dataclass
class MidiNote:
    """A single note in the canonical buffer."""
    id: str
    pitch: int          # MIDI note number [0..127]
    velocity: int       # [0..127]
    start_tick: int     # >= 0
    duration_ticks: int  # > 0
    channel: int = 0    # [0..15]


class MutationError(Exception):
    """Raised when a mutation cannot be applied to the current state."""


# ----------------------------------------------------------------------
# Mutations (typed records — serialisable in future PRs)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AddNote:
    note: MidiNote


@dataclass(frozen=True)
class RemoveNote:
    note_id: str


@dataclass(frozen=True)
class SetPitch:
    note_id: str
    new_pitch: int


@dataclass(frozen=True)
class SetVelocity:
    note_id: str
    new_velocity: int


@dataclass(frozen=True)
class MoveNote:
    note_id: str
    new_start_tick: int


Mutation = AddNote | RemoveNote | SetPitch | SetVelocity | MoveNote


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class MidiMutationEngine:
    """Holds a canonical list of notes and applies transactional edits."""

    def __init__(self) -> None:
        self._notes: dict[str, MidiNote] = {}
        self._txn_snapshot: Optional[dict[str, MidiNote]] = None
        self._txn_log: list[Mutation] = []

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def notes(self) -> list[MidiNote]:
        """Snapshot of current notes, sorted by (start_tick, id)."""
        return sorted(self._notes.values(), key=lambda n: (n.start_tick, n.id))

    def get(self, note_id: str) -> Optional[MidiNote]:
        n = self._notes.get(note_id)
        return copy.copy(n) if n else None

    def __len__(self) -> int:
        return len(self._notes)

    def __contains__(self, note_id: object) -> bool:
        return isinstance(note_id, str) and note_id in self._notes

    # ------------------------------------------------------------------
    # Transaction lifecycle
    # ------------------------------------------------------------------

    def begin(self) -> None:
        if self._txn_snapshot is not None:
            raise MutationError("transaction already in progress")
        self._txn_snapshot = copy.deepcopy(self._notes)
        self._txn_log = []

    def commit(self) -> list[Mutation]:
        if self._txn_snapshot is None:
            raise MutationError("no transaction in progress")
        log = list(self._txn_log)
        self._txn_snapshot = None
        self._txn_log = []
        return log

    def rollback(self) -> None:
        if self._txn_snapshot is None:
            raise MutationError("no transaction in progress")
        self._notes = self._txn_snapshot
        self._txn_snapshot = None
        self._txn_log = []

    def in_transaction(self) -> bool:
        return self._txn_snapshot is not None

    def transaction(self) -> "_TransactionGuard":
        return _TransactionGuard(self)

    # ------------------------------------------------------------------
    # Mutation application
    # ------------------------------------------------------------------

    def apply(self, mutation: Mutation) -> None:
        """Apply a single mutation against current state.

        Raises ``MutationError`` if the mutation is invalid (e.g.
        target note doesn't exist, pitch out of range). Validation
        failures inside a transaction do NOT auto-rollback; the caller
        decides whether to continue or rollback.
        """
        if isinstance(mutation, AddNote):
            self._apply_add(mutation)
        elif isinstance(mutation, RemoveNote):
            self._apply_remove(mutation)
        elif isinstance(mutation, SetPitch):
            self._apply_set_pitch(mutation)
        elif isinstance(mutation, SetVelocity):
            self._apply_set_velocity(mutation)
        elif isinstance(mutation, MoveNote):
            self._apply_move(mutation)
        else:  # pragma: no cover — exhaustive over the Mutation union
            raise MutationError(f"unknown mutation type: {type(mutation).__name__}")
        if self._txn_snapshot is not None:
            self._txn_log.append(mutation)

    def apply_many(self, mutations: list[Mutation]) -> None:
        """Apply mutations in order. Failure leaves partial state in the
        transaction (caller must rollback explicitly)."""
        for m in mutations:
            self.apply(m)

    # ------------------------------------------------------------------
    # Internal: per-mutation handlers
    # ------------------------------------------------------------------

    def _apply_add(self, m: AddNote) -> None:
        n = m.note
        _validate_pitch(n.pitch)
        _validate_velocity(n.velocity)
        if n.duration_ticks <= 0:
            raise MutationError(f"duration_ticks must be > 0, got {n.duration_ticks}")
        if n.start_tick < 0:
            raise MutationError(f"start_tick must be >= 0, got {n.start_tick}")
        if n.channel < 0 or n.channel > 15:
            raise MutationError(f"channel out of range: {n.channel}")
        if n.id in self._notes:
            raise MutationError(f"note id already exists: {n.id!r}")
        self._notes[n.id] = copy.copy(n)

    def _apply_remove(self, m: RemoveNote) -> None:
        if m.note_id not in self._notes:
            raise MutationError(f"unknown note id: {m.note_id!r}")
        del self._notes[m.note_id]

    def _apply_set_pitch(self, m: SetPitch) -> None:
        n = self._require(m.note_id)
        _validate_pitch(m.new_pitch)
        n.pitch = m.new_pitch

    def _apply_set_velocity(self, m: SetVelocity) -> None:
        n = self._require(m.note_id)
        _validate_velocity(m.new_velocity)
        n.velocity = m.new_velocity

    def _apply_move(self, m: MoveNote) -> None:
        n = self._require(m.note_id)
        if m.new_start_tick < 0:
            raise MutationError(f"new_start_tick must be >= 0, got {m.new_start_tick}")
        n.start_tick = m.new_start_tick

    def _require(self, note_id: str) -> MidiNote:
        n = self._notes.get(note_id)
        if n is None:
            raise MutationError(f"unknown note id: {note_id!r}")
        return n


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_pitch(pitch: int) -> None:
    if not (0 <= pitch <= 127):
        raise MutationError(f"pitch out of range [0..127]: {pitch}")


def _validate_velocity(velocity: int) -> None:
    if not (0 <= velocity <= 127):
        raise MutationError(f"velocity out of range [0..127]: {velocity}")


# ----------------------------------------------------------------------
# Context manager
# ----------------------------------------------------------------------

@dataclass
class _TransactionGuard:
    engine: MidiMutationEngine
    _committed: bool = field(default=False, init=False)

    def __enter__(self) -> "_TransactionGuard":
        self.engine.begin()
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:
        if exc_type is not None:
            # Propagate the exception, but rollback first.
            self.engine.rollback()
            return
        if not self._committed:
            self.engine.commit()

    def apply(self, mutation: Mutation) -> None:
        self.engine.apply(mutation)

    def apply_many(self, mutations: list[Mutation]) -> None:
        self.engine.apply_many(mutations)

    def commit(self) -> list[Mutation]:
        log = self.engine.commit()
        self._committed = True
        return log

    def rollback(self) -> None:
        self.engine.rollback()
        self._committed = True  # prevent double-commit on __exit__


__all__ = [
    "AddNote",
    "MidiMutationEngine",
    "MidiNote",
    "MoveNote",
    "Mutation",
    "MutationError",
    "RemoveNote",
    "SetPitch",
    "SetVelocity",
]
