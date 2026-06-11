"""GenerationTransaction — sandboxed mutation with explicit commit / rollback.

Wraps a mutable state dict in a ``with``-block. Inside the block, every
mutation lives on a deep-copied snapshot; the parent dict is only touched
when the caller explicitly calls :meth:`GenerationTransaction.commit`. Any
exception inside the block rolls back even if :meth:`commit` was called —
the rollback semantics intentionally win over commit on the error path,
so "audition this regeneration" code can't accidentally leak a partial
mutation when it crashes mid-way.

Goal-list ties:
  - Generation scope isolation (direct).
  - Realtime audition-safe regeneration (the primitive enabling it).
  - Realtime rollback system (rollback semantics).

Stdlib only.
"""

from __future__ import annotations

import copy
from typing import Any, Dict


class RollbackError(Exception):
    """Misuse: committing after rollback / committing twice / committing
    after the scope has closed."""


class GenerationTransaction:
    """Sandbox a mutable state dict for trial-and-error generation.

    Usage::

        with GenerationTransaction(state) as scope:
            scope.state["chord"] = "F"
            if happy_with_it:
                scope.commit()
            # otherwise: implicit rollback on exit

    Default behaviour is *rollback*; the caller must explicitly call
    :meth:`commit` to keep the changes. If the body raises, rollback wins
    even when ``commit()`` was already invoked.
    """

    def __init__(self, parent: Dict[str, Any]) -> None:
        self._parent = parent
        self._snapshot: Dict[str, Any] | None = None
        # Captured at commit() time so post-commit in-scope mutations don't
        # leak into the parent on __exit__.
        self._commit_state: Dict[str, Any] | None = None
        self._committed = False
        self._closed = False
        self._active = False
        self._rolled_back = False

    @property
    def is_active(self) -> bool:
        """True while the ``with`` block is running."""
        return self._active

    @property
    def has_committed(self) -> bool:
        """True after :meth:`commit` has been called (still True post-exit)."""
        return self._committed

    @property
    def state(self) -> Dict[str, Any]:
        """The mutable in-scope snapshot. Inside the with-block this is the
        deep-copied working state; outside the block it's a no-op reference."""
        return self._snapshot if self._snapshot is not None else self._parent

    # ------------------------------------------------------------------ ctx
    def __enter__(self) -> "GenerationTransaction":
        self._snapshot = copy.deepcopy(self._parent)
        self._committed = False
        self._active = True
        self._closed = False
        self._rolled_back = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is not None:
                # Exception: rollback wins over any prior commit().
                self._committed = False
                return False  # re-raise
            if self._committed and self._commit_state is not None:
                # Propagate the at-commit-time state to parent. Mutations
                # after commit() are intentionally invisible.
                self._parent.clear()
                self._parent.update(self._commit_state)
            return False
        finally:
            self._active = False
            self._closed = True

    # ------------------------------------------------------------------ api
    def commit(self) -> None:
        """Mark this scope as committing on exit. Idempotent within a single
        with-block is *not* allowed — call once. Calling after the scope has
        closed raises :class:`RollbackError`."""
        if self._closed:
            raise RollbackError("scope is closed; cannot commit after exit")
        if self._rolled_back:
            raise RollbackError("scope has been rolled back; cannot commit")
        if self._committed:
            raise RollbackError("scope has already been committed")
        self._committed = True
        # Snapshot the at-commit-time state so mutations after commit()
        # don't leak into the parent on __exit__.
        self._commit_state = copy.deepcopy(self._snapshot)

    def rollback(self) -> None:
        """Explicit rollback. After this, even a later ``commit()`` won't
        propagate changes (until the with-block exits and a new scope is
        entered)."""
        if self._closed:
            raise RollbackError("scope is closed; cannot rollback after exit")
        self._committed = False
        self._rolled_back = True
        # Re-snapshot from parent so the in-scope state matches the parent
        # again (the caller may want to keep experimenting).
        self._snapshot = copy.deepcopy(self._parent)
