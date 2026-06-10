"""Bar-range generation scope isolation.

Closes **Generation scope isolation** and **Adaptive state-aware
delivery**. The scope is a [start_bar, end_bar) half-open range plus
an optional source ``intent_id``; downstream emitters check that any
write they make falls inside the scope or they raise
``ScopeViolation``. The orchestrator hands one scope per pending
intent into each generator agent so concurrent edits to disjoint bar
ranges cannot collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")


class ScopeViolation(Exception):
    """Raised when a generator attempts to write outside its scope."""

    def __init__(self, scope: "GenerationScope", attempted_bar: int) -> None:
        super().__init__(
            f"attempted to write bar {attempted_bar} "
            f"outside scope [{scope.start_bar}, {scope.end_bar})"
        )
        self.scope = scope
        self.attempted_bar = attempted_bar
        self.intent_id = scope.intent_id


@dataclass(frozen=True)
class GenerationScope:
    """Half-open bar range with optional intent_id provenance."""

    start_bar: int
    end_bar: int
    intent_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.start_bar < 0:
            raise ValueError(f"start_bar must be >= 0, got {self.start_bar}")
        if self.end_bar <= self.start_bar:
            raise ValueError(f"end_bar ({self.end_bar}) must be > start_bar ({self.start_bar})")

    @property
    def length_bars(self) -> int:
        return self.end_bar - self.start_bar

    def contains_bar(self, bar: int) -> bool:
        return self.start_bar <= bar < self.end_bar

    def enforce_bar(self, bar: int) -> None:
        if not self.contains_bar(bar):
            raise ScopeViolation(self, bar)

    def collect(self, items: Iterable[T], *, bar_of: Callable[[T], int]) -> List[T]:
        """Return only the items whose bar falls inside the scope."""
        return [x for x in items if self.contains_bar(bar_of(x))]

    def overlap(self, other: "GenerationScope") -> Optional["GenerationScope"]:
        lo = max(self.start_bar, other.start_bar)
        hi = min(self.end_bar, other.end_bar)
        if hi <= lo:
            return None
        return GenerationScope(start_bar=lo, end_bar=hi, intent_id=self.intent_id)


__all__ = ["GenerationScope", "ScopeViolation"]
