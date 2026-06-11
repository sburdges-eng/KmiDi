"""ContextWindow — sliding token-budgeted context with overflow summarisation.

Wraps a token budget around a list of conversational entries (role + text +
token count). When an append would exceed the budget, the oldest entries
are popped off and — if a ``summarizer`` callback is configured — passed to
that callback so their content can be folded into a persistent ``summary``
prefix instead of being lost outright.

This is the host-side primitive behind "Context persistence beyond token
window": each step the model gets a ``summary || recent_entries`` view,
and the summary keeps growing as older context falls off. The summarizer
itself is the caller's choice — extractive, abstractive, embedding-based,
or just identity for "remember verbatim".

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

# A summarizer takes (dropped_entries, current_summary) and returns the new summary.
SummarizerFn = Callable[[Sequence["ContextEntry"], str], str]


@dataclass
class ContextEntry:
    """One turn in the conversation."""

    role: str
    text: str
    tokens: int

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens must be >= 0")


class ContextWindow:
    """Sliding-window context with optional overflow summarisation.

    Parameters
    ----------
    max_tokens:
        Soft upper bound on the in-window entries' total token count.
        Oversized single entries are stored anyway (the alternative is
        silently dropping caller data); use :meth:`would_fit` to gate
        appends if that's not acceptable.
    summarizer:
        Optional callback invoked with the entries that just fell out of
        the window plus the current summary; returns the new summary.
        ``None`` → dropped entries are discarded.
    """

    def __init__(
        self,
        max_tokens: int,
        summarizer: Optional[SummarizerFn] = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self._max_tokens = int(max_tokens)
        self._summarizer = summarizer
        self._entries: List[ContextEntry] = []
        self._summary = ""

    # -------------------------------------------------------------- inspect
    @property
    def entries(self) -> List[ContextEntry]:
        return list(self._entries)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens for e in self._entries)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def would_fit(self, tokens: int) -> bool:
        return self.total_tokens + tokens <= self._max_tokens

    # -------------------------------------------------------------- mutate
    def append(self, entry: ContextEntry) -> None:
        if entry.role not in _VALID_ROLES:
            raise ValueError(
                f"role {entry.role!r} not in {sorted(_VALID_ROLES)}"
            )
        self._entries.append(entry)
        self._evict_until_fits()

    def _evict_until_fits(self) -> None:
        """Pop oldest entries until the window fits ``max_tokens``.

        A single oversized entry is kept (no entries → would re-evict to
        empty; we don't drop the just-appended data either).
        """
        dropped: List[ContextEntry] = []
        while (
            len(self._entries) > 1
            and sum(e.tokens for e in self._entries) > self._max_tokens
        ):
            dropped.append(self._entries.pop(0))
        if dropped and self._summarizer is not None:
            self._summary = self._summarizer(dropped, self._summary)

    # -------------------------------------------------------------- render
    def render(self) -> str:
        """Render the window as ``summary || recent_entries`` text.

        Format is intentionally minimal — callers will usually replace this
        with their model-specific chat formatter. Suitable as a smoke check.
        """
        lines: List[str] = []
        if self._summary:
            lines.append(f"[summary] {self._summary}")
        for entry in self._entries:
            lines.append(f"{entry.role}: {entry.text}")
        return "\n".join(lines)
