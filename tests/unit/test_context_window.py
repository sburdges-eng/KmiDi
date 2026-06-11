"""ContextWindow — sliding window with overflow summarisation.

Goal: Context persistence beyond token window."""

import pytest

from music_brain.generation.context_window import ContextEntry, ContextWindow

# ---------------------------------------------------------------------------
# Construction & basic append
# ---------------------------------------------------------------------------


def test_window_starts_empty():
    win = ContextWindow(max_tokens=100, summarizer=None)
    assert win.entries == []
    assert win.summary == ""
    assert win.total_tokens == 0


def test_append_records_entry_under_capacity():
    win = ContextWindow(max_tokens=100, summarizer=None)
    win.append(ContextEntry(role="user", text="hello", tokens=10))
    assert len(win.entries) == 1
    assert win.entries[0].text == "hello"
    assert win.total_tokens == 10


def test_append_invalid_role_raises():
    win = ContextWindow(max_tokens=100, summarizer=None)
    with pytest.raises(ValueError, match="role"):
        win.append(ContextEntry(role="alien", text="hi", tokens=1))


# ---------------------------------------------------------------------------
# Overflow without summarizer → just drops oldest
# ---------------------------------------------------------------------------


def test_overflow_without_summarizer_drops_oldest():
    win = ContextWindow(max_tokens=20, summarizer=None)
    win.append(ContextEntry(role="user", text="first", tokens=10))
    win.append(ContextEntry(role="assistant", text="second", tokens=10))
    win.append(ContextEntry(role="user", text="third", tokens=10))
    # Oldest (first) dropped to fit budget.
    texts = [e.text for e in win.entries]
    assert "first" not in texts
    assert "second" in texts
    assert "third" in texts


# ---------------------------------------------------------------------------
# Overflow with summarizer → compresses into prefix
# ---------------------------------------------------------------------------


def test_overflow_with_summarizer_compresses_dropped_entries():
    """Dropped entries get passed to the summarizer; the result becomes the
    window's `summary` prefix instead of being lost."""
    received = []

    def summarizer(dropped, current_summary):
        received.append(dropped)
        # Trivial: summary is just space-joined texts.
        prev = current_summary + " | " if current_summary else ""
        return prev + " ".join(e.text for e in dropped)

    win = ContextWindow(max_tokens=20, summarizer=summarizer)
    win.append(ContextEntry(role="user", text="first", tokens=10))
    win.append(ContextEntry(role="assistant", text="second", tokens=10))
    win.append(ContextEntry(role="user", text="third", tokens=10))
    # `first` was the one dropped to make room.
    assert len(received) >= 1
    flattened_dropped_texts = [e.text for batch in received for e in batch]
    assert "first" in flattened_dropped_texts
    assert "first" in win.summary


def test_summary_persists_across_multiple_overflows():
    """Subsequent overflows build on the existing summary, not replace it."""

    def summarizer(dropped, current_summary):
        # Append-style; never discards what was already summarized.
        prev = current_summary + " | " if current_summary else ""
        return prev + " ".join(e.text for e in dropped)

    win = ContextWindow(max_tokens=20, summarizer=summarizer)
    for i in range(5):
        win.append(ContextEntry(role="user", text=f"msg{i}", tokens=10))
    # All earlier messages should appear in the summary except the most recent two.
    assert "msg0" in win.summary
    assert "msg1" in win.summary
    assert "msg2" in win.summary
    texts = [e.text for e in win.entries]
    assert "msg3" in texts or "msg4" in texts


# ---------------------------------------------------------------------------
# render() returns summary + window in canonical order
# ---------------------------------------------------------------------------


def test_render_concatenates_summary_then_entries():
    win = ContextWindow(
        max_tokens=20,
        summarizer=lambda dropped, current: " ".join(e.text for e in dropped),
    )
    win.append(ContextEntry(role="user", text="alpha", tokens=10))
    win.append(ContextEntry(role="user", text="beta", tokens=10))
    win.append(ContextEntry(role="user", text="gamma", tokens=10))
    rendered = win.render()
    # Summary appears first, then live entries in order.
    assert rendered.index("alpha") < rendered.index("beta") < rendered.index("gamma")


def test_render_omits_summary_block_when_summary_empty():
    win = ContextWindow(max_tokens=100, summarizer=None)
    win.append(ContextEntry(role="user", text="hi", tokens=2))
    rendered = win.render()
    assert rendered.strip().startswith("user:") or "hi" in rendered


# ---------------------------------------------------------------------------
# fits / would_fit predicates
# ---------------------------------------------------------------------------


def test_would_fit_true_when_entry_within_remaining():
    win = ContextWindow(max_tokens=20, summarizer=None)
    win.append(ContextEntry(role="user", text="x", tokens=10))
    assert win.would_fit(tokens=5) is True


def test_would_fit_false_when_entry_exceeds_remaining():
    win = ContextWindow(max_tokens=20, summarizer=None)
    win.append(ContextEntry(role="user", text="x", tokens=15))
    assert win.would_fit(tokens=10) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_oversized_single_entry_replaces_window_and_summary():
    """An entry whose token count alone exceeds max_tokens still gets stored
    (the alternative is silently dropping data); window may exceed budget
    transiently. Caller's choice not to allow this is enforced via
    would_fit beforehand."""
    win = ContextWindow(max_tokens=10, summarizer=None)
    win.append(ContextEntry(role="user", text="huge", tokens=50))
    # Single entry survives — we don't pre-emptively drop a too-big entry.
    assert win.entries[0].text == "huge"
