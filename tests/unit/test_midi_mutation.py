"""Tests for ``music_brain.midi_mutation``."""

from __future__ import annotations

import pytest

from music_brain.midi_mutation import (
    AddNote,
    MidiMutationEngine,
    MidiNote,
    MoveNote,
    MutationError,
    RemoveNote,
    SetPitch,
    SetVelocity,
)


def _note(id_: str = "n1", **overrides) -> MidiNote:
    defaults = dict(
        id=id_,
        pitch=60,
        velocity=100,
        start_tick=0,
        duration_ticks=480,
        channel=0,
    )
    defaults.update(overrides)
    return MidiNote(**defaults)


# ----------------------------------------------------------------------
# Single-mutation behaviour
# ----------------------------------------------------------------------

def test_add_note_inserts() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    assert len(eng) == 1
    assert "n1" in eng


def test_add_note_duplicate_id_raises() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    with pytest.raises(MutationError, match="already exists"):
        eng.apply(AddNote(_note()))


def test_remove_note_drops_entry() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    eng.apply(RemoveNote("n1"))
    assert "n1" not in eng


def test_remove_unknown_raises() -> None:
    eng = MidiMutationEngine()
    with pytest.raises(MutationError, match="unknown note"):
        eng.apply(RemoveNote("nope"))


def test_set_pitch_updates() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    eng.apply(SetPitch("n1", 72))
    got = eng.get("n1")
    assert got is not None and got.pitch == 72


def test_set_pitch_out_of_range_raises() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    with pytest.raises(MutationError, match="pitch out of range"):
        eng.apply(SetPitch("n1", 200))


def test_set_velocity_updates_and_validates() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    eng.apply(SetVelocity("n1", 0))
    got = eng.get("n1")
    assert got is not None and got.velocity == 0
    with pytest.raises(MutationError, match="velocity"):
        eng.apply(SetVelocity("n1", 128))


def test_move_note_updates_and_validates() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    eng.apply(MoveNote("n1", 960))
    got = eng.get("n1")
    assert got is not None and got.start_tick == 960
    with pytest.raises(MutationError, match=">= 0"):
        eng.apply(MoveNote("n1", -1))


def test_add_validates_each_field() -> None:
    eng = MidiMutationEngine()
    with pytest.raises(MutationError, match="pitch"):
        eng.apply(AddNote(_note(pitch=200)))
    with pytest.raises(MutationError, match="velocity"):
        eng.apply(AddNote(_note(velocity=-1)))
    with pytest.raises(MutationError, match="duration_ticks"):
        eng.apply(AddNote(_note(duration_ticks=0)))
    with pytest.raises(MutationError, match="start_tick"):
        eng.apply(AddNote(_note(start_tick=-1)))
    with pytest.raises(MutationError, match="channel"):
        eng.apply(AddNote(_note(channel=16)))


# ----------------------------------------------------------------------
# Transaction semantics
# ----------------------------------------------------------------------

def test_explicit_commit_keeps_changes() -> None:
    eng = MidiMutationEngine()
    eng.begin()
    eng.apply(AddNote(_note()))
    log = eng.commit()
    assert "n1" in eng
    assert len(log) == 1


def test_explicit_rollback_restores_prior_state() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note(id_="pre")))
    eng.begin()
    eng.apply(AddNote(_note(id_="added-in-txn")))
    eng.apply(RemoveNote("pre"))
    eng.rollback()
    # 'pre' restored, 'added-in-txn' gone.
    assert "pre" in eng
    assert "added-in-txn" not in eng
    assert eng.in_transaction() is False


def test_context_manager_auto_commits_on_clean_exit() -> None:
    eng = MidiMutationEngine()
    with eng.transaction() as txn:
        txn.apply(AddNote(_note()))
    assert "n1" in eng


def test_context_manager_auto_rollbacks_on_exception() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note(id_="anchor")))
    with pytest.raises(MutationError):
        with eng.transaction() as txn:
            txn.apply(AddNote(_note(id_="will-survive-mid-txn")))
            # This raises mid-transaction.
            txn.apply(RemoveNote("does-not-exist"))
    # anchor is back, the mid-txn add was rolled back.
    assert "anchor" in eng
    assert "will-survive-mid-txn" not in eng
    assert eng.in_transaction() is False


def test_double_begin_raises() -> None:
    eng = MidiMutationEngine()
    eng.begin()
    with pytest.raises(MutationError, match="already in progress"):
        eng.begin()
    eng.rollback()


def test_commit_without_begin_raises() -> None:
    eng = MidiMutationEngine()
    with pytest.raises(MutationError, match="no transaction"):
        eng.commit()


def test_rollback_without_begin_raises() -> None:
    eng = MidiMutationEngine()
    with pytest.raises(MutationError, match="no transaction"):
        eng.rollback()


def test_failure_mid_transaction_does_not_auto_rollback() -> None:
    """Caller chooses; the engine surfaces the error but keeps partial state."""
    eng = MidiMutationEngine()
    eng.begin()
    eng.apply(AddNote(_note(id_="a")))
    with pytest.raises(MutationError):
        eng.apply(RemoveNote("not-here"))
    # Partial state still present.
    assert "a" in eng
    assert eng.in_transaction() is True
    eng.rollback()
    assert "a" not in eng


def test_explicit_rollback_inside_with_block_supresses_auto_commit() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note(id_="anchor")))
    with eng.transaction() as txn:
        txn.apply(AddNote(_note(id_="ephemeral")))
        txn.rollback()
        # No further mutations after manual rollback.
    assert "anchor" in eng
    assert "ephemeral" not in eng


def test_notes_returns_sorted_snapshot() -> None:
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note(id_="late", start_tick=960)))
    eng.apply(AddNote(_note(id_="mid", start_tick=480)))
    eng.apply(AddNote(_note(id_="early", start_tick=0)))
    ordered = [n.id for n in eng.notes()]
    assert ordered == ["early", "mid", "late"]


def test_get_returns_copy_not_alias() -> None:
    """Mutating the returned note must not bypass the engine."""
    eng = MidiMutationEngine()
    eng.apply(AddNote(_note()))
    got = eng.get("n1")
    assert got is not None
    got.pitch = 0
    canonical = eng.get("n1")
    assert canonical is not None and canonical.pitch == 60
