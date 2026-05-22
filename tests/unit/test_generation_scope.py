"""GenerationScope context manager (Goal: Generation scope isolation,
Realtime audition-safe regeneration, Realtime rollback system)."""

import pytest

from music_brain.generation.scope import GenerationScope, RollbackError


# ---------------------------------------------------------------------------
# Basic snapshot / commit / rollback
# ---------------------------------------------------------------------------


def test_scope_returns_snapshot_of_state_on_enter():
    parent = {"chord": "Am", "tempo": 120, "tracks": ["lead", "bass"]}
    with GenerationScope(parent) as scope:
        # Inside the scope, mutations to `scope.state` don't reach parent.
        scope.state["chord"] = "F"
        scope.state["tracks"].append("pad")
    # Default exit behaviour is rollback — parent unchanged.
    assert parent == {"chord": "Am", "tempo": 120, "tracks": ["lead", "bass"]}


def test_scope_commit_propagates_changes_to_parent():
    parent = {"chord": "Am", "tempo": 120}
    with GenerationScope(parent) as scope:
        scope.state["chord"] = "F"
        scope.commit()
    assert parent == {"chord": "F", "tempo": 120}


def test_scope_explicit_rollback_discards_changes():
    parent = {"chord": "Am"}
    with GenerationScope(parent) as scope:
        scope.state["chord"] = "F"
        scope.rollback()
    assert parent == {"chord": "Am"}


def test_scope_commit_then_mutations_after_commit_dont_propagate():
    """Once committed, further in-scope mutations don't keep escaping."""
    parent = {"chord": "Am"}
    with GenerationScope(parent) as scope:
        scope.state["chord"] = "F"
        scope.commit()
        # Post-commit mutation: should not silently land in parent.
        scope.state["chord"] = "G"
    assert parent == {"chord": "F"}


# ---------------------------------------------------------------------------
# Nesting & isolation
# ---------------------------------------------------------------------------


def test_nested_scopes_independent_rollback():
    parent = {"chord": "Am"}
    with GenerationScope(parent) as outer:
        outer.state["chord"] = "Dm"
        with GenerationScope(outer.state) as inner:
            inner.state["chord"] = "G"
            # inner rolls back on exit (default)
        # outer still sees its own change
        assert outer.state["chord"] == "Dm"
        outer.commit()
    assert parent == {"chord": "Dm"}


def test_nested_commit_propagates_to_outer_only():
    parent = {"chord": "Am"}
    with GenerationScope(parent) as outer:
        with GenerationScope(outer.state) as inner:
            inner.state["chord"] = "G"
            inner.commit()
        # outer now sees inner's commit
        assert outer.state["chord"] == "G"
        # but parent doesn't until outer commits
        assert parent["chord"] == "Am"
        outer.commit()
    assert parent["chord"] == "G"


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


def test_exception_inside_scope_triggers_rollback_not_commit():
    """If the body raises, the scope must roll back even if commit was called."""
    parent = {"chord": "Am"}
    with pytest.raises(RuntimeError, match="boom"):
        with GenerationScope(parent) as scope:
            scope.state["chord"] = "F"
            scope.commit()
            raise RuntimeError("boom")
    # Auto-rollback wins over an earlier commit.
    assert parent == {"chord": "Am"}


def test_double_commit_raises():
    """Committing twice is an API misuse."""
    parent = {"chord": "Am"}
    with pytest.raises(RollbackError, match="already"):
        with GenerationScope(parent) as scope:
            scope.commit()
            scope.commit()


def test_commit_outside_scope_raises():
    """commit() called after exiting the with-block must error."""
    parent = {"chord": "Am"}
    scope = GenerationScope(parent)
    with scope:
        pass
    with pytest.raises(RollbackError, match="closed"):
        scope.commit()


# ---------------------------------------------------------------------------
# Deep-copy semantics for nested mutables
# ---------------------------------------------------------------------------


def test_deep_copy_prevents_in_place_mutation_of_parent_list():
    """Mutating a list inside the scope must not touch the parent's list
    when rollback is the outcome."""
    parent = {"tracks": ["lead", "bass"]}
    with GenerationScope(parent) as scope:
        scope.state["tracks"].append("pad")
        # No commit → rollback.
    assert parent["tracks"] == ["lead", "bass"]


def test_deep_copy_prevents_in_place_mutation_after_commit_other_keys():
    """Commit copies its own snapshot into parent, not a shared reference,
    so further post-commit mutations on the scope's state don't leak."""
    parent = {"tracks": ["lead", "bass"]}
    with GenerationScope(parent) as scope:
        scope.state["tracks"].append("pad")
        scope.commit()
        scope.state["tracks"].append("DO_NOT_LEAK")
    assert parent["tracks"] == ["lead", "bass", "pad"]


# ---------------------------------------------------------------------------
# is_active / has_committed introspection
# ---------------------------------------------------------------------------


def test_scope_is_active_inside_with_block():
    parent = {"chord": "Am"}
    scope = GenerationScope(parent)
    assert scope.is_active is False
    with scope:
        assert scope.is_active is True
    assert scope.is_active is False


def test_scope_has_committed_after_explicit_commit():
    parent = {"chord": "Am"}
    with GenerationScope(parent) as scope:
        assert scope.has_committed is False
        scope.commit()
        assert scope.has_committed is True
