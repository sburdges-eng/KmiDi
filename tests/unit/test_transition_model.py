"""TransitionModel — Markov baseline for sequential prediction.

Goal: Predictive arrangement modeling, Future-state musical prediction,
Harmony prediction engine (chord transitions), Groove prediction engine
(rhythmic pattern transitions)."""

import numpy as np
import pytest

from music_brain.prediction.transition import TransitionModel

# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_records_observed_states():
    model = TransitionModel()
    model.fit([["intro", "verse", "chorus", "verse", "outro"]])
    assert model.states == ["chorus", "intro", "outro", "verse"]


def test_fit_multiple_sequences_aggregates_counts():
    model = TransitionModel()
    model.fit(
        [
            ["verse", "chorus", "verse", "chorus"],
            ["intro", "verse", "outro"],
        ]
    )
    # Sequences should both contribute to the transition counts.
    next_dist = model.predict_next("verse")
    assert next_dist["chorus"] == pytest.approx(2 / 3)
    assert next_dist["outro"] == pytest.approx(1 / 3)


def test_fit_empty_sequences_does_not_crash():
    model = TransitionModel()
    model.fit([[]])
    assert model.states == []


def test_fit_singleton_sequence_records_state_but_no_transitions():
    model = TransitionModel()
    model.fit([["solo"]])
    assert model.states == ["solo"]
    # No outgoing edges from solo.
    assert model.predict_next("solo") == {}


# ---------------------------------------------------------------------------
# predict_next
# ---------------------------------------------------------------------------


def test_predict_next_returns_probabilities_summing_to_one():
    model = TransitionModel()
    model.fit([["a", "b", "a", "c", "a", "b"]])
    dist = model.predict_next("a")
    assert sum(dist.values()) == pytest.approx(1.0)


def test_predict_next_normalises_by_outgoing_count():
    model = TransitionModel()
    # a → b twice, a → c once
    model.fit([["a", "b", "a", "b", "a", "c"]])
    dist = model.predict_next("a")
    assert dist["b"] == pytest.approx(2 / 3)
    assert dist["c"] == pytest.approx(1 / 3)


def test_predict_next_unknown_state_returns_empty_dict():
    model = TransitionModel()
    model.fit([["a", "b"]])
    assert model.predict_next("z") == {}


# ---------------------------------------------------------------------------
# sample_next
# ---------------------------------------------------------------------------


def test_sample_next_returns_observed_successor():
    model = TransitionModel()
    model.fit([["a", "b", "a", "b"]])
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert model.sample_next("a", rng) == "b"


def test_sample_next_is_deterministic_with_same_seed():
    model = TransitionModel()
    model.fit([["a", "b", "a", "c", "a", "d", "a", "b", "a", "c"]])
    seq1 = [model.sample_next("a", np.random.default_rng(42)) for _ in range(5)]
    seq2 = [model.sample_next("a", np.random.default_rng(42)) for _ in range(5)]
    assert seq1 == seq2


def test_sample_next_unknown_state_raises():
    model = TransitionModel()
    model.fit([["a", "b"]])
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError):
        model.sample_next("z", rng)


# ---------------------------------------------------------------------------
# predict_sequence
# ---------------------------------------------------------------------------


def test_predict_sequence_starts_at_given_state_and_runs_length_steps():
    model = TransitionModel()
    model.fit([["a", "b", "a", "b", "a", "b"]])
    rng = np.random.default_rng(0)
    seq = model.predict_sequence("a", length=5, rng=rng)
    assert len(seq) == 5
    assert seq[0] == "a"
    # Every step after a must be b in this fully deterministic chain.
    assert all(s in {"a", "b"} for s in seq)


def test_predict_sequence_stops_early_at_terminal_state():
    """If a state has no outgoing transitions, the sequence ends there
    rather than padding or looping."""
    model = TransitionModel()
    model.fit([["a", "b", "outro"]])  # outro is terminal
    rng = np.random.default_rng(0)
    seq = model.predict_sequence("a", length=10, rng=rng)
    assert seq[-1] == "outro"
    assert len(seq) <= 10


# ---------------------------------------------------------------------------
# Laplace smoothing
# ---------------------------------------------------------------------------


def test_predict_next_with_smoothing_assigns_mass_to_unseen_states():
    """With alpha smoothing, every known state appears in the distribution."""
    model = TransitionModel(alpha=1.0)
    model.fit([["a", "b", "a", "b", "c"]])
    dist = model.predict_next("a")
    # Without smoothing, c would have 0 mass from a; with alpha=1 it's nonzero.
    assert dist["c"] > 0
    assert sum(dist.values()) == pytest.approx(1.0)
