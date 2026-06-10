"""
APSC (Audio-Permutational Self-Consistency) Multi-Stem Orchestrator Wrapper.

Training-free prompt-intercept middleware to mitigate positional bias when the LALM
processes multiple audio stems. Duplicates the prompt with permuted stem order, runs
parallel (or batched) inference at low temperature, then aggregates outputs via
majority vote so the final decision is not dominated by "lost in the middle" effects.

Orchestration output schema (for majority vote): list of
  {"track_id": str, "action": str, "params": dict}
e.g. [{"track_id": "bass", "action": "eq", "params": {"freq": 100, "gain_db": -2}}].
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Minimal schema: per-track decision for aggregation.
# Each model response can be List[Dict] with keys track_id, action, params.
ORCHESTRATION_DECISION_KEYS = ("track_id", "action", "params")


def permute_stem_orders(n_stems: int) -> List[Tuple[int, ...]]:
    """
    Return all permutations of stem indices for APSC (or a subset if n_stems is large).

    For n_stems=3 this returns 6 orderings; for n_stems=4, 24. Caller may cap.
    """
    if n_stems <= 0:
        return []
    indices = tuple(range(n_stems))
    return list(itertools.permutations(indices))


def _decision_signature(d: Dict[str, Any]) -> Tuple[Tuple[str, str], Tuple[Tuple[str, Any], ...]]:
    """Hashable signature for a single decision for majority vote."""
    action = d.get("action", "")
    params = d.get("params") or {}
    params_t = tuple(sorted((k, params[k]) for k in sorted(params.keys())))
    return ((d.get("track_id") or "", action), params_t)


def aggregate_json_majority(
    responses: List[List[Dict[str, Any]]],
    track_id_key: str = "track_id",
    action_key: str = "action",
    params_key: str = "params",
) -> List[Dict[str, Any]]:
    """
    Aggregate multiple orchestration responses (one per permutation) by majority vote.

    Each response is a list of decisions: [{"track_id": str, "action": str, "params": dict}, ...].
    For each track_id, the (action, params) pair that appears most often across responses
    is chosen. Returns a single list of decisions (one per track_id that appeared).
    """
    if not responses:
        return []

    # Collect (track_id, (action, params)) from all responses
    by_track: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for resp in responses:
        if not isinstance(resp, list):
            continue
        for item in resp:
            if not isinstance(item, dict):
                continue
            tid = item.get(track_id_key) or ""
            act = item.get(action_key) or ""
            prm = item.get(params_key)
            if prm is None:
                prm = {}
            if not isinstance(prm, dict):
                prm = {}
            if tid not in by_track:
                by_track[tid] = []
            by_track[tid].append((act, prm))

    # Majority vote per track_id: pick most common (action, params)
    result: List[Dict[str, Any]] = []
    for track_id in sorted(by_track.keys()):
        pairs = by_track[track_id]

        # Make (action, frozenset of params items) hashable for Counter
        def _key(act: str, prm: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
            return (act, tuple(sorted((k, prm[k]) for k in sorted(prm.keys()))))

        counter: Counter[Tuple[str, Tuple[Tuple[str, Any], ...]]] = Counter()
        for act, prm in pairs:
            counter[_key(act, prm)] += 1
        (action, params_t), _ = counter.most_common(1)[0]
        params = dict(params_t)
        result.append(
            {
                track_id_key: track_id,
                action_key: action,
                params_key: params,
            }
        )
    return result


def run_apsc(
    prompt: str,
    stems: Sequence[Tuple[str, Any]],
    model_invoke: Callable[[str, Sequence[Tuple[str, Any]]], Any],
    temperature: float = 0.2,
    num_permutations: Optional[int] = 6,
) -> List[Dict[str, Any]]:
    """
    Run APSC: permute stem order, call model for each permutation, aggregate by majority.

    Args:
        prompt: User prompt (e.g. "Compare Bass, Drums, Synth and fix masking").
        stems: List of (track_id, audio_ref) in canonical order. audio_ref is passed
               through to model_invoke; can be tokens, path, or embedding ref.
        model_invoke: Callable(prompt, ordered_stems) -> response. response should be
                      a list of dicts with track_id, action, params for aggregation.
        temperature: Not used here; pass through to your model if model_invoke wraps it.
        num_permutations: Max number of permutations to run (default 6). If None, use all.

    Returns:
        Single list of decisions from aggregate_json_majority.
    """
    n = len(stems)
    if n == 0:
        return []
    orders = permute_stem_orders(n)
    if num_permutations is not None and len(orders) > num_permutations:
        orders = orders[:num_permutations]
    responses: List[List[Dict[str, Any]]] = []
    for perm in orders:
        ordered_stems = [stems[i] for i in perm]
        out = model_invoke(prompt, ordered_stems)
        if isinstance(out, list):
            responses.append(out)
        elif isinstance(out, dict) and "decisions" in out:
            responses.append(out["decisions"])
        else:
            responses.append([])
    return aggregate_json_majority(responses)
