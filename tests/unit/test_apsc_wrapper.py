"""Unit tests for APSC (Audio-Permutational Self-Consistency) wrapper."""

import sys
from pathlib import Path


# Prefer penta_core on path (e.g. PYTHONPATH=music_brain); else load apsc_wrapper as submodule
_root = Path(__file__).resolve().parents[2]
_music_brain = _root / "music_brain"
if str(_music_brain) not in sys.path:
    sys.path.insert(0, str(_music_brain))
try:
    from penta_core.ml.apsc_wrapper import (
        permute_stem_orders,
        aggregate_json_majority,
        run_apsc,
    )
except ImportError:
    from music_brain.penta_core.ml import apsc_wrapper

    permute_stem_orders = apsc_wrapper.permute_stem_orders
    aggregate_json_majority = apsc_wrapper.aggregate_json_majority
    run_apsc = apsc_wrapper.run_apsc


class TestPermuteStemOrders:
    """Test permutation count and content."""

    def test_zero_stems(self):
        assert permute_stem_orders(0) == []

    def test_one_stem(self):
        assert permute_stem_orders(1) == [(0,)]

    def test_three_stems_six_permutations(self):
        orders = permute_stem_orders(3)
        assert len(orders) == 6
        assert set(orders) == {
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        }

    def test_four_stems_24_permutations(self):
        orders = permute_stem_orders(4)
        assert len(orders) == 24


class TestAggregateJsonMajority:
    """Test majority aggregation of orchestration decisions."""

    def test_empty_responses(self):
        assert aggregate_json_majority([]) == []

    def test_single_response(self):
        resp = [
            {"track_id": "bass", "action": "eq", "params": {"freq": 100}},
            {"track_id": "drums", "action": "gain", "params": {"db": -2}},
        ]
        assert aggregate_json_majority([resp]) == resp

    def test_majority_wins(self):
        r1 = [{"track_id": "bass", "action": "eq", "params": {"freq": 100}}]
        r2 = [{"track_id": "bass", "action": "eq", "params": {"freq": 100}}]
        r3 = [{"track_id": "bass", "action": "gain", "params": {"db": -1}}]
        out = aggregate_json_majority([r1, r2, r3])
        assert len(out) == 1
        assert out[0]["track_id"] == "bass"
        assert out[0]["action"] == "eq"
        assert out[0]["params"] == {"freq": 100}

    def test_multiple_tracks_majority(self):
        r1 = [
            {"track_id": "bass", "action": "eq", "params": {}},
            {"track_id": "drums", "action": "gate", "params": {"thresh": 0.5}},
        ]
        r2 = [
            {"track_id": "bass", "action": "eq", "params": {}},
            {"track_id": "drums", "action": "gate", "params": {"thresh": 0.5}},
        ]
        r3 = [
            {"track_id": "bass", "action": "gain", "params": {}},
            {"track_id": "drums", "action": "compress", "params": {}},
        ]
        out = aggregate_json_majority([r1, r2, r3])
        assert len(out) == 2
        by_track = {d["track_id"]: d for d in out}
        assert by_track["bass"]["action"] == "eq"
        assert by_track["drums"]["action"] == "gate"
        assert by_track["drums"]["params"] == {"thresh": 0.5}


class TestRunApsc:
    """Test run_apsc with stub model_invoke."""

    def test_empty_stems(self):
        def invoke(prompt, stems):
            return []

        assert run_apsc("prompt", [], invoke) == []

    def test_single_stem_one_permutation(self):
        def invoke(prompt, stems):
            assert len(stems) == 1
            return [{"track_id": "bass", "action": "eq", "params": {}}]

        stems = [("bass", "ref1")]
        out = run_apsc("prompt", stems, invoke, num_permutations=1)
        assert len(out) == 1
        assert out[0]["track_id"] == "bass"

    def test_three_stems_caps_permutations(self):
        calls = []

        def invoke(prompt, stems):
            calls.append(stems)
            return [
                {"track_id": stems[i][0], "action": "eq", "params": {}} for i in range(len(stems))
            ]

        stems = [("bass", 1), ("drums", 2), ("synth", 3)]
        out = run_apsc("prompt", stems, invoke, num_permutations=3)
        assert len(calls) == 3
        assert len(out) == 3
        track_ids = {d["track_id"] for d in out}
        assert track_ids == {"bass", "drums", "synth"}
