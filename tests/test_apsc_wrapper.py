import pytest
from music_brain.penta_core.ml.apsc_wrapper import (
    permute_stem_orders,
    aggregate_json_majority,
    run_apsc
)

def test_permute_stem_orders():
    assert len(permute_stem_orders(3)) == 6
    assert len(permute_stem_orders(4)) == 24
    assert len(permute_stem_orders(0)) == 0

def test_aggregate_json_majority():
    responses = [
        [
            {"track_id": "bass", "action": "eq", "params": {"gain_db": -2}},
            {"track_id": "drums", "action": "comp", "params": {"ratio": 4}}
        ],
        [
            {"track_id": "bass", "action": "eq", "params": {"gain_db": -2}},
            {"track_id": "drums", "action": "comp", "params": {"ratio": 4}}
        ],
        [
            {"track_id": "bass", "action": "eq", "params": {"gain_db": 0}},  # minority
            {"track_id": "drums", "action": "comp", "params": {"ratio": 2}}   # minority
        ]
    ]
    result = aggregate_json_majority(responses)
    assert len(result) == 2
    # Convert result to a dict for easy checking
    res_dict = {d["track_id"]: d for d in result}
    
    assert res_dict["bass"]["action"] == "eq"
    assert res_dict["bass"]["params"] == {"gain_db": -2}
    assert res_dict["drums"]["action"] == "comp"
    assert res_dict["drums"]["params"] == {"ratio": 4}

def test_run_apsc():
    stems = [("bass", "bass_audio"), ("drums", "drums_audio"), ("synth", "synth_audio")]
    
    # We will mock the model such that it always votes for a specific action for "bass"
    # regardless of permutation, but changes it for "drums" based on position.
    def mock_model_invoke(prompt, ordered_stems):
        # Let's say if "bass" is first, it gives gain -2, else gain -1.
        # But we run enough permutations that gain -1 will be the majority (since it's first only 1/3 of the time).
        bass_pos = [s[0] for s in ordered_stems].index("bass")
        gain = -2 if bass_pos == 0 else -1
        
        return [{"track_id": "bass", "action": "eq", "params": {"gain": gain}}]
        
    result = run_apsc("Test prompt", stems, mock_model_invoke, num_permutations=6)
    
    assert len(result) == 1
    assert result[0]["track_id"] == "bass"
    assert result[0]["action"] == "eq"
    assert result[0]["params"] == {"gain": -1}
