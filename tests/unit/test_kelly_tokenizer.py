from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = REPO_ROOT / "final_kel" / "Brain" / "kellybrain" / "tokenizer.py"


def load_tokenizer_module():
    spec = importlib.util.spec_from_file_location("kmidi_tokenizer", TOKENIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simple_tokenizer_round_trip():
    module = load_tokenizer_module()
    tokenizer = module.SimpleMIDITokenizer()

    events = [
        {"type": "bar"},
        {"type": "note_on", "note": 60, "velocity": 92, "time_in_bar": 0.25},
    ]
    tokens = tokenizer.encode(events)

    assert tokens[0] == tokenizer.bos_token
    assert tokens[-1] == tokenizer.eos_token

    decoded = tokenizer.decode(tokens)
    assert decoded[0]["type"] == "bar"
    assert decoded[1]["note"] == 60


def test_miditok_wrapper_rejects_remote_loading_by_default(monkeypatch):
    module = load_tokenizer_module()

    class FakeREMI:
        @staticmethod
        def from_pretrained(source):
            raise AssertionError(f"unexpected remote load: {source}")

    monkeypatch.setitem(sys.modules, "miditok", types.SimpleNamespace(REMI=FakeREMI))
    monkeypatch.delenv("KMIDI_TOKENIZER_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("MIDITOK_TOKENIZER_PATH", raising=False)

    wrapper = module.MidiTokWrapper()

    assert wrapper.tokenizer is None
    assert "local" in wrapper.load_error


def test_miditok_wrapper_normalizes_maestro_tokens_for_language_model(tmp_path, monkeypatch):
    module = load_tokenizer_module()
    tokenizer_dir = tmp_path / "Maestro-REMI-bpe20k"
    tokenizer_dir.mkdir()

    class FakeTokenSequence:
        ids = [11, 12, 13]

    class FakeScore:
        def __init__(self):
            self.dumped_to = None

        def dump_midi(self, output_path):
            self.dumped_to = output_path

    class FakeTokenizer:
        special_tokens_ids = {"PAD": 0, "BOS": 1, "EOS": 2}

        def __call__(self, midi_path):
            assert midi_path.endswith(".mid")
            return [FakeTokenSequence()]

        def decode(self, token_batches):
            assert token_batches == [[1, 11, 12, 13, 2]]
            return FakeScore()

    class FakeREMI:
        @staticmethod
        def from_pretrained(source):
            assert source == str(tokenizer_dir)
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "miditok", types.SimpleNamespace(REMI=FakeREMI))

    wrapper = module.MidiTokWrapper(tokenizer_path=tokenizer_dir)
    encoded = wrapper.encode_for_language_model(tmp_path / "example.mid")

    assert encoded == [1, 11, 12, 13, 2]

    output_path = tmp_path / "decoded.mid"
    score = wrapper.decode(encoded, output_path=output_path)
    assert score.dumped_to == str(output_path)
