"""Tests for ``music_brain.tokenizer_consistency``."""

from __future__ import annotations

from pathlib import Path

from music_brain.tokenizer_consistency import (
    ConsistencyReport,
    TokenizerFingerprint,
    compare,
    hash_merges,
    hash_vocab,
)


def _fp(**overrides) -> TokenizerFingerprint:
    defaults = dict(
        name="bpe-32k",
        vocab_size=32000,
        vocab_hash=hash_vocab({"<pad>": 0, "<bos>": 1, "a": 2, "b": 3}),
        merges_hash=hash_merges([("a", "b"), ("ab", "c")]),
        special_tokens={"pad": 0, "bos": 1, "eos": 2},
        version="1",
    )
    defaults.update(overrides)
    return TokenizerFingerprint(**defaults)


# ----------------------------------------------------------------------
# Hash helpers
# ----------------------------------------------------------------------

def test_hash_vocab_is_order_independent() -> None:
    a = hash_vocab({"x": 0, "y": 1, "z": 2})
    b = hash_vocab({"z": 2, "y": 1, "x": 0})
    assert a == b


def test_hash_vocab_distinguishes_swapped_ids() -> None:
    """Same surface forms but swapped IDs must produce different hashes."""
    a = hash_vocab({"x": 0, "y": 1})
    b = hash_vocab({"x": 1, "y": 0})
    assert a != b


def test_hash_merges_order_is_significant() -> None:
    a = hash_merges([("a", "b"), ("c", "d")])
    b = hash_merges([("c", "d"), ("a", "b")])
    assert a != b, "BPE merge order changes encoding behavior"


# ----------------------------------------------------------------------
# Fingerprint serialisation
# ----------------------------------------------------------------------

def test_fingerprint_json_roundtrip() -> None:
    fp = _fp()
    decoded = TokenizerFingerprint.from_json(fp.to_json())
    assert decoded == fp
    assert decoded.fingerprint() == fp.fingerprint()


def test_fingerprint_file_roundtrip(tmp_path: Path) -> None:
    fp = _fp()
    path = tmp_path / "tokenizer.fingerprint.json"
    fp.save(path)
    loaded = TokenizerFingerprint.load(path)
    assert loaded == fp


def test_fingerprint_summary_is_short_hex() -> None:
    s = _fp().fingerprint()
    assert len(s) == 16
    assert all(c in "0123456789abcdef" for c in s)


# ----------------------------------------------------------------------
# compare()
# ----------------------------------------------------------------------

def test_identical_fingerprints_are_consistent() -> None:
    report = compare(_fp(), _fp())
    assert isinstance(report, ConsistencyReport)
    assert bool(report) is True
    assert report.mismatches == ()


def test_name_diff_detected() -> None:
    report = compare(_fp(name="A"), _fp(name="B"))
    assert not report
    assert any("name" in m for m in report.mismatches)


def test_vocab_size_diff_detected() -> None:
    report = compare(_fp(vocab_size=100), _fp(vocab_size=200))
    assert not report
    assert any("vocab_size" in m for m in report.mismatches)


def test_vocab_hash_diff_detected() -> None:
    report = compare(
        _fp(vocab_hash="aaaaaaaaaaaaaaaa"),
        _fp(vocab_hash="bbbbbbbbbbbbbbbb"),
    )
    assert not report
    assert any("vocab_hash" in m for m in report.mismatches)


def test_merges_hash_diff_detected() -> None:
    report = compare(
        _fp(merges_hash="aaaaaaaaaaaaaaaa"),
        _fp(merges_hash="bbbbbbbbbbbbbbbb"),
    )
    assert not report
    assert any("merges_hash" in m for m in report.mismatches)


def test_special_tokens_diff_detected() -> None:
    report = compare(
        _fp(special_tokens={"pad": 0, "bos": 1}),
        _fp(special_tokens={"pad": 0, "bos": 2}),  # bos id moved
    )
    assert not report
    assert any("special_tokens" in m for m in report.mismatches)


def test_version_diff_detected() -> None:
    report = compare(_fp(version="1"), _fp(version="2"))
    assert not report
    assert any("version" in m for m in report.mismatches)


def test_multiple_diffs_all_reported() -> None:
    report = compare(
        _fp(name="A", vocab_size=100, version="1"),
        _fp(name="B", vocab_size=200, version="2"),
    )
    assert not report
    # All three mismatches should appear in the report.
    text = " ".join(report.mismatches)
    assert "name" in text and "vocab_size" in text and "version" in text


def test_non_bpe_tokenizer_with_none_merges_hash() -> None:
    """A unigram/char tokenizer has merges_hash=None; equality still works."""
    fp = _fp(merges_hash=None)
    decoded = TokenizerFingerprint.from_json(fp.to_json())
    assert decoded == fp
    assert compare(fp, fp).consistent is True
