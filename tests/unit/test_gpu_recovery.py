"""Tests for ``music_brain.gpu_recovery``."""

from __future__ import annotations

import pytest

from music_brain.gpu_recovery import gpu_recoverable, looks_like_gpu_failure


# ----------------------------------------------------------------------
# looks_like_gpu_failure
# ----------------------------------------------------------------------

def test_cuda_oom_message_matches() -> None:
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert looks_like_gpu_failure(exc) is True


def test_mps_oom_message_matches() -> None:
    exc = RuntimeError("MPS backend out of memory")
    assert looks_like_gpu_failure(exc) is True


def test_illegal_memory_access_matches() -> None:
    exc = RuntimeError("an illegal memory access was encountered")
    assert looks_like_gpu_failure(exc) is True


def test_unrelated_message_does_not_match() -> None:
    exc = ValueError("something else entirely")
    assert looks_like_gpu_failure(exc) is False


def test_match_is_case_insensitive() -> None:
    exc = RuntimeError("CUDA OUT OF MEMORY at line 42")
    assert looks_like_gpu_failure(exc) is True


# ----------------------------------------------------------------------
# gpu_recoverable
# ----------------------------------------------------------------------

def test_clean_call_returns_value() -> None:
    @gpu_recoverable()
    def f(x: int) -> int:
        return x * 2

    assert f(3) == 6


def test_non_matching_exception_re_raises_immediately() -> None:
    calls: list[int] = []

    @gpu_recoverable(cleanup=lambda: calls.append(0))
    def f() -> None:
        raise ValueError("not a GPU failure")

    with pytest.raises(ValueError):
        f()
    # Cleanup must NOT run for non-matching exceptions.
    assert calls == []


def test_matching_exception_triggers_cleanup_then_succeeds_on_retry() -> None:
    cleanup_calls: list[int] = []
    attempt = {"n": 0}

    def cleanup() -> None:
        cleanup_calls.append(attempt["n"])

    @gpu_recoverable(cleanup=cleanup, max_retries=1)
    def f() -> str:
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    assert f() == "ok"
    assert cleanup_calls == [1], "cleanup runs after the failing attempt"
    assert attempt["n"] == 2


def test_repeated_failure_eventually_re_raises() -> None:
    @gpu_recoverable(cleanup=None, max_retries=2)
    def f() -> None:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        f()


def test_max_retries_zero_re_raises_after_one_cleanup() -> None:
    calls: list[int] = []

    @gpu_recoverable(cleanup=lambda: calls.append(0), max_retries=0)
    def f() -> None:
        raise RuntimeError("device lost")

    with pytest.raises(RuntimeError):
        f()
    assert calls == [0], "cleanup runs once before re-raising"


def test_on_recover_hook_fires_with_exception_and_attempt() -> None:
    seen: list[tuple[str, int]] = []

    def recover(exc: BaseException, n: int) -> None:
        seen.append((type(exc).__name__, n))

    attempt = {"n": 0}

    @gpu_recoverable(max_retries=2,
                     on_recover=recover)
    def f() -> str:
        attempt["n"] += 1
        if attempt["n"] < 3:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    assert f() == "ok"
    # Two failures before success -> on_recover fires twice (attempts 1 and 2).
    assert seen == [("RuntimeError", 1), ("RuntimeError", 2)]


def test_cleanup_exception_does_not_mask_original() -> None:
    def bad_cleanup() -> None:
        raise OSError("cleanup itself failed")

    @gpu_recoverable(cleanup=bad_cleanup, max_retries=0)
    def f() -> None:
        raise RuntimeError("CUDA out of memory")

    # Original RuntimeError surfaces, not the cleanup OSError.
    with pytest.raises(RuntimeError, match="out of memory"):
        f()


def test_custom_predicate_replaces_default() -> None:
    class MyGPUError(Exception):
        pass

    attempt = {"n": 0}

    @gpu_recoverable(
        exception_predicate=lambda exc: isinstance(exc, MyGPUError),
        max_retries=1,
    )
    def f() -> str:
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise MyGPUError("custom matcher")
        return "ok"

    assert f() == "ok"


def test_negative_max_retries_rejected() -> None:
    with pytest.raises(ValueError):
        gpu_recoverable(max_retries=-1)
