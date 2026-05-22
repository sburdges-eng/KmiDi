"""Tests for ``music_brain.sandbox``.

Uses ``/bin/echo``, ``/bin/sleep``, and ``/usr/bin/env`` as test
executables — all are present on the macOS / Linux dev hosts the project
targets. If a test host lacks these, pytest will skip the affected tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from music_brain.sandbox import (
    SandboxResult,
    SandboxViolation,
    make_policy,
    run_sandboxed,
)


def _binary(*names: str) -> Path:
    """Return the first binary found from ``names``, or skip the test."""
    for n in names:
        found = shutil.which(n)
        if found:
            return Path(found).resolve(strict=False)
    pytest.skip(f"none of {names} found on PATH")


def test_make_policy_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_policy(allowed_executables=[], allowed_cwds=[tmp_path], wall_timeout_s=0)
    with pytest.raises(ValueError):
        make_policy(allowed_executables=[], allowed_cwds=[tmp_path],
                    wall_timeout_s=1, memory_limit_bytes=0)
    with pytest.raises(ValueError):
        make_policy(allowed_executables=[], allowed_cwds=[tmp_path],
                    wall_timeout_s=1, cpu_seconds=0)


def test_run_returns_stdout_on_success(tmp_path: Path) -> None:
    echo = _binary("echo")
    policy = make_policy(
        allowed_executables=[echo],
        allowed_cwds=[tmp_path],
        wall_timeout_s=5.0,
    )
    result = run_sandboxed(echo, ["hello", "sandbox"], cwd=tmp_path, policy=policy)
    assert isinstance(result, SandboxResult)
    assert result.ok
    assert result.returncode == 0
    assert result.timed_out is False
    assert b"hello sandbox" in result.stdout


def test_denied_executable_raises(tmp_path: Path) -> None:
    echo = _binary("echo")
    other = _binary("true", "ls")
    if other == echo:
        pytest.skip("need two distinct executables for this test")
    policy = make_policy(
        allowed_executables=[other],
        allowed_cwds=[tmp_path],
        wall_timeout_s=1.0,
    )
    with pytest.raises(SandboxViolation, match="executable not in allowlist"):
        run_sandboxed(echo, [], cwd=tmp_path, policy=policy)


def test_denied_cwd_raises(tmp_path: Path) -> None:
    echo = _binary("echo")
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    policy = make_policy(
        allowed_executables=[echo],
        allowed_cwds=[tmp_path],  # forbidden is NOT in the allowlist
        wall_timeout_s=1.0,
    )
    with pytest.raises(SandboxViolation, match="cwd not in allowlist"):
        run_sandboxed(echo, [], cwd=forbidden, policy=policy)


def test_missing_executable_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "definitely-not-here"
    policy = make_policy(
        allowed_executables=[bogus],
        allowed_cwds=[tmp_path],
        wall_timeout_s=1.0,
    )
    with pytest.raises(SandboxViolation, match="does not exist"):
        run_sandboxed(bogus, [], cwd=tmp_path, policy=policy)


def test_timeout_sets_timed_out_flag(tmp_path: Path) -> None:
    sleep = _binary("sleep")
    policy = make_policy(
        allowed_executables=[sleep],
        allowed_cwds=[tmp_path],
        wall_timeout_s=0.2,
    )
    result = run_sandboxed(sleep, ["5"], cwd=tmp_path, policy=policy)
    assert result.timed_out is True
    assert result.ok is False
    assert result.returncode == -1


def test_env_is_scrubbed_unless_passthrough(tmp_path: Path) -> None:
    env_bin = _binary("env")
    # Without passthrough — KMIDI_TEST_SECRET should not appear.
    import os
    os.environ["KMIDI_TEST_SECRET"] = "must-not-leak"
    try:
        policy = make_policy(
            allowed_executables=[env_bin],
            allowed_cwds=[tmp_path],
            wall_timeout_s=2.0,
        )
        result = run_sandboxed(env_bin, [], cwd=tmp_path, policy=policy)
        assert result.ok
        assert b"KMIDI_TEST_SECRET" not in result.stdout

        # With passthrough — it should appear.
        policy2 = make_policy(
            allowed_executables=[env_bin],
            allowed_cwds=[tmp_path],
            wall_timeout_s=2.0,
            env_passthrough=["KMIDI_TEST_SECRET"],
        )
        result2 = run_sandboxed(env_bin, [], cwd=tmp_path, policy=policy2)
        assert result2.ok
        assert b"KMIDI_TEST_SECRET=must-not-leak" in result2.stdout
    finally:
        os.environ.pop("KMIDI_TEST_SECRET", None)


def test_nonzero_exit_does_not_raise(tmp_path: Path) -> None:
    # Use `false` which exits 1; not an exception, just non-ok.
    false_bin = _binary("false")
    policy = make_policy(
        allowed_executables=[false_bin],
        allowed_cwds=[tmp_path],
        wall_timeout_s=1.0,
    )
    result = run_sandboxed(false_bin, [], cwd=tmp_path, policy=policy)
    assert result.returncode != 0
    assert result.ok is False
    assert result.timed_out is False


def test_stdin_is_forwarded(tmp_path: Path) -> None:
    cat = _binary("cat")
    policy = make_policy(
        allowed_executables=[cat],
        allowed_cwds=[tmp_path],
        wall_timeout_s=2.0,
    )
    result = run_sandboxed(cat, [], cwd=tmp_path, policy=policy,
                            stdin=b"hello\nfrom stdin\n")
    assert result.ok
    assert b"hello" in result.stdout
    assert b"from stdin" in result.stdout
