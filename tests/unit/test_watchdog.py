"""Tests for ``music_brain.watchdog.Watchdog``.

The tests drive ``_check_once`` directly so they are deterministic
(no sleeps, no real threads needed for timing assertions).
"""

from __future__ import annotations

import time
from typing import List

import pytest

from music_brain.watchdog import Watchdog


def test_register_requires_positive_timeout() -> None:
    wd = Watchdog()
    with pytest.raises(ValueError):
        wd.register("w", 0.0, lambda _n: None)
    with pytest.raises(ValueError):
        wd.register("w", -1.0, lambda _n: None)


def test_register_requires_non_empty_name() -> None:
    wd = Watchdog()
    with pytest.raises(ValueError):
        wd.register("", 1.0, lambda _n: None)


def test_poll_interval_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Watchdog(poll_interval_s=0)


def test_heartbeat_resets_age() -> None:
    wd = Watchdog()
    fired: List[str] = []
    wd.register("worker", 0.001, lambda n: fired.append(n))
    # Sleep past the stale window then heartbeat — should not have fired yet.
    time.sleep(0.005)
    wd.heartbeat("worker")
    wd._check_once()
    assert fired == []


def test_callback_fires_once_per_stale_episode() -> None:
    wd = Watchdog()
    fired: List[str] = []
    wd.register("slow", 0.001, lambda n: fired.append(n))
    time.sleep(0.005)
    wd._check_once()
    wd._check_once()
    wd._check_once()
    # Three checks but only one fire — the watchdog must not spam recovery.
    assert fired == ["slow"]


def test_callback_re_arms_after_heartbeat() -> None:
    wd = Watchdog()
    fired: List[str] = []
    wd.register("flaky", 0.001, lambda n: fired.append(n))
    time.sleep(0.005)
    wd._check_once()
    assert fired == ["flaky"]
    # Recover: heartbeat resets fired flag.
    wd.heartbeat("flaky")
    time.sleep(0.005)
    wd._check_once()
    assert fired == ["flaky", "flaky"]


def test_multiple_workers_tracked_independently() -> None:
    wd = Watchdog()
    fired: List[str] = []
    wd.register("a", 0.001, lambda n: fired.append(n))
    wd.register("b", 1.0, lambda n: fired.append(n))  # generous budget
    time.sleep(0.005)
    wd._check_once()
    assert fired == ["a"], "Only the slow worker should have fired"


def test_unregister_stops_tracking() -> None:
    wd = Watchdog()
    fired: List[str] = []
    wd.register("ghost", 0.001, lambda n: fired.append(n))
    assert wd.unregister("ghost") is True
    assert wd.unregister("ghost") is False
    time.sleep(0.005)
    wd._check_once()
    assert fired == []


def test_callback_exception_does_not_crash_watchdog() -> None:
    wd = Watchdog()
    other_fired: List[str] = []

    def explode(_name: str) -> None:
        raise RuntimeError("recovery handler bug")

    wd.register("bad", 0.001, explode)
    wd.register("good", 0.001, lambda n: other_fired.append(n))
    time.sleep(0.005)
    wd._check_once()
    # 'good' must still have fired even though 'bad' raised.
    assert "good" in other_fired


def test_is_stale_and_age() -> None:
    wd = Watchdog()
    wd.register("w", 0.01, lambda _n: None)
    assert wd.is_stale("w") is False
    assert wd.last_heartbeat_age_s("w") is not None
    time.sleep(0.02)
    assert wd.is_stale("w") is True
    # Unknown worker returns None / False.
    assert wd.last_heartbeat_age_s("unknown") is None
    assert wd.is_stale("unknown") is False


def test_context_manager_lifecycle() -> None:
    fired: List[str] = []
    with Watchdog(poll_interval_s=0.01) as wd:
        wd.register("ctx", 0.005, lambda n: fired.append(n))
        # Poll until the callback lands rather than sleeping a fixed 50ms:
        # on loaded CI runners the poll thread can be starved past any
        # fixed margin. Generous deadline, exits immediately on success.
        deadline = time.perf_counter() + 2.0
        while "ctx" not in fired and time.perf_counter() < deadline:
            time.sleep(0.005)
    # After exit, the thread is joined. The callback should have fired at
    # least once during the with-block.
    assert "ctx" in fired
    # Stopping is idempotent.
    wd.stop()
