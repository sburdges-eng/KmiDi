"""Heartbeat-driven watchdog for long-running workers.

Closes the **Watchdog recovery** spec item documented as a gap in
``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``.

A worker registers itself with a name and a stale-timeout. It then calls
``heartbeat(name)`` from its loop. If the watchdog observes that
``now - last_heartbeat > stale_timeout``, it invokes the registered
recovery callback (typically: stop the worker, log telemetry, restart).

Design constraints:
- Stdlib only (``time`` + ``threading`` + ``dataclasses``). No background
  thread is started until ``start()`` is called.
- Ages are measured with ``time.perf_counter()``, not ``time.monotonic()``:
  on Windows for CPython <= 3.12 ``monotonic()`` is ``GetTickCount64()``
  with ~15.6 ms granularity, which cannot resolve sub-tick stale budgets.
- Not an RT-thread primitive. Use this for ML inference workers,
  background generators, MIDI-CI daemon, etc.; never call from
  ``processBlock``.
- Multi-worker safe: each registered worker has independent state.
- Callbacks run on the watchdog thread; they should be fast or
  hand off to another executor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class _WorkerState:
    name: str
    stale_timeout_s: float
    on_stale: Callable[[str], None]
    last_heartbeat: float = field(default_factory=time.perf_counter)
    fired: bool = False


class Watchdog:
    """Background heartbeat monitor with per-worker stale-timeout.

    Args:
        poll_interval_s: How often the monitor thread checks for stale
            workers. Default 1.0 seconds — fine for ML/orchestration
            workers; not appropriate for sub-millisecond RT paths.
    """

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self._poll = float(poll_interval_s)
        self._workers: dict[str, _WorkerState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    def register(self, name: str, stale_timeout_s: float, on_stale: Callable[[str], None]) -> None:
        """Track ``name`` with a ``stale_timeout_s`` budget.

        ``on_stale`` is invoked from the watchdog thread when
        ``now - last_heartbeat > stale_timeout_s``. It fires at most once
        per stale episode; re-arms automatically on the next heartbeat.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be positive")
        with self._lock:
            self._workers[name] = _WorkerState(
                name=name,
                stale_timeout_s=float(stale_timeout_s),
                on_stale=on_stale,
            )

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._workers.pop(name, None) is not None

    def heartbeat(self, name: str) -> None:
        """Called from inside the worker loop on every iteration / event."""
        with self._lock:
            state = self._workers.get(name)
            if state is None:
                return
            state.last_heartbeat = time.perf_counter()
            state.fired = False  # re-arm after recovery

    def is_stale(self, name: str) -> bool:
        with self._lock:
            state = self._workers.get(name)
            if state is None:
                return False
            return (time.perf_counter() - state.last_heartbeat) > state.stale_timeout_s

    def last_heartbeat_age_s(self, name: str) -> Optional[float]:
        with self._lock:
            state = self._workers.get(name)
            if state is None:
                return None
            return time.perf_counter() - state.last_heartbeat

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="kmidi-watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> "Watchdog":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    def _check_once(self) -> None:
        """Single check pass — exposed for deterministic testing."""
        now = time.perf_counter()
        to_fire: list[tuple[Callable[[str], None], str]] = []
        with self._lock:
            for state in self._workers.values():
                age = now - state.last_heartbeat
                if age > state.stale_timeout_s and not state.fired:
                    state.fired = True
                    to_fire.append((state.on_stale, state.name))
        # Invoke callbacks outside the lock to avoid deadlocks if the
        # callback re-enters the watchdog (e.g. unregister/restart).
        for cb, name in to_fire:
            try:
                cb(name)
            except Exception:  # noqa: BLE001 — recovery must not crash the watchdog
                pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._check_once()
            self._stop_event.wait(self._poll)


__all__ = ["Watchdog"]
