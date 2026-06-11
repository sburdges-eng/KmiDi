"""StemBus — multi-stem audio container with mix-down.

A small DAW-style routing primitive: per-named-stem audio buffers, per-stem
gain in linear units, mute/solo flags, and ``mix()`` that returns the
gain-scaled, mute-and-solo-respecting sum. Designed to sit between a
multi-instrument generator (which emits one buffer per role/patch/instrument)
and a downstream renderer that expects a single mixed buffer or a stem-set.

Iteration order preserves insertion (matches DAW track-list expectations).

Goal-list ties: Stem-aware generation, Multi-instrument orchestration.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class StemBus:
    """Multi-stem audio bus indexed by name.

    All stems share a fixed sample length; attempting to set a stem of a
    different length raises ``ValueError`` so callers can't mix unaligned
    buffers by accident.
    """

    def __init__(self, num_samples: int) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self._n = int(num_samples)
        self._stems: Dict[str, np.ndarray] = {}
        self._gain: Dict[str, float] = {}
        self._mute: Dict[str, bool] = {}
        self._solo: Dict[str, bool] = {}

    # -------------------------------------------------------------- basics
    @property
    def num_samples(self) -> int:
        return self._n

    @property
    def names(self) -> List[str]:
        """Stem names in insertion order."""
        return list(self._stems.keys())

    def set(self, name: str, buffer: np.ndarray) -> None:
        buf = np.asarray(buffer, dtype=np.float32)
        if buf.shape != (self._n,):
            raise ValueError(f"buffer length {buf.shape} != bus length ({self._n},)")
        if name not in self._stems:
            self._gain[name] = 1.0
            self._mute[name] = False
            self._solo[name] = False
        self._stems[name] = buf

    def get(self, name: str) -> np.ndarray:
        return self._stems[name]

    def remove(self, name: str) -> None:
        del self._stems[name]
        self._gain.pop(name, None)
        self._mute.pop(name, None)
        self._solo.pop(name, None)

    # ------------------------------------------------------- gain / mute / solo
    def set_gain(self, name: str, gain: float) -> None:
        if name not in self._stems:
            raise KeyError(name)
        self._gain[name] = float(gain)

    def get_gain(self, name: str) -> float:
        return self._gain[name]

    def mute(self, name: str, value: bool) -> None:
        if name not in self._stems:
            raise KeyError(name)
        self._mute[name] = bool(value)

    def solo(self, name: str, value: bool) -> None:
        if name not in self._stems:
            raise KeyError(name)
        self._solo[name] = bool(value)

    # -------------------------------------------------------------- mix
    def mix(self) -> np.ndarray:
        """Sum all audible stems with their gain applied.

        DAW convention: when any stem is soloed, only soloed stems are
        audible; mute is ignored on a soloed stem. With no solo active,
        mute simply excludes the stem.
        """
        any_solo = any(self._solo.values())
        out = np.zeros(self._n, dtype=np.float32)
        for name, buf in self._stems.items():
            if any_solo:
                if not self._solo[name]:
                    continue
                # solo overrides mute, so don't check mute here
            else:
                if self._mute[name]:
                    continue
            out += buf * self._gain[name]
        return out
