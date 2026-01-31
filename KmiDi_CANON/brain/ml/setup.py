"""
Coexist router setup — Build registry and router from env.

Reads KMI_DI_MODEL_MODE (current|shadow|jepa), KMI_DI_SHADOW_LOG_DIR.
Registers placeholder backends for midi_understanding, audio_understanding, specto_understanding.
Replace placeholders with real models as JEPA training progresses.
"""

import os
from typing import Optional

from .registry import CoexistRegistry
from .router import CapabilityRouter, Mode, Result
from .shadow_logger import ShadowLogger


def _stub_current_infer(inputs: dict) -> Result:
    """Placeholder current backend: returns minimal payload. Replace with real encoder."""
    return Result(payload={"source": "current", "status": "completed"}, status="completed")


def _stub_jepa_infer(inputs: dict) -> Result:
    """Placeholder JEPA backend: not yet trained. Replace when JEPA models are available."""
    return Result(payload={"source": "jepa", "status": "stub"}, status="stub")


def _build_registry() -> CoexistRegistry:
    """Build registry with current and JEPA backends per capability."""
    registry = CoexistRegistry()

    # Placeholder adapters — same interface: .infer(inputs) -> Result
    class StubCurrent:
        def infer(self, x): return _stub_current_infer(x)

    class StubJepa:
        def infer(self, x): return _stub_jepa_infer(x)

    for cap in ("midi_understanding", "audio_understanding", "specto_understanding"):
        registry.register_current(cap, StubCurrent())
        registry.register_jepa(cap, StubJepa())

    return registry


def create_coexist_router() -> Optional[CapabilityRouter]:
    """
    Create CapabilityRouter from env. Returns None if ml module unavailable or mode not set.

    Env:
        KMI_DI_MODEL_MODE: current | shadow | jepa (default: current)
        KMI_DI_SHADOW_LOG_DIR: directory for shadow JSONL logs
    """
    mode_str = os.environ.get("KMI_DI_MODEL_MODE", "current").lower().strip()
    if mode_str not in ("current", "shadow", "jepa"):
        mode_str = "current"

    mode = Mode(mode_str)
    registry = _build_registry()

    shadow_logger = None
    if mode == Mode.SHADOW:
        log_dir = os.environ.get("KMI_DI_SHADOW_LOG_DIR", "logs/shadow")
        shadow_logger = ShadowLogger(base_dir=log_dir)

    return CapabilityRouter(registry=registry, mode=mode, shadow_logger=shadow_logger)
