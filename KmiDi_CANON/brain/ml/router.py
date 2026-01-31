"""
CapabilityRouter — Routes inference to current or JEPA based on mode.

Modes: current (default), shadow (parallel + log), jepa (JEPA output).
Contract-safe: response shape unchanged; JEPA diagnostics under debug only when shadow.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional


class Mode(Enum):
    CURRENT = "current"
    SHADOW = "shadow"
    JEPA = "jepa"


@dataclass
class Result:
    """Inference result. debug is optional; used for shadow diagnostics."""
    payload: Dict[str, Any]
    status: str
    debug: Optional[Dict[str, Any]] = None


class CapabilityRouter:
    """
    Routes capability requests to current or JEPA backend.

    In SHADOW mode: runs both, logs comparison, returns current output
    with optional debug.shadow_status.
    """

    def __init__(self, registry: "CoexistRegistry", mode: Mode, shadow_logger: Optional["ShadowLogger"] = None):
        self.registry = registry
        self.mode = mode
        self.shadow_logger = shadow_logger

    def run(self, capability: str, inputs: Dict[str, Any]) -> Result:
        if self.mode == Mode.CURRENT:
            if capability not in self.registry.current:
                return Result(
                    payload={},
                    status="error",
                    debug={"details": f"capability '{capability}' not registered"},
                )
            return self.registry.current[capability].infer(inputs)

        if self.mode == Mode.JEPA:
            if capability not in self.registry.jepa:
                return Result(
                    payload={},
                    status="error",
                    debug={"details": f"jepa backend for '{capability}' not registered"},
                )
            return self.registry.jepa[capability].infer(inputs)

        # SHADOW: run current, run JEPA in parallel, log, return current
        if capability not in self.registry.current:
            return Result(payload={}, status="error", debug={"details": f"capability '{capability}' not registered"})

        primary = self.registry.current[capability].infer(inputs)
        shadow = (
            self.registry.jepa[capability].infer(inputs)
            if capability in self.registry.jepa
            else Result(payload={}, status="stub", debug={"details": "jepa not registered"})
        )

        if self.shadow_logger is not None:
            self.shadow_logger.log(capability, inputs, primary, shadow)

        # Attach shadow status to debug only (contract-safe)
        primary.debug = {**(primary.debug or {}), "shadow_status": shadow.status}
        return primary
