"""
CoexistRegistry — Maps capability names to current and JEPA backends.

Holds {capability: {current: model, jepa: model}} for routing.
Each model must expose infer(inputs: Dict) -> Result.
"""

from typing import Any, Dict, Optional


class CoexistRegistry:
    """
    Registry of current and JEPA models per capability.

    current: dict mapping capability name -> model with .infer(inputs) -> Result
    jepa:    dict mapping capability name -> model with .infer(inputs) -> Result
    """

    def __init__(
        self,
        current: Optional[Dict[str, Any]] = None,
        jepa: Optional[Dict[str, Any]] = None,
    ):
        self.current: Dict[str, Any] = current or {}
        self.jepa: Dict[str, Any] = jepa or {}

    def register_current(self, capability: str, model: Any) -> None:
        """Register a current (legacy) model for a capability."""
        self.current[capability] = model

    def register_jepa(self, capability: str, model: Any) -> None:
        """Register a JEPA model for a capability."""
        self.jepa[capability] = model

    def has_capability(self, capability: str) -> bool:
        """Check if capability is registered for current mode."""
        return capability in self.current

    def has_jepa(self, capability: str) -> bool:
        """Check if capability has a JEPA backend."""
        return capability in self.jepa
