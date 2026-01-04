"""
Integrations module for DAiW-Music-Brain.

This module provides integration interfaces with external systems while
maintaining the core philosophy of "Interrogate Before Generate" -
emotional intent should drive technical decisions.

Provides:
- PentaCoreIntegration: HTTP-based integration with penta-core service
- LocalPentaCoreIntegration: Direct integration with local Python bindings
- DynamicsIntegration: Cross-module dynamics processing
- EmotionToDynamicsMapper: Emotion-to-dynamics parameter mapping
"""

from .penta_core import (
    PentaCoreConfig,
    PentaCoreIntegration,
    LocalPentaCoreIntegration,
)

from .dynamics_integration import (
    SectionType,
    EmotionState,
    DynamicsParameters,
    SectionContext,
    EmotionToDynamicsMapper,
    DynamicsIntegration,
    create_section_dynamics,
    get_dynamics_for_emotion,
)

__all__ = [
    # Penta-Core Integration
    "PentaCoreConfig",
    "PentaCoreIntegration",
    "LocalPentaCoreIntegration",
    # Dynamics Integration
    "SectionType",
    "EmotionState",
    "DynamicsParameters",
    "SectionContext",
    "EmotionToDynamicsMapper",
    "DynamicsIntegration",
    "create_section_dynamics",
    "get_dynamics_for_emotion",
]
