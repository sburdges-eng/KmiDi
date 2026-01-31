"""
Brain ML — Coexist layer for JEPA vs current models.

CapabilityRouter chooses current|shadow|jepa per capability.
ModelRegistry holds current and JEPA backends.
ShadowLogger records side-by-side metrics when mode=shadow.

Env: KMI_DI_MODEL_MODE=current|shadow|jepa
"""

from .router import CapabilityRouter, Mode, Result
from .registry import CoexistRegistry
from .shadow_logger import ShadowLogger
from .setup import create_coexist_router

__all__ = [
    "CapabilityRouter",
    "Mode",
    "Result",
    "CoexistRegistry",
    "ShadowLogger",
    "create_coexist_router",
]
