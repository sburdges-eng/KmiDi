"""
Cascade Dispatcher — Universal fallback chain for KMiDi.

Routes Thought requests through the tier hierarchy based on ML Intensity:
    cloud_t3 -> cloud_t2 -> cloud_t1 -> local_engine -> penta_core -> embedded_ml -> raw_dsp

Each tier is a fallback for the one above.  If a tier times out or errors,
execution automatically falls through to the next lower tier.  The user
controls maximum tier depth via the ML Intensity slider (0-100%).
"""

from .dispatcher import CascadeDispatcher, CascadeResult
from .config import CascadeConfig, TierName
from .tiers import (
    CloudT1Handler,
    CloudT2Handler,
    CloudT3Handler,
    CloudT4Handler,
    LocalEngineHandler,
    EmbeddedMLHandler,
    RawDSPHandler,
)
from .engine_bridge import EngineBridge

__all__ = [
    "CascadeDispatcher",
    "CascadeResult",
    "CascadeConfig",
    "TierName",
    "CloudT1Handler",
    "CloudT2Handler",
    "CloudT3Handler",
    "CloudT4Handler",
    "LocalEngineHandler",
    "EmbeddedMLHandler",
    "RawDSPHandler",
    "EngineBridge",
]
