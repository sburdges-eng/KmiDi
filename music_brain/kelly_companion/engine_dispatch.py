"""
Engine dispatch table for Kelly Companion engines.

Maps Intent IR parameter names to engine calls, enabling the orchestration
layer to route ML-influenced generation requests to the correct engine.

Sprint 8 — infrastructure wiring for per-engine generation.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
# Each entry maps a logical engine name to:
#   engine       – class name inside the module
#   module       – dotted module path (lazy-imported)
#   ir_params    – Intent IR parameters consumed by this engine
#   ml_influence – key into the ML influences dict that gates activation
# ---------------------------------------------------------------------------

ENGINE_DISPATCH: Dict[str, Dict[str, Any]] = {
    "rhythm": {
        "engine": "RhythmEngine",
        "module": "kelly_companion.kellymidicompanion_rhythm_engine",
        "ir_params": ["rhythmic_density", "groove_strength"],
        "ml_influence": "groove_influence",
    },
    "dynamics": {
        "engine": "DynamicsEngine",
        "module": "kelly_companion.kellymidicompanion_dynamics_engine",
        "ir_params": ["dynamic_range"],
        "ml_influence": "dynamics_influence",
    },
    "counter_melody": {
        "engine": "CounterMelodyEngine",
        "module": "kelly_companion.kellymidicompanion_counter_melody_engine",
        "ir_params": ["melodic_activity"],
        "ml_influence": "melody_influence",
    },
    "string": {
        "engine": "StringEngine",
        "module": "kelly_companion.kellymidicompanion_string_engine",
        "ir_params": ["texture_density"],
        "ml_influence": "ml_intensity",
    },
    "fill": {
        "engine": "FillEngine",
        "module": "kelly_companion.kellymidicompanion_fill_engine",
        "ir_params": ["transition_events"],
        "ml_influence": "groove_influence",
    },
    "groove": {
        "engine": "GrooveEngine",
        "module": "kelly_companion.kellymidicompanion_groove.kellymidicompanion_groove_engine",
        "ir_params": ["groove_strength"],
        "ml_influence": "groove_influence",
    },
    "tension": {
        "engine": "TensionEngine",
        "module": "kelly_companion.kellymidicompanion_tension_engine",
        "ir_params": ["harmonic_tension"],
        "ml_influence": "harmony_influence",
    },
    "transition": {
        "engine": "TransitionEngine",
        "module": "kelly_companion.kellymidicompanion_transition_engine",
        "ir_params": ["section_events"],
        "ml_influence": "ml_intensity",
    },
    "variation": {
        "engine": "VariationEngine",
        "module": "kelly_companion.kellymidicompanion_variation_engine",
        "ir_params": ["contour_variance"],
        "ml_influence": "melody_influence",
    },
    "bass": {
        "engine": "BassEngine",
        "module": "kelly_companion.kellymidicompanion_bass_engine",
        "ir_params": ["mode_preference"],
        "ml_influence": "melody_influence",
    },
    "melody": {
        "engine": "MelodyEngine",
        "module": "kelly_companion.kellymidicompanion_melody_engine",
        "ir_params": ["melodic_activity", "contour_variance"],
        "ml_influence": "melody_influence",
    },
    "pad": {
        "engine": "PadEngine",
        "module": "kelly_companion.kellymidicompanion_pad_engine",
        "ir_params": ["texture_density"],
        "ml_influence": "ml_intensity",
    },
}

# ---------------------------------------------------------------------------
# Singleton cache for instantiated engines
# ---------------------------------------------------------------------------
_engine_cache: Dict[str, Any] = {}

# Default activation threshold — engines whose ml_influence value falls
# below this are skipped during dispatch.
ACTIVATION_THRESHOLD = 0.1


def get_engine(name: str) -> Any:
    """Lazily import and instantiate the engine identified by *name*.

    Returns the cached instance on subsequent calls.

    Raises ``KeyError`` if *name* is not in :data:`ENGINE_DISPATCH`.
    """
    if name not in ENGINE_DISPATCH:
        raise KeyError(f"Unknown engine: {name!r}")

    if name in _engine_cache:
        return _engine_cache[name]

    entry = ENGINE_DISPATCH[name]
    module_path = entry["module"]
    class_name = entry["engine"]

    logger.debug("Lazy-loading %s from %s", class_name, module_path)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()
    _engine_cache[name] = instance
    return instance


def dispatch_intent(
    intent_ir_dict: Dict[str, Any],
    ml_influences_dict: Dict[str, float],
    *,
    threshold: float = ACTIVATION_THRESHOLD,
) -> Dict[str, Any]:
    """Route an Intent IR through all engines whose influence exceeds *threshold*.

    Parameters
    ----------
    intent_ir_dict:
        Flat dictionary of Intent IR parameters (e.g. ``{"rhythmic_density": 0.7, ...}``).
    ml_influences_dict:
        Dictionary mapping influence names to float strengths
        (e.g. ``{"groove_influence": 0.8, "melody_influence": 0.3, ...}``).
    threshold:
        Minimum influence value required to activate an engine.

    Returns
    -------
    dict
        Mapping of ``engine_name -> output`` for every engine that was activated.
    """
    results: Dict[str, Any] = {}

    for engine_name, entry in ENGINE_DISPATCH.items():
        influence_key = entry["ml_influence"]
        influence_value = ml_influences_dict.get(influence_key, 0.0)

        if influence_value <= threshold:
            logger.debug(
                "Skipping %s (influence %s=%.3f <= %.3f)",
                engine_name,
                influence_key,
                influence_value,
                threshold,
            )
            continue

        # Collect relevant IR params for this engine
        engine_params: Dict[str, Any] = {}
        for param in entry["ir_params"]:
            if param in intent_ir_dict:
                engine_params[param] = intent_ir_dict[param]

        try:
            engine = get_engine(engine_name)
        except ImportError:
            logger.warning("Engine %s not available (missing module)", engine_name)
            results[engine_name] = None
            continue

        try:
            output = engine.generate(**engine_params)
            results[engine_name] = output
            logger.info("Engine %s produced output", engine_name)
        except Exception:
            logger.exception("Engine %s failed during generation", engine_name)
            results[engine_name] = None

    return results
