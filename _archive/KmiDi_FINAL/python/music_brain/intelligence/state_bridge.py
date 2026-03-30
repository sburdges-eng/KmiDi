"""
State Bridge - Python interface for C++ state synchronization.

Provides functions to receive state updates from C++ engines and provide
current state queries.
"""

from typing import Dict, List, Any, Optional
import json
from collections import defaultdict
from datetime import datetime

# Global state storage
_engine_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
_current_state: Dict[str, Any] = {}


def emit_state_update(engine_type: str, state_json: str):
    """
    Receive state update from C++ engine.

    This function is designed to be called from C++ via Python bridge.

    Args:
        engine_type: Engine type ("melody", "bass", "drum", "midi_generator", etc.)
        state_json: JSON string with state update:
            {
                "chords": ["Am", "Dm", "F", "C"],
                "notes": [...],
                "parameters": {"complexity": 0.4, ...},
                "timestamp": 1234567890
            }
    """
    try:
        state = json.loads(state_json)
        state["timestamp"] = datetime.now().isoformat()
        state["engine_type"] = engine_type

        # Store engine-specific state
        _engine_states[engine_type] = state

        # Update global current state
        _update_current_state(engine_type, state)

    except Exception as e:
        # Silently fail - state updates are not critical
        pass


def get_current_state() -> str:
    """
    Get current aggregated state from all engines.

    Returns:
        JSON string with current state:
        {
            "emotion": "grief",
            "chords": [...],
            "parameters": {...},
            "context": {...},
            "engines": {
                "melody": {...},
                "bass": {...},
                ...
            }
        }
    """
    try:
        # Aggregate state from all engines
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "engines": dict(_engine_states),
        }

        # Add global state
        aggregated.update(_current_state)

        return json.dumps(aggregated)

    except Exception as e:
        return json.dumps({})


def get_engine_state(engine_type: str) -> str:
    """
    Get state for specific engine.

    Args:
        engine_type: Engine type

    Returns:
        JSON string with engine state
    """
    try:
        state = _engine_states.get(engine_type, {})
        return json.dumps(state)

    except Exception as e:
        return json.dumps({})


def _update_current_state(engine_type: str, state: Dict[str, Any]):
    """
    Update global current state with engine state.

    Args:
        engine_type: Engine type
        state: Engine state dictionary
    """
    # Aggregate chords from all engines
    if "chords" in state:
        if "chords" not in _current_state:
            _current_state["chords"] = []
        # Merge chords (avoid duplicates)
        for chord in state["chords"]:
            if chord not in _current_state["chords"]:
                _current_state["chords"].append(chord)

    # Aggregate parameters
    if "parameters" in state:
        if "parameters" not in _current_state:
            _current_state["parameters"] = {}
        _current_state["parameters"].update(state["parameters"])

    # Update emotion if present
    if "emotion" in state:
        _current_state["emotion"] = state["emotion"]

    # Update key/mode if present
    if "key" in state:
        _current_state["key"] = state["key"]
    if "mode" in state:
        _current_state["mode"] = state["mode"]
