"""
ComprehensiveIntegrationManager — Facade wiring AIOrchestrator, DAW bridges,
behavior lab, and the voice pipeline into a single entry point.

This module intentionally uses lazy imports so it remains importable even when
optional dependencies (behavior_lab, DAW bridges, realtime engine) are absent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["ComprehensiveIntegrationManager"]

# ---------------------------------------------------------------------------
# Name → bridge class lookup (populated lazily on first use of send_to_daw)
# ---------------------------------------------------------------------------

_DAW_BRIDGE_MAP: Dict[str, str] = {
    "ableton": "AbletonDAWBridge",
    "bitwig": "BitwigBridge",
    "logic": "LogicProBridge",
    "logicpro": "LogicProBridge",
    "logic_pro": "LogicProBridge",
    "reaper": "ReaperBridge",
}


class ComprehensiveIntegrationManager:
    """
    Facade wiring AIOrchestrator, DAW bridges, behavior lab, and voice pipeline.

    All external dependencies are imported lazily so the manager can be
    instantiated in environments where only a subset of components is installed.
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        """
        Initialise the manager.

        Parameters
        ----------
        orchestrator:
            An ``AIOrchestrator`` instance to delegate pipeline work to.
            If *None* a default instance is created lazily on first use.
        config:
            Optional configuration forwarded to ``AIOrchestrator`` when one
            is created internally.
        """
        self._config = config
        self._orchestrator = orchestrator

        # Cache of already-instantiated DAW bridges keyed by normalised name.
        self._daw_bridges: Dict[str, Any] = {}

        # Realtime engine handle — resolved once and cached.
        self._rt_engine: Optional[Any] = None
        self._rt_checked: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_orchestrator(self) -> Any:
        """Return the orchestrator, creating a default one if necessary."""
        if self._orchestrator is None:
            try:
                from music_brain.orchestrator.orchestrator import AIOrchestrator, OrchestratorConfig

                cfg = self._config if isinstance(self._config, OrchestratorConfig) else None
                self._orchestrator = AIOrchestrator(cfg)
            except ImportError:
                self._orchestrator = None
        return self._orchestrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_behavior_scenario(self, scenario_spec: dict) -> dict:
        """
        Delegate *scenario_spec* to ``behavior_lab.runner.run_scenario``.

        Returns a plain dict result on success, or
        ``{"error": "behavior_lab not available"}`` if the package cannot be
        imported.
        """
        try:
            from behavior_lab.runner import run_scenario  # type: ignore[import]
        except ImportError:
            return {"error": "behavior_lab not available"}

        try:
            result = run_scenario(scenario_spec)
            # Normalise to dict; some runners return custom objects.
            if isinstance(result, dict):
                return result
            if hasattr(result, "__dict__"):
                return vars(result)
            return {"result": result}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def send_to_daw(self, daw_name: str, midi_data: Any) -> bool:
        """
        Route *midi_data* to the named DAW bridge.

        Supported *daw_name* values (case-insensitive): ``ableton``,
        ``bitwig``, ``logic`` / ``logicpro`` / ``logic_pro``, ``reaper``.

        Returns ``True`` if the data was dispatched successfully, ``False``
        otherwise (unknown DAW, import error, connection failure, etc.).
        """
        key = daw_name.lower().replace(" ", "")
        class_name = _DAW_BRIDGE_MAP.get(key)
        if class_name is None:
            return False

        # Retrieve or create the bridge instance.
        bridge = self._daw_bridges.get(key)
        if bridge is None:
            try:
                import music_brain.agents.daw_bridges as _bridges

                bridge_cls = getattr(_bridges, class_name, None)
                if bridge_cls is None:
                    return False
                bridge = bridge_cls()
                self._daw_bridges[key] = bridge
            except ImportError:
                return False

        # Ensure the bridge is connected.
        try:
            if not bridge.is_connected:
                connected = bridge.connect()
                if not connected:
                    return False
        except Exception:  # noqa: BLE001
            return False

        # Dispatch MIDI data.  The bridge protocol exposes send_note_on for
        # individual notes; when midi_data is a dict we delegate accordingly.
        try:
            if isinstance(midi_data, dict):
                note = midi_data.get("note", 60)
                velocity = midi_data.get("velocity", 100)
                channel = midi_data.get("channel", 0)
                bridge.send_note_on(note, velocity, channel)
            elif hasattr(midi_data, "__iter__"):
                # List/tuple of MIDI event dicts or raw note numbers.
                for item in midi_data:
                    if isinstance(item, dict):
                        bridge.send_note_on(
                            item.get("note", 60),
                            item.get("velocity", 100),
                            item.get("channel", 0),
                        )
                    else:
                        bridge.send_note_on(int(item))
            else:
                bridge.send_note_on(int(midi_data))
            return True
        except Exception:  # noqa: BLE001
            return False

    def realtime_available(self) -> bool:
        """
        Return *True* if the real-time engine is reachable.

        Checks ``music_brain.realtime`` and falls back to probing any
        initialised RT engine on the orchestrator.  Returns a plain bool.
        """
        if not self._rt_checked:
            self._rt_engine = self._probe_realtime()
            self._rt_checked = True

        return self._rt_engine is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _probe_realtime(self) -> Optional[Any]:
        """Attempt to locate a live real-time engine handle. Returns it or None."""
        # Attempt 1 — dedicated realtime sub-package.
        try:
            import music_brain.realtime as _rt  # type: ignore[import]

            engine = getattr(_rt, "get_engine", None)
            if callable(engine):
                return engine()
            # Module present but no explicit accessor — treat presence as enough.
            return _rt
        except ImportError:
            pass

        # Attempt 2 — check if the orchestrator itself exposes an RT engine.
        orch = self._get_orchestrator()
        if orch is not None:
            rt_attr = getattr(orch, "rt_engine", None) or getattr(orch, "realtime_engine", None)
            if rt_attr is not None:
                return rt_attr

        return None
