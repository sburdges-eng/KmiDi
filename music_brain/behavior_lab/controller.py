"""
ClosedLoopController — Reacts to engine events and adjusts parameters.

Subscribes to EventBus events emitted by ComprehensiveIntegrationManager
and applies parameter adjustments via the bridge's push methods.

Usage:
    controller = ClosedLoopController(bridge=manager, event_bus=bus)
    await controller.start()
    # ... engine emits events, controller reacts ...
    await controller.stop()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParameterRule:
    """
    A rule that maps an engine condition to a parameter adjustment.

    Attributes:
        name: Human-readable rule name.
        condition: Callable that receives an RTSnapshot dict and returns True to fire.
        target_param: Intent parameter name (e.g., "groove_strength") or "emotion".
        adjustment: Float delta to apply when condition fires.
        cooldown_bars: Minimum bars between firings (prevents oscillation).
        clamp: (min, max) range for the target parameter.
    """

    name: str
    condition: Callable[[dict], bool]
    target_param: str
    adjustment: float
    cooldown_bars: int = 2
    clamp: tuple[float, float] = (0.0, 1.0)


class ClosedLoopController:
    """
    Subscribes to engine events and adjusts parameters through the bridge.

    The controller maintains a set of ParameterRules. On each engine event,
    it evaluates rules and pushes parameter changes if conditions are met.
    """

    def __init__(
        self,
        bridge: Any,  # ComprehensiveIntegrationManager
        event_bus: Any,  # EventBus
        rules: Optional[List[ParameterRule]] = None,
    ):
        self._bridge = bridge
        self._bus = event_bus
        self._rules: List[ParameterRule] = rules or []
        self._running = False
        self._handler_ids: List[Any] = []
        self._last_fired_bar: Dict[str, int] = {}  # rule_name -> last bar fired
        self._current_values: Dict[str, float] = {}  # param_name -> current value

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: ParameterRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> None:
        self._rules = [r for r in self._rules if r.name != name]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Subscribe to engine events
        self._bus.on("engine.new_bar", handler=self._on_new_bar)
        self._bus.on("engine.emotion_change", handler=self._on_emotion_change)
        self._bus.on("engine.state_update", handler=self._on_state_update)

        logger.info("ClosedLoopController started with %d rules", len(self._rules))

    async def stop(self) -> None:
        self._running = False
        logger.info("ClosedLoopController stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_state_update(self, event: Any) -> None:
        """Process state updates — evaluate rules on every tick."""
        if not self._running:
            return
        data = event.data if hasattr(event, "data") else event
        await self._evaluate_rules(data)

    async def _on_new_bar(self, event: Any) -> None:
        """Log bar transitions for cooldown tracking."""
        data = event.data if hasattr(event, "data") else event
        logger.debug("New bar: %s", data.get("bar", "?"))

    async def _on_emotion_change(self, event: Any) -> None:
        """React to significant emotion changes."""
        data = event.data if hasattr(event, "data") else event
        logger.debug(
            "Emotion change: V=%.2f A=%.2f D=%.2f",
            data.get("valence", 0),
            data.get("arousal", 0),
            data.get("dominance", 0),
        )

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    async def _evaluate_rules(self, state: dict) -> None:
        current_bar = state.get("bar", 0)

        for rule in self._rules:
            # Cooldown check
            last_bar = self._last_fired_bar.get(rule.name, -999)
            if current_bar - last_bar < rule.cooldown_bars:
                continue

            # Condition check
            try:
                if not rule.condition(state):
                    continue
            except Exception:
                logger.exception("Rule '%s' condition raised", rule.name)
                continue

            # Apply adjustment
            current = self._current_values.get(rule.target_param, 0.5)
            new_val = max(rule.clamp[0], min(rule.clamp[1], current + rule.adjustment))

            if rule.target_param == "emotion":
                # Special case: push full emotion update
                self._bridge.push_emotion(
                    valence=state.get("valence", 0.0) + rule.adjustment,
                    arousal=state.get("arousal", 0.5),
                    dominance=state.get("dominance", 0.5),
                )
            else:
                self._bridge.push_intent(rule.target_param, new_val)

            self._current_values[rule.target_param] = new_val
            self._last_fired_bar[rule.name] = current_bar

            logger.info(
                "Rule '%s' fired: %s %.2f -> %.2f (bar %d)",
                rule.name,
                rule.target_param,
                current,
                new_val,
                current_bar,
            )
