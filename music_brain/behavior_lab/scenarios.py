"""
Scenario runner — declarative behavior definitions for behavior_lab.

A Scenario is a collection of Rules that define how the controller should
react to engine state. Scenarios can be loaded from dicts/YAML and applied
to a ClosedLoopController.

Usage:
    scenario = Scenario(
        name="emotional-buildup",
        rules=[
            Rule(
                name="increase-tension-on-low-arousal",
                event="engine.state_update",
                condition=lambda s: s.get("arousal", 0) < 0.3,
                target_param="harmonic_tension",
                adjustment=0.05,
                cooldown_bars=4,
            ),
        ],
    )
    await run_scenario(scenario, controller, event_bus)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .controller import ClosedLoopController, ParameterRule

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """
    Declarative rule for a behavior scenario.

    Attributes:
        name: Unique rule identifier.
        event: EventBus event type that triggers evaluation (e.g., "engine.state_update").
        condition: Callable(state_dict) -> bool. Fires when True.
        target_param: Parameter to adjust (intent name or "emotion").
        adjustment: Float delta per firing.
        cooldown_bars: Minimum bars between firings.
        clamp: (min, max) value range.
    """

    name: str
    event: str = "engine.state_update"
    condition: Callable[[dict], bool] = lambda _: False
    target_param: str = "groove_strength"
    adjustment: float = 0.0
    cooldown_bars: int = 2
    clamp: tuple[float, float] = (0.0, 1.0)

    def to_parameter_rule(self) -> ParameterRule:
        return ParameterRule(
            name=self.name,
            condition=self.condition,
            target_param=self.target_param,
            adjustment=self.adjustment,
            cooldown_bars=self.cooldown_bars,
            clamp=self.clamp,
        )


@dataclass
class Scenario:
    """
    A named collection of behavior rules.

    Attributes:
        name: Scenario identifier.
        description: What this scenario achieves.
        rules: List of Rules to apply.
        duration_bars: Optional max duration (None = run indefinitely).
    """

    name: str
    description: str = ""
    rules: List[Rule] = field(default_factory=list)
    duration_bars: Optional[int] = None


async def run_scenario(
    scenario: Scenario,
    controller: ClosedLoopController,
    event_bus: Any,
    timeout_seconds: Optional[float] = None,
) -> None:
    """
    Apply a scenario's rules to the controller and run until completion.

    Args:
        scenario: The behavior scenario to execute.
        controller: ClosedLoopController instance (must already have a bridge).
        event_bus: EventBus for monitoring bar progression.
        timeout_seconds: Max wall-clock time before stopping. None = no limit.
    """
    logger.info("Starting scenario '%s' (%d rules)", scenario.name, len(scenario.rules))

    # Install rules
    for rule in scenario.rules:
        controller.add_rule(rule.to_parameter_rule())

    await controller.start()

    try:
        if scenario.duration_bars is not None:
            # Wait for N bars to pass
            start_bar: Optional[int] = None

            bar_done = asyncio.Event()

            async def _bar_watcher(event: Any) -> None:
                nonlocal start_bar
                data = event.data if hasattr(event, "data") else event
                bar = data.get("bar", 0)
                if start_bar is None:
                    start_bar = bar
                if bar - start_bar >= scenario.duration_bars:
                    bar_done.set()

            event_bus.on("engine.new_bar", handler=_bar_watcher)

            if timeout_seconds:
                await asyncio.wait_for(bar_done.wait(), timeout=timeout_seconds)
            else:
                await bar_done.wait()
        elif timeout_seconds:
            await asyncio.sleep(timeout_seconds)
        else:
            # Run indefinitely — caller is responsible for stopping
            while True:
                await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        logger.info("Scenario '%s' cancelled", scenario.name)
    except asyncio.TimeoutError:
        logger.info("Scenario '%s' timed out after %.1fs", scenario.name, timeout_seconds)
    finally:
        await controller.stop()
        # Remove scenario rules
        for rule in scenario.rules:
            controller.remove_rule(rule.name)
        logger.info("Scenario '%s' complete", scenario.name)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------


def emotional_buildup_scenario() -> Scenario:
    """Gradually increase tension and density when arousal is low."""
    return Scenario(
        name="emotional-buildup",
        description="Builds tension when energy drops — increases harmonic tension and rhythmic "
        "density",
        rules=[
            Rule(
                name="tension-on-low-arousal",
                condition=lambda s: s.get("arousal", 0.5) < 0.3,
                target_param="harmonic_tension",
                adjustment=0.03,
                cooldown_bars=4,
                clamp=(0.0, 0.9),
            ),
            Rule(
                name="density-on-low-activity",
                condition=lambda s: s.get("melodic_activity", 0.5) < 0.2,
                target_param="rhythmic_density",
                adjustment=0.02,
                cooldown_bars=4,
                clamp=(0.1, 0.8),
            ),
        ],
    )


def call_and_response_scenario() -> Scenario:
    """Alternate between high and low groove strength every 4 bars."""
    bar_counter = {"n": 0}

    def _alternate(state: dict) -> bool:
        bar_counter["n"] += 1
        return bar_counter["n"] % 8 < 4  # first 4 bars: increase, next 4: condition fails

    return Scenario(
        name="call-and-response",
        description="Alternating groove intensity for call-and-response feel",
        rules=[
            Rule(
                name="groove-up",
                condition=_alternate,
                target_param="groove_strength",
                adjustment=0.1,
                cooldown_bars=4,
                clamp=(0.2, 0.9),
            ),
            Rule(
                name="groove-down",
                condition=lambda s: not _alternate(s),
                target_param="groove_strength",
                adjustment=-0.1,
                cooldown_bars=4,
                clamp=(0.2, 0.9),
            ),
        ],
    )
