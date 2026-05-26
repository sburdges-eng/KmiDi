"""Tests for behavior_lab module (Phase 3)."""

import asyncio
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any, Callable


# Mock EventBus for testing without full music_brain import chain
class MockEventBus:
    def __init__(self):
        self._handlers: dict[str, list] = {}
        self._emitted: list[tuple[str, Any]] = []

    def on(self, event_type: str, handler: Callable = None, **kwargs):
        if handler is None:

            def decorator(fn):
                self._handlers.setdefault(event_type, []).append(fn)
                return fn

            return decorator
        self._handlers.setdefault(event_type, []).append(handler)

    async def emit(self, event_type: str, data: Any = None):
        self._emitted.append((event_type, data))

        @dataclass
        class FakeEvent:
            type: str
            data: Any

        event = FakeEvent(type=event_type, data=data)
        for handler in self._handlers.get(event_type, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)


class MockBridge:
    def __init__(self):
        self.pushed_intents: list[tuple[str, float]] = []
        self.pushed_emotions: list[dict] = []

    def push_intent(self, param_name: str, value: float) -> bool:
        self.pushed_intents.append((param_name, value))
        return True

    def push_emotion(
        self, valence: float, arousal: float, dominance: float, intensity: float = 1.0
    ) -> bool:
        self.pushed_emotions.append(
            {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
            }
        )
        return True


@pytest.fixture
def event_bus():
    return MockEventBus()


@pytest.fixture
def bridge():
    return MockBridge()


class TestClosedLoopController:
    def test_import(self):
        from music_brain.behavior_lab import ClosedLoopController

        assert ClosedLoopController is not None

    @pytest.mark.anyio
    async def test_rule_fires_on_condition(self, bridge, event_bus):
        from music_brain.behavior_lab.controller import ClosedLoopController, ParameterRule

        rule = ParameterRule(
            name="test-rule",
            condition=lambda s: s.get("arousal", 0.5) < 0.3,
            target_param="harmonic_tension",
            adjustment=0.05,
            cooldown_bars=0,
        )

        controller = ClosedLoopController(bridge=bridge, event_bus=event_bus, rules=[rule])
        await controller.start()

        # Simulate a state update with low arousal
        await controller._on_state_update(MagicMock(data={"arousal": 0.1, "bar": 1}))

        assert len(bridge.pushed_intents) == 1
        assert bridge.pushed_intents[0][0] == "harmonic_tension"

        await controller.stop()

    @pytest.mark.anyio
    async def test_cooldown_prevents_rapid_fire(self, bridge, event_bus):
        from music_brain.behavior_lab.controller import ClosedLoopController, ParameterRule

        rule = ParameterRule(
            name="cooldown-test",
            condition=lambda s: True,
            target_param="groove_strength",
            adjustment=0.1,
            cooldown_bars=4,
        )

        controller = ClosedLoopController(bridge=bridge, event_bus=event_bus, rules=[rule])
        await controller.start()

        # Fire at bar 0
        await controller._on_state_update(MagicMock(data={"bar": 0}))
        assert len(bridge.pushed_intents) == 1

        # Try again at bar 1 — should be blocked by cooldown
        await controller._on_state_update(MagicMock(data={"bar": 1}))
        assert len(bridge.pushed_intents) == 1  # still 1

        # Try at bar 4 — cooldown elapsed
        await controller._on_state_update(MagicMock(data={"bar": 4}))
        assert len(bridge.pushed_intents) == 2

        await controller.stop()


class TestScenarios:
    def test_import(self):
        from music_brain.behavior_lab import Scenario

        assert Scenario is not None

    def test_rule_to_parameter_rule(self):
        from music_brain.behavior_lab.scenarios import Rule

        rule = Rule(
            name="test",
            condition=lambda s: True,
            target_param="groove_strength",
            adjustment=0.1,
        )
        pr = rule.to_parameter_rule()
        assert pr.name == "test"
        assert pr.adjustment == 0.1

    def test_builtin_scenarios_exist(self):
        from music_brain.behavior_lab.scenarios import (
            emotional_buildup_scenario,
            call_and_response_scenario,
        )

        s1 = emotional_buildup_scenario()
        s2 = call_and_response_scenario()
        assert len(s1.rules) == 2
        assert len(s2.rules) == 2


class TestRTSnapshot:
    def test_import(self):
        from music_brain.integrations.comprehensive_bridge import RTSnapshot

        snap = RTSnapshot()
        assert snap.bpm == 120.0
        assert snap.bar == 0
        assert len(snap.track_params) == 16


class TestCKellyRTState:
    def test_struct_size(self):
        """Verify ctypes struct has a reasonable size (sanity check)."""
        from music_brain.integrations.comprehensive_bridge import CKellyRTState
        import ctypes

        # Should be > 100 bytes (lots of fields)
        assert ctypes.sizeof(CKellyRTState) > 100
