"""
Unit tests for ComprehensiveIntegrationManager.

All tests are designed to pass without any external dependencies
(behavior_lab, mido, python-osc, etc.) being installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(**kwargs):
    """Import and instantiate the manager fresh for each test."""
    from music_brain.orchestrator.comprehensive_manager import ComprehensiveIntegrationManager

    return ComprehensiveIntegrationManager(**kwargs)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_no_args(self):
        """Manager can be created with zero arguments."""
        mgr = _make_manager()
        assert mgr is not None

    def test_with_none_orchestrator(self):
        mgr = _make_manager(orchestrator=None, config=None)
        assert mgr._orchestrator is None

    def test_with_explicit_orchestrator(self):
        mock_orch = MagicMock()
        mgr = _make_manager(orchestrator=mock_orch)
        assert mgr._get_orchestrator() is mock_orch


# ---------------------------------------------------------------------------
# realtime_available
# ---------------------------------------------------------------------------


class TestRealtimeAvailable:
    def test_returns_bool(self):
        """realtime_available() must always return a plain bool."""
        mgr = _make_manager()
        result = mgr.realtime_available()
        assert isinstance(result, bool)

    def test_returns_false_when_module_missing(self):
        """Returns False when music_brain.realtime is not importable."""
        mgr = _make_manager()
        # Ensure the module is not cached in sys.modules.
        sys.modules.pop("music_brain.realtime", None)
        with patch.dict(sys.modules, {"music_brain.realtime": None}):
            mgr._rt_checked = False
            mgr._rt_engine = None
            result = mgr.realtime_available()
        assert result is False

    def test_returns_true_when_module_present(self):
        """Returns True when music_brain.realtime is importable."""
        fake_rt = types.ModuleType("music_brain.realtime")
        mgr = _make_manager()
        mgr._rt_checked = False
        mgr._rt_engine = None
        with patch.dict(sys.modules, {"music_brain.realtime": fake_rt}):
            result = mgr.realtime_available()
        assert result is True

    def test_result_cached(self):
        """Probe is run only once; subsequent calls use the cached value."""
        mgr = _make_manager()
        mgr._rt_checked = True
        mgr._rt_engine = None  # Force False

        with patch.object(mgr, "_probe_realtime") as probe:
            _ = mgr.realtime_available()
            _ = mgr.realtime_available()
            probe.assert_not_called()


# ---------------------------------------------------------------------------
# send_to_daw
# ---------------------------------------------------------------------------


class TestSendToDaw:
    def test_unknown_daw_returns_false(self):
        mgr = _make_manager()
        assert mgr.send_to_daw("fruityloops", {"note": 60}) is False

    def test_unknown_daw_empty_name_returns_false(self):
        mgr = _make_manager()
        assert mgr.send_to_daw("", None) is False

    def test_known_daw_import_error_returns_false(self):
        """If daw_bridges cannot be imported, return False gracefully."""
        mgr = _make_manager()
        with patch.dict(sys.modules, {
            "music_brain.agents": None,
            "music_brain.agents.daw_bridges": None,
        }):
            result = mgr.send_to_daw("ableton", {"note": 60})
        assert result is False

    @staticmethod
    def _fake_bridges_patch(bridge_attr_name: str, mock_bridge_cls):
        """Return a sys.modules patch dict with a fake daw_bridges module."""
        fake_bridges_mod = types.ModuleType("music_brain.agents.daw_bridges")
        setattr(fake_bridges_mod, bridge_attr_name, mock_bridge_cls)
        fake_agents_mod = types.ModuleType("music_brain.agents")
        fake_agents_mod.daw_bridges = fake_bridges_mod  # type: ignore[attr-defined]
        return {
            "music_brain.agents": fake_agents_mod,
            "music_brain.agents.daw_bridges": fake_bridges_mod,
        }

    def test_known_daw_connection_failure_returns_false(self):
        """If bridge.connect() returns False, send_to_daw returns False."""
        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.is_connected = False
        mock_bridge.connect.return_value = False
        mock_bridge_cls.return_value = mock_bridge

        mgr = _make_manager()
        patch_dict = self._fake_bridges_patch("AbletonDAWBridge", mock_bridge_cls)
        with patch.dict(sys.modules, patch_dict):
            result = mgr.send_to_daw("ableton", {"note": 60})
        assert result is False

    def test_known_daw_sends_successfully(self):
        """If bridge is connected, send_to_daw returns True."""
        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.send_note_on.return_value = None
        mock_bridge_cls.return_value = mock_bridge

        mgr = _make_manager()
        patch_dict = self._fake_bridges_patch("AbletonDAWBridge", mock_bridge_cls)
        with patch.dict(sys.modules, patch_dict):
            result = mgr.send_to_daw("ableton", {"note": 64, "velocity": 80})
        assert result is True
        mock_bridge.send_note_on.assert_called_once_with(64, 80, 0)

    def test_case_insensitive_daw_name(self):
        """DAW name lookup is case-insensitive."""
        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.send_note_on.return_value = None
        mock_bridge_cls.return_value = mock_bridge

        mgr = _make_manager()
        patch_dict = self._fake_bridges_patch("AbletonDAWBridge", mock_bridge_cls)
        with patch.dict(sys.modules, patch_dict):
            result = mgr.send_to_daw("ABLETON", 60)
        assert result is True

    def test_bridge_cached_across_calls(self):
        """The bridge is only instantiated once for repeated calls."""
        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.send_note_on.return_value = None
        mock_bridge_cls.return_value = mock_bridge

        mgr = _make_manager()
        patch_dict = self._fake_bridges_patch("ReaperBridge", mock_bridge_cls)
        with patch.dict(sys.modules, patch_dict):
            mgr.send_to_daw("reaper", 60)
            mgr.send_to_daw("reaper", 61)
        assert mock_bridge_cls.call_count == 1


# ---------------------------------------------------------------------------
# run_behavior_scenario
# ---------------------------------------------------------------------------


class TestRunBehaviorScenario:
    def test_returns_error_dict_when_missing(self):
        """Returns {"error": "behavior_lab not available"} if package absent."""
        mgr = _make_manager()
        # Ensure behavior_lab is not importable.
        sys.modules.pop("behavior_lab", None)
        sys.modules.pop("behavior_lab.runner", None)
        with patch.dict(sys.modules, {"behavior_lab": None, "behavior_lab.runner": None}):
            result = mgr.run_behavior_scenario({"name": "test"})
        assert result == {"error": "behavior_lab not available"}

    def test_delegates_to_run_scenario(self):
        """Calls behavior_lab.runner.run_scenario with the spec."""
        expected = {"status": "ok", "events": []}

        fake_runner = MagicMock()
        fake_runner.run_scenario = MagicMock(return_value=expected)

        fake_bl = types.ModuleType("behavior_lab")
        fake_bl.runner = fake_runner  # type: ignore[attr-defined]

        mgr = _make_manager()
        with patch.dict(
            sys.modules,
            {
                "behavior_lab": fake_bl,
                "behavior_lab.runner": fake_runner,
            },
        ):
            result = mgr.run_behavior_scenario({"name": "scene_1"})

        fake_runner.run_scenario.assert_called_once_with({"name": "scene_1"})
        assert result == expected

    def test_returns_error_on_runner_exception(self):
        """If run_scenario raises, a dict with 'error' key is returned."""
        fake_runner = MagicMock()
        fake_runner.run_scenario = MagicMock(side_effect=RuntimeError("boom"))

        fake_bl = types.ModuleType("behavior_lab")

        mgr = _make_manager()
        with patch.dict(
            sys.modules,
            {
                "behavior_lab": fake_bl,
                "behavior_lab.runner": fake_runner,
            },
        ):
            result = mgr.run_behavior_scenario({})

        assert "error" in result
        assert "boom" in result["error"]
