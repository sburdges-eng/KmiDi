"""
Tests for newly implemented mcp_workstation placeholder functions.

Tests debug.py, ai_specializations.py, and cpp_planner.py implementations.
"""

import time
import pytest
from mcp_workstation.debug import (
    get_debug, DebugCategory, log_error, log_warning, measure_performance
)
from mcp_workstation.ai_specializations import (
    suggest_task_assignment, get_capabilities, TaskType, format_capability_report,
    get_agent_for_task_type
)
from mcp_workstation.cpp_planner import CppTransitionPlanner, format_cpp_plan
from mcp_workstation.models import AIAgent


class TestDebugImplementation:
    """Test debug.py error tracking and performance monitoring."""

    def test_error_logging(self):
        """Test error logging with stack traces."""
        debug = get_debug()
        debug.clear()  # Start fresh

        # Log an error with exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_error("Test error message", exception=e, details={"context": "test"})

        errors = debug.get_errors(limit=10)
        assert len(errors) == 1
        assert errors[0].message == "Test error message"
        assert errors[0].category == "error"
        assert errors[0].stack_trace is not None
        assert "ValueError" in errors[0].stack_trace

    def test_warning_logging(self):
        """Test warning logging."""
        debug = get_debug()
        debug.clear()

        log_warning("Test warning", details={"level": "medium"})

        warnings = debug.get_warnings(limit=10)
        assert len(warnings) == 1
        assert warnings[0].message == "Test warning"
        assert warnings[0].details["level"] == "medium"

    def test_performance_measurement(self):
        """Test performance timing with context manager."""
        debug = get_debug()
        debug.clear()

        # Measure a simple operation
        with measure_performance("test_operation"):
            time.sleep(0.01)  # 10ms

        report = debug.get_performance_report()
        assert report["total_operations"] == 1
        assert "test_operation" in report["operations"]
        stats = report["operations"]["test_operation"]
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 10  # Should be at least 10ms

    def test_performance_statistics(self):
        """Test performance statistics calculation."""
        debug = get_debug()
        debug.clear()

        # Record multiple timings
        for i in range(10):
            debug.record_performance("test_op", float(i * 10))

        report = debug.get_performance_report()
        stats = report["operations"]["test_op"]

        assert stats["count"] == 10
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 90.0
        assert stats["mean_ms"] == 45.0
        assert "p50_ms" in stats
        assert "p95_ms" in stats

    def test_get_summary(self):
        """Test overall debug summary."""
        debug = get_debug()
        debug.clear()

        log_error("Error 1")
        log_error("Error 2")
        log_warning("Warning 1")
        debug.record_performance("op1", 100.0)

        summary = debug.get_summary()
        assert summary["error_count"] == 2
        assert summary["warning_count"] == 1
        assert summary["total_events"] == 3
        assert summary["total_metrics"] == 1


class TestAISpecializations:
    """Test ai_specializations.py task assignment logic."""

    def test_task_assignment_direct_match(self):
        """Test direct task type to agent mapping."""
        tasks = [
            ("parse_text", TaskType.LLM),
            ("generate_midi", TaskType.MIDI),
            ("create_image", TaskType.IMAGE),
            ("generate_audio", TaskType.AUDIO),
        ]

        assignments = suggest_task_assignment(tasks)

        assert assignments["parse_text"] == AIAgent.LLM
        assert assignments["generate_midi"] == AIAgent.MIDI
        assert assignments["create_image"] == AIAgent.IMAGE
        assert assignments["generate_audio"] == AIAgent.AUDIO

    def test_empty_task_list(self):
        """Test handling of empty task list."""
        assignments = suggest_task_assignment([])
        assert assignments == {}

    def test_get_capabilities(self):
        """Test agent capability retrieval."""
        llm_caps = get_capabilities(AIAgent.LLM)

        assert llm_caps.display_name == "LLM Reasoning Engine"
        assert llm_caps.description != ""
        assert len(llm_caps.strengths) > 0
        assert len(llm_caps.special_abilities) > 0
        assert len(llm_caps.limitations) > 0

    def test_capability_report_formatting(self):
        """Test human-readable capability report."""
        report = format_capability_report(AIAgent.MIDI)

        assert "MIDI Generation Pipeline" in report
        assert "Strengths:" in report
        assert "Special Abilities:" in report
        assert "Limitations:" in report

    def test_get_agent_for_task_type(self):
        """Test direct task type to agent lookup."""
        assert get_agent_for_task_type(TaskType.LLM) == AIAgent.LLM
        assert get_agent_for_task_type(TaskType.MIDI) == AIAgent.MIDI
        assert get_agent_for_task_type(TaskType.IMAGE) == AIAgent.IMAGE
        assert get_agent_for_task_type(TaskType.AUDIO) == AIAgent.AUDIO


class TestCppPlannerDeferred:
    """Test cpp_planner.py deferred functionality."""

    def test_planner_returns_deferred_status(self):
        """Test that planner correctly indicates deferred status."""
        planner = CppTransitionPlanner()

        summary = planner.get_progress_summary()
        assert summary["status"] == "deferred"
        assert summary["total_modules"] == 0

    def test_planner_to_dict(self):
        """Test planner serialization includes deferred flag."""
        planner = CppTransitionPlanner()

        state = planner.to_dict()
        assert state["deferred"] is True
        assert len(state["tasks"]) == 0

    def test_format_cpp_plan_shows_deferral(self):
        """Test formatted plan shows deferral notice."""
        planner = CppTransitionPlanner()

        plan_text = format_cpp_plan(planner)
        assert "DEFERRED" in plan_text
        assert "Status: DEFERRED" in plan_text

    def test_module_operations_are_noops(self):
        """Test that module operations don't crash (no-ops)."""
        planner = CppTransitionPlanner()

        # These should not crash
        planner.start_module("test_module", AIAgent.LLM)
        planner.update_module_progress("test_module", 0.5)

        # Should still return empty/default values
        assert planner.get_build_plan() != ""  # Has deferred notice
        assert "deferred" in planner.get_build_plan().lower()
