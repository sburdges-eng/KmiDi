"""
Tests for the root-level mcp_workstation re-export facade.

Verifies that ``from mcp_workstation import X`` and
``from mcp_workstation.submodule import X`` both work when the repo
root is on sys.path, and that the facade delegates to the canonical
implementation in scripts/mcp/mcp_workstation/.
"""

import sys
import os
import pytest

# conftest.py already ensures repo root is on sys.path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Flat top-level imports
# ---------------------------------------------------------------------------

class TestFlatImports:
    """All names exported from the root __init__ must be importable."""

    def test_models(self):
        from mcp_workstation import (
            AIAgent,
            ProposalStatus,
            ProposalCategory,
            PhaseStatus,
            Proposal,
            Phase,
            PhaseTask,
            WorkstationState,
        )
        assert AIAgent.CLAUDE is not None
        assert ProposalStatus is not None

    def test_orchestrator(self):
        from mcp_workstation import Workstation, get_workstation, shutdown_workstation
        assert callable(get_workstation)
        assert callable(shutdown_workstation)
        assert issubclass(Workstation, object)

    def test_proposals(self):
        from mcp_workstation import (
            ProposalManager,
            ProposalVote,
            format_proposal,
            format_proposal_list,
            get_proposal_template,
        )
        assert callable(format_proposal)
        assert callable(get_proposal_template)

    def test_phases(self):
        from mcp_workstation import (
            PhaseManager,
            IDAW_PHASES,
            format_phase_progress,
            get_next_actions,
        )
        assert isinstance(IDAW_PHASES, list)
        assert len(IDAW_PHASES) > 0

    def test_cpp_planner(self):
        from mcp_workstation import (
            CppTransitionPlanner,
            CppModule,
            CppTask,
            CppPriority,
            PortingStrategy,
            IDAW_CPP_MODULES,
            format_cpp_plan,
        )
        assert isinstance(IDAW_CPP_MODULES, list)
        assert callable(format_cpp_plan)

    def test_debug(self):
        from mcp_workstation import (
            DebugProtocol,
            DebugCategory,
            DebugEvent,
            LogLevel,
            get_debug,
            trace,
            log_debug,
            log_info,
            log_warning,
            log_error,
        )
        assert callable(log_info)
        assert callable(log_error)

    def test_ai_specializations(self):
        from mcp_workstation import (
            AICapabilities,
            TaskType,
            AI_CAPABILITIES,
            get_capabilities,
            get_best_agent_for_task,
            get_best_agents_for_task,
            get_agents_for_category,
            suggest_task_assignment,
            get_collaboration_strategy,
            print_ai_summary,
            get_task_assignment_summary,
        )
        assert isinstance(AI_CAPABILITIES, dict)
        assert callable(get_best_agent_for_task)

    def test_server(self):
        from mcp_workstation import get_mcp_tools, handle_tool_call, run_server
        assert callable(get_mcp_tools)
        assert callable(handle_tool_call)
        assert callable(run_server)

    def test_version(self):
        from mcp_workstation import __version__
        assert __version__ == "1.0.0"


# ---------------------------------------------------------------------------
# Submodule-style imports
# ---------------------------------------------------------------------------

class TestSubmoduleImports:
    """``from mcp_workstation.X import Y`` must also work."""

    def test_models_submodule(self):
        from mcp_workstation.models import AIAgent, WorkstationState
        assert AIAgent.CLAUDE is not None

    def test_orchestrator_submodule(self):
        from mcp_workstation.orchestrator import Workstation, get_workstation
        assert callable(get_workstation)

    def test_phases_submodule(self):
        from mcp_workstation.phases import IDAW_PHASES, PhaseManager
        assert len(IDAW_PHASES) > 0

    def test_debug_submodule(self):
        from mcp_workstation.debug import log_info, log_error
        assert callable(log_info)

    def test_ai_specializations_submodule(self):
        from mcp_workstation.ai_specializations import AI_CAPABILITIES, TaskType
        assert isinstance(AI_CAPABILITIES, dict)

    def test_proposals_submodule(self):
        from mcp_workstation.proposals import ProposalManager, format_proposal
        assert callable(format_proposal)

    def test_cpp_planner_submodule(self):
        from mcp_workstation.cpp_planner import CppTransitionPlanner, IDAW_CPP_MODULES
        assert isinstance(IDAW_CPP_MODULES, list)


# ---------------------------------------------------------------------------
# Identity / single-source-of-truth checks
# ---------------------------------------------------------------------------

class TestSingleSourceOfTruth:
    """The facade must delegate to the real implementation, not shadow it."""

    def test_canonical_package_unaffected(self):
        """scripts/mcp/mcp_workstation still importable when scripts/mcp is on sys.path."""
        scripts_mcp = os.path.join(REPO_ROOT, "scripts", "mcp")
        # The facade adds scripts/mcp to sys.path; the real package should be
        # accessible there under its original name via the private alias loaded
        # by the facade.
        assert "_mcp_workstation_impl" in sys.modules, (
            "Facade must have loaded the real package as _mcp_workstation_impl"
        )
        impl = sys.modules["_mcp_workstation_impl"]
        assert impl.__version__ == "1.0.0"
        # The canonical directory must still be present and unmodified on disk.
        canonical_init = os.path.join(scripts_mcp, "mcp_workstation", "__init__.py")
        assert os.path.isfile(canonical_init), "Canonical __init__.py must still exist on disk"

    def test_no_circular_import(self):
        """Importing mcp_workstation twice must be idempotent (no crash)."""
        import mcp_workstation
        import mcp_workstation as mw2  # noqa: F401
        assert mcp_workstation is mw2

    def test_workstation_is_real_class(self):
        """Workstation from facade must be the actual class, not a stub."""
        from mcp_workstation import Workstation
        # Should be instantiable (basic smoke test).
        ws = Workstation()
        assert ws is not None

    def test_symbols_are_identical_objects(self):
        import sys
        from mcp_workstation import AIAgent
        impl = sys.modules["_mcp_workstation_impl"]
        assert AIAgent is impl.AIAgent, "Facade must re-export the same object, not a copy"
