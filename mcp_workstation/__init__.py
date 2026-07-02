"""
mcp_workstation - Re-export facade for scripts/mcp/mcp_workstation/

This package is a thin shim that makes ``from mcp_workstation.X import Y``
work when the repo root is on sys.path.  The single source of truth is
``scripts/mcp/mcp_workstation/``; nothing is duplicated here.
"""

import importlib.util
import sys
import os as _os

# ---------------------------------------------------------------------------
# Ensure the canonical package location is on sys.path so that the real
# mcp_workstation package can be loaded under a temporary alias.
# ---------------------------------------------------------------------------
_SCRIPTS_MCP = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                              "scripts", "mcp")

# Note: this mutates sys.path for the lifetime of the process. Intentional:
# the facade needs scripts/mcp on the path so the canonical package's
# relative imports resolve correctly.
if _SCRIPTS_MCP not in sys.path:
    sys.path.insert(0, _SCRIPTS_MCP)

# ---------------------------------------------------------------------------
# Load the real package under a private alias so that re-importing this
# facade does not recurse into itself.  If it is already present (e.g. the
# canonical path was already on sys.path before this facade was imported),
# just grab the cached module.
# ---------------------------------------------------------------------------
_REAL_PKG_NAME = "_mcp_workstation_impl"

if _REAL_PKG_NAME not in sys.modules:
    # Load mcp_workstation from scripts/mcp/ and register it under the alias.
    _spec = importlib.util.spec_from_file_location(
        _REAL_PKG_NAME,
        _os.path.join(_SCRIPTS_MCP, "mcp_workstation", "__init__.py"),
        submodule_search_locations=[_os.path.join(_SCRIPTS_MCP, "mcp_workstation")],
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_REAL_PKG_NAME] = _mod
    _spec.loader.exec_module(_mod)

_impl = sys.modules[_REAL_PKG_NAME]

# ---------------------------------------------------------------------------
# Re-export everything from the real package into this namespace.
# ---------------------------------------------------------------------------

# Models
from _mcp_workstation_impl import (  # noqa: E402
    AIAgent,
    ProposalStatus,
    ProposalCategory,
    PhaseStatus,
    Proposal,
    Phase,
    PhaseTask,
    WorkstationState,
)

# Orchestrator
from _mcp_workstation_impl import (  # noqa: E402
    Workstation,
    get_workstation,
    shutdown_workstation,
)

# Proposals
from _mcp_workstation_impl import (  # noqa: E402
    ProposalManager,
    ProposalVote,
    format_proposal,
    format_proposal_list,
    get_proposal_template,
)

# Phases
from _mcp_workstation_impl import (  # noqa: E402
    PhaseManager,
    IDAW_PHASES,
    format_phase_progress,
    get_next_actions,
)

# C++ Planner
from _mcp_workstation_impl import (  # noqa: E402
    CppTransitionPlanner,
    CppModule,
    CppTask,
    CppPriority,
    PortingStrategy,
    IDAW_CPP_MODULES,
    format_cpp_plan,
)

# Debug
from _mcp_workstation_impl import (  # noqa: E402
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

# AI Specializations
from _mcp_workstation_impl import (  # noqa: E402
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

# Server
from _mcp_workstation_impl import (  # noqa: E402
    get_mcp_tools,
    handle_tool_call,
    run_server,
)

__version__ = _impl.__version__
__author__ = getattr(_impl, "__author__", "DAiW")

# ---------------------------------------------------------------------------
# Register the real submodules under the mcp_workstation.* namespace so that
# ``from mcp_workstation.models import AIAgent`` (and similar) also work.
# ---------------------------------------------------------------------------
_SUBMODULES = [
    "models",
    "orchestrator",
    "proposals",
    "phases",
    "cpp_planner",
    "debug",
    "ai_specializations",
    "server",
]

for _submod in _SUBMODULES:
    _real_name = f"{_REAL_PKG_NAME}.{_submod}"
    _alias = f"mcp_workstation.{_submod}"
    if _real_name in sys.modules and _alias not in sys.modules:
        sys.modules[_alias] = sys.modules[_real_name]

__all__ = [
    # Models
    "AIAgent",
    "ProposalStatus",
    "ProposalCategory",
    "PhaseStatus",
    "Proposal",
    "Phase",
    "PhaseTask",
    "WorkstationState",
    # Orchestrator
    "Workstation",
    "get_workstation",
    "shutdown_workstation",
    # Proposals
    "ProposalManager",
    "ProposalVote",
    "format_proposal",
    "format_proposal_list",
    "get_proposal_template",
    # Phases
    "PhaseManager",
    "IDAW_PHASES",
    "format_phase_progress",
    "get_next_actions",
    # C++ Planner
    "CppTransitionPlanner",
    "CppModule",
    "CppTask",
    "CppPriority",
    "PortingStrategy",
    "IDAW_CPP_MODULES",
    "format_cpp_plan",
    # Debug
    "DebugProtocol",
    "DebugCategory",
    "DebugEvent",
    "LogLevel",
    "get_debug",
    "trace",
    "log_debug",
    "log_info",
    "log_warning",
    "log_error",
    # AI Specializations
    "AICapabilities",
    "TaskType",
    "AI_CAPABILITIES",
    "get_capabilities",
    "get_best_agent_for_task",
    "get_best_agents_for_task",
    "get_agents_for_category",
    "suggest_task_assignment",
    "get_collaboration_strategy",
    "print_ai_summary",
    "get_task_assignment_summary",
    # Server
    "get_mcp_tools",
    "handle_tool_call",
    "run_server",
    # Meta
    "__version__",
]
