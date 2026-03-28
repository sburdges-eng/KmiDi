"""
behavior_lab — Real-time behavioral feedback loop (Phase 3).

Analyzes incoming engine events and adjusts the Mind's parameters
through the ComprehensiveIntegrationManager.

Core components:
    ClosedLoopController - Adjusts penta_core parameters in response to engine state
    run_scenario         - Execute a behavior scenario (sequence of rules + triggers)
    Scenario / Rule      - Data structures for declarative behavior definitions
"""

from .controller import ClosedLoopController
from .scenarios import Scenario, Rule, run_scenario

__all__ = [
    "ClosedLoopController",
    "Scenario",
    "Rule",
    "run_scenario",
]
