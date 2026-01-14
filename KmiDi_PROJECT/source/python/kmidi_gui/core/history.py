"""Session history and undo/redo system.

Supports non-destructive experimentation with per-component undo/redo stacks.
Undo applies to user actions only - AI suggestions are not actions until committed.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HistoryComponent(str, Enum):
    """Components that have separate undo/redo stacks."""
    INTENT = "intent"
    ML_SETTINGS = "ml_settings"
    TRUST = "trust"
    MIDI = "midi"
    PRESET = "preset"


@dataclass
class HistoryAction(ABC):
    """Abstract base class for undoable actions.

    All user actions that should be undoable must implement this interface.
    AI suggestions are not actions until committed.
    """

    action_type: str = ""
    description: str = ""

    @abstractmethod
    def apply(self) -> None:
        """Apply the action."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Reverse the action."""
        pass

    def to_dict(self) -> Dict:
        """Serialize action to dictionary."""
        return {
            "action_type": self.action_type,
            "description": self.description,
        }


@dataclass
class ApplyMLSuggestionAction(HistoryAction):
    """Action for committing an ML suggestion."""

    suggestion_data: Dict = field(default_factory=dict)
    previous_state: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.action_type = "apply_ml_suggestion"
        if not self.description:
            self.description = "Apply ML suggestion"

    def apply(self) -> None:
        """Apply ML suggestion."""
        # TODO: Implement actual application
        logger.info(f"Applying ML suggestion: {self.description}")

    def undo(self) -> None:
        """Revert ML suggestion."""
        # TODO: Implement actual reversion
        logger.info(f"Undoing ML suggestion: {self.description}")


@dataclass
class ChangeIntentAction(HistoryAction):
    """Action for changing intent schema."""

    new_intent: Dict = field(default_factory=dict)
    previous_intent: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.action_type = "change_intent"
        if not self.description:
            self.description = "Change intent schema"

    def apply(self) -> None:
        """Apply intent change."""
        # TODO: Implement actual application
        logger.info(f"Applying intent change: {self.description}")

    def undo(self) -> None:
        """Revert intent change."""
        # TODO: Implement actual reversion
        logger.info(f"Undoing intent change: {self.description}")


@dataclass
class ChangeTrustSettingsAction(HistoryAction):
    """Action for updating trust settings."""

    new_settings: Dict[str, float] = field(default_factory=dict)
    previous_settings: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.action_type = "change_trust"
        if not self.description:
            self.description = "Change trust settings"

    def apply(self) -> None:
        """Apply trust settings change."""
        # TODO: Implement actual application
        logger.info(f"Applying trust settings change: {self.description}")

    def undo(self) -> None:
        """Revert trust settings change."""
        # TODO: Implement actual reversion
        logger.info(f"Undoing trust settings change: {self.description}")


@dataclass
class CommitMIDIAction(HistoryAction):
    """Action for committing generated MIDI."""

    midi_data: Dict = field(default_factory=dict)
    previous_midi: Optional[Dict] = None

    def __post_init__(self):
        self.action_type = "commit_midi"
        if not self.description:
            self.description = "Commit MIDI"

    def apply(self) -> None:
        """Apply MIDI commit."""
        # TODO: Implement actual application
        logger.info(f"Applying MIDI commit: {self.description}")

    def undo(self) -> None:
        """Revert MIDI commit."""
        # TODO: Implement actual reversion
        logger.info(f"Undoing MIDI commit: {self.description}")


@dataclass
class ApplyPresetAction(HistoryAction):
    """Action for applying a preset."""

    preset_id: str = ""
    previous_state: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.action_type = "apply_preset"
        if not self.description:
            self.description = f"Apply preset {self.preset_id[:8]}"

    def apply(self) -> None:
        """Apply preset."""
        # TODO: Implement actual application
        logger.info(f"Applying preset: {self.description}")

    def undo(self) -> None:
        """Revert preset application."""
        # TODO: Implement actual reversion
        logger.info(f"Undoing preset application: {self.description}")


class HistoryManager:
    """Manages per-component undo/redo stacks.

    Each component has its own undo/redo stack, allowing independent
    undo operations for different aspects of the application.
    """

    def __init__(self):
        """Initialize history manager."""
        # Per-component undo/redo stacks
        self._undo_stacks: Dict[str, List[HistoryAction]] = {
            component.value: [] for component in HistoryComponent
        }
        self._redo_stacks: Dict[str, List[HistoryAction]] = {
            component.value: [] for component in HistoryComponent
        }

        # Transaction grouping support
        self._transaction_groups: Dict[str, List[HistoryAction]] = {}
        self._in_transaction: Dict[str, bool] = {
            component.value: False for component in HistoryComponent
        }

    def push_action(self, component: str, action: HistoryAction) -> None:
        """Push action onto undo stack.

        Args:
            component: Component name (from HistoryComponent enum)
            action: Action to push
        """
        if component not in self._undo_stacks:
            logger.warning(f"Unknown component: {component}")
            return

        # If in transaction, add to transaction group
        if self._in_transaction.get(component, False):
            if component not in self._transaction_groups:
                self._transaction_groups[component] = []
            self._transaction_groups[component].append(action)
            return

        # Clear redo stack when new action is pushed
        self._redo_stacks[component].clear()

        # Push to undo stack
        self._undo_stacks[component].append(action)

        # Limit stack size (prevent memory issues)
        max_stack_size = 100
        if len(self._undo_stacks[component]) > max_stack_size:
            self._undo_stacks[component].pop(0)

        logger.debug(f"Pushed action to {component}: {action.description}")

    def undo(self, component: str) -> Optional[HistoryAction]:
        """Undo last action for component.

        Args:
            component: Component name

        Returns:
            Undone action, or None if stack is empty
        """
        if component not in self._undo_stacks:
            logger.warning(f"Unknown component: {component}")
            return None

        if not self._undo_stacks[component]:
            logger.debug(f"No actions to undo for {component}")
            return None

        # Pop from undo stack
        action = self._undo_stacks[component].pop()

        try:
            # Undo the action
            action.undo()

            # Push to redo stack
            self._redo_stacks[component].append(action)

            logger.info(f"Undid action in {component}: {action.description}")
            return action
        except Exception as e:
            logger.error(f"Failed to undo action: {e}")
            # Put action back on stack
            self._undo_stacks[component].append(action)
            return None

    def redo(self, component: str) -> Optional[HistoryAction]:
        """Redo last undone action for component.

        Args:
            component: Component name

        Returns:
            Redone action, or None if stack is empty
        """
        if component not in self._redo_stacks:
            logger.warning(f"Unknown component: {component}")
            return None

        if not self._redo_stacks[component]:
            logger.debug(f"No actions to redo for {component}")
            return None

        # Pop from redo stack
        action = self._redo_stacks[component].pop()

        try:
            # Redo the action
            action.apply()

            # Push back to undo stack
            self._undo_stacks[component].append(action)

            logger.info(f"Redid action in {component}: {action.description}")
            return action
        except Exception as e:
            logger.error(f"Failed to redo action: {e}")
            # Put action back on stack
            self._redo_stacks[component].append(action)
            return None

    def can_undo(self, component: str) -> bool:
        """Check if undo is available for component.

        Args:
            component: Component name

        Returns:
            True if undo is available
        """
        return component in self._undo_stacks and len(self._undo_stacks[component]) > 0

    def can_redo(self, component: str) -> bool:
        """Check if redo is available for component.

        Args:
            component: Component name

        Returns:
            True if redo is available
        """
        return component in self._redo_stacks and len(self._redo_stacks[component]) > 0

    def clear(self, component: str) -> None:
        """Clear undo/redo stacks for component.

        Args:
            component: Component name
        """
        if component in self._undo_stacks:
            self._undo_stacks[component].clear()
        if component in self._redo_stacks:
            self._redo_stacks[component].clear()
        logger.debug(f"Cleared history for {component}")

    def start_group(self, component: str) -> None:
        """Start transaction grouping for component.

        All actions pushed until end_group() will be grouped together.

        Args:
            component: Component name
        """
        if component not in self._in_transaction:
            logger.warning(f"Unknown component: {component}")
            return

        self._in_transaction[component] = True
        self._transaction_groups[component] = []
        logger.debug(f"Started transaction group for {component}")

    def end_group(self, component: str) -> None:
        """End transaction grouping for component.

        All actions since start_group() will be treated as a single action.

        Args:
            component: Component name
        """
        if component not in self._in_transaction:
            logger.warning(f"Unknown component: {component}")
            return

        if not self._in_transaction[component]:
            logger.warning(f"No active transaction for {component}")
            return

        self._in_transaction[component] = False

        # Push grouped actions as a single compound action
        if component in self._transaction_groups and self._transaction_groups[component]:
            # For now, just push all actions individually
            # Future: Create CompoundAction class
            for action in self._transaction_groups[component]:
                self.push_action(component, action)

            self._transaction_groups[component].clear()

        logger.debug(f"Ended transaction group for {component}")
