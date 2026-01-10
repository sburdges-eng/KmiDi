"""
Preference Integration Utilities

Helper functions to integrate preference learning into UI components.
Provides hooks and utilities for tracking user interactions.

Part of the "All-Knowing Interactive Musical Customization System".
"""

from typing import Optional, Dict, Any, Callable
from .user_preferences import UserPreferenceModel


class PreferenceTracker:
    """
    Helper class to track user interactions and record preferences.

    Usage:
        tracker = PreferenceTracker()
        tracker.on_parameter_change("valence", 0.5, 0.7)
        tracker.on_emotion_selected("grief")
    """

    def __init__(self, preference_model: Optional[UserPreferenceModel] = None):
        """Initialize tracker with preference model."""
        self.model = preference_model or UserPreferenceModel()
        self.current_context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set current context for tracking."""
        self.current_context.update(kwargs)

    def clear_context(self):
        """Clear current context."""
        self.current_context = {}

    def on_parameter_change(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float
    ):
        """Record parameter change."""
        self.model.record_parameter_adjustment(
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            context=self.current_context.copy()
        )

    def on_emotion_selected(
        self,
        emotion_name: str,
        emotion_id: Optional[int] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        intensity: Optional[float] = None
    ):
        """Record emotion selection."""
        self.model.record_emotion_selection(
            emotion_name=emotion_name,
            emotion_id=emotion_id,
            valence=valence,
            arousal=arousal,
            intensity=intensity
        )

    def on_generation_accepted(self, parameters: Dict[str, float], emotion: Optional[str] = None):
        """Record accepted generation."""
        self.model.record_generation(
            parameters=parameters,
            emotion=emotion,
            accepted=True
        )

    def on_generation_rejected(self, parameters: Dict[str, float], emotion: Optional[str] = None):
        """Record rejected generation."""
        self.model.record_generation(
            parameters=parameters,
            emotion=emotion,
            accepted=False
        )

    def on_rule_break_added(self, rule_break_type: str):
        """Record rule-break addition."""
        self.model.record_rule_break_modification(
            rule_break_type=rule_break_type,
            action="added",
            context=self.current_context.copy()
        )

    def on_rule_break_removed(self, rule_break_type: str):
        """Record rule-break removal."""
        self.model.record_rule_break_modification(
            rule_break_type=rule_break_type,
            action="removed",
            context=self.current_context.copy()
        )

    def on_midi_edit(
        self,
        edit_type: str,
        target_part: str,
        change_details: Dict[str, Any]
    ):
        """Record MIDI edit."""
        self.model.record_midi_edit(
            edit_type=edit_type,
            target_part=target_part,
            change_details=change_details
        )


def create_parameter_change_handler(tracker: PreferenceTracker) -> Callable:
    """
    Create a callback function for parameter changes.

    Returns:
        Callback function that can be attached to slider change listeners
    """
    def handler(parameter_name: str, old_value: float, new_value: float):
        tracker.on_parameter_change(parameter_name, old_value, new_value)

    return handler


def create_emotion_selection_handler(tracker: PreferenceTracker) -> Callable:
    """
    Create a callback function for emotion selections.

    Returns:
        Callback function that can be attached to emotion wheel selection handlers
    """
    def handler(emotion_name: str, **kwargs):
        tracker.on_emotion_selected(emotion_name, **kwargs)

    return handler
