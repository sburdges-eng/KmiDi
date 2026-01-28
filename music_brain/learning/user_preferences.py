"""
User Preference Learning System

Tracks user parameter adjustments, emotion selections, MIDI acceptance/rejection,
and rule-break modifications to build a personalized user profile.

Stores preferences locally in JSON format at ~/.kelly/user_preferences.json
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import statistics
import uuid


@dataclass
class ParameterAdjustment:
    """Record of a single parameter adjustment."""
    parameter_name: str  # e.g., "valence", "arousal", "intensity"
    old_value: float
    new_value: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class EmotionSelection:
    """Record of emotion wheel selection."""
    emotion_name: str
    valence: float
    arousal: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MidiGenerationEvent:
    """Record of MIDI generation and user response."""
    generation_id: str
    intent_text: str
    parameters: Dict[str, float]  # All parameter values at generation time
    emotion: Optional[str] = None
    rule_breaks: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    accepted: Optional[bool] = None  # True/False if user gave explicit feedback
    modifications: List[ParameterAdjustment] = field(default_factory=list)
    # If user modified parameters after generation, track those changes


@dataclass
class RuleBreakModification:
    """Record of rule-break additions/removals."""
    rule_break: str
    action: str  # "added" or "removed"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestionEvent:
    """Record of suggestion interaction."""
    suggestion_id: str  # Unique identifier for the suggestion
    suggestion_type: str  # "parameter", "emotion", "rule_break", "style"
    action: str  # "shown", "accepted", "dismissed"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)  # Current state when shown


@dataclass
class UserPreferenceProfile:
    """Complete user preference profile."""
    user_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Parameter adjustment history
    parameter_adjustments: List[ParameterAdjustment] = field(default_factory=list)

    # Emotion selection history
    emotion_selections: List[EmotionSelection] = field(default_factory=list)

    # MIDI generation history
    midi_generations: List[MidiGenerationEvent] = field(default_factory=list)

    # Rule-break modifications
    rule_break_modifications: List[RuleBreakModification] = field(default_factory=list)

    # Suggestion interactions
    suggestion_events: List[SuggestionEvent] = field(default_factory=list)

    # Genre/style preferences (learned over time)
    genre_preferences: Dict[str, float] = field(default_factory=dict)  # genre -> preference score

    # Computed statistics (cached)
    _statistics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "parameter_adjustments": [asdict(adj) for adj in self.parameter_adjustments],
            "emotion_selections": [asdict(sel) for sel in self.emotion_selections],
            "midi_generations": [asdict(gen) for gen in self.midi_generations],
            "rule_break_modifications": [asdict(mod) for mod in self.rule_break_modifications],
            "suggestion_events": [asdict(evt) for evt in self.suggestion_events],
            "genre_preferences": self.genre_preferences,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferenceProfile":
        """Create from dictionary."""
        profile = cls(
            user_id=data.get("user_id", "default"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            genre_preferences=data.get("genre_preferences", {}),
        )

        # Reconstruct lists
        for adj_data in data.get("parameter_adjustments", []):
            profile.parameter_adjustments.append(ParameterAdjustment(**adj_data))

        for sel_data in data.get("emotion_selections", []):
            profile.emotion_selections.append(EmotionSelection(**sel_data))

        for gen_data in data.get("midi_generations", []):
            gen = MidiGenerationEvent(**gen_data)
            # Reconstruct modifications
            gen.modifications = [
                ParameterAdjustment(**mod_data)
                for mod_data in gen_data.get("modifications", [])
            ]
            profile.midi_generations.append(gen)

        for mod_data in data.get("rule_break_modifications", []):
            profile.rule_break_modifications.append(RuleBreakModification(**mod_data))

        for evt_data in data.get("suggestion_events", []):
            profile.suggestion_events.append(SuggestionEvent(**evt_data))

        return profile


class UserPreferenceModel:
    """
    Main class for tracking and learning user preferences.

    Tracks:
    - Parameter adjustment patterns
    - Emotion selection frequency
    - Generated MIDI acceptance/rejection rates
    - Rule-break modifications
    - Genre/style preferences
    """

    def __init__(self, user_id: str = "default", preferences_path: Optional[Path] = None):
        """
        Initialize user preference model.

        Args:
            user_id: Unique identifier for user
            preferences_path: Path to preferences JSON file (defaults to ~/.kelly/user_preferences.json)
        """
        self.user_id = user_id

        if preferences_path is None:
            # Default to ~/.kelly/user_preferences.json
            home = Path.home()
            kelly_dir = home / ".kelly"
            kelly_dir.mkdir(exist_ok=True)
            preferences_path = kelly_dir / "user_preferences.json"

        self.preferences_path = preferences_path
        self.profile = self._load_profile()

    def _load_profile(self) -> UserPreferenceProfile:
        """Load user profile from disk."""
        if self.preferences_path.exists():
            try:
                with open(self.preferences_path, 'r') as f:
                    data = json.load(f)
                    # Support both single user and multi-user formats
                    if "user_id" in data:
                        return UserPreferenceProfile.from_dict(data)
                    elif self.user_id in data:
                        return UserPreferenceProfile.from_dict(data[self.user_id])
                    else:
                        # Legacy format or first user
                        return UserPreferenceProfile.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error loading preferences: {e}. Starting with fresh profile.")
                return UserPreferenceProfile(user_id=self.user_id)
        else:
            return UserPreferenceProfile(user_id=self.user_id)

    def _save_profile(self):
        """Save user profile to disk."""
        self.profile.last_updated = datetime.now().isoformat()
        try:
            # Try to preserve other users if file exists
            existing_data = {}
            if self.preferences_path.exists():
                try:
                    with open(self.preferences_path, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Update or add this user's data
            existing_data[self.user_id] = self.profile.to_dict()

            with open(self.preferences_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")

    def record_parameter_adjustment(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a parameter adjustment."""
        adjustment = ParameterAdjustment(
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            context=context or {}
        )
        self.profile.parameter_adjustments.append(adjustment)
        self._save_profile()

    def record_emotion_selection(
        self,
        emotion_name: str,
        valence: float,
        arousal: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an emotion wheel selection."""
        selection = EmotionSelection(
            emotion_name=emotion_name,
            valence=valence,
            arousal=arousal,
            context=context or {}
        )
        self.profile.emotion_selections.append(selection)
        self._save_profile()

    def record_midi_generation(
        self,
        generation_id: str,
        intent_text: str,
        parameters: Dict[str, float],
        emotion: Optional[str] = None,
        rule_breaks: Optional[List[str]] = None
    ) -> str:
        """
        Record a MIDI generation event.

        Returns:
            generation_id for tracking subsequent modifications
        """
        event = MidiGenerationEvent(
            generation_id=generation_id,
            intent_text=intent_text,
            parameters=parameters.copy(),
            emotion=emotion,
            rule_breaks=rule_breaks or []
        )
        self.profile.midi_generations.append(event)
        self._save_profile()
        return generation_id

    def record_midi_feedback(
        self,
        generation_id: str,
        accepted: bool
    ):
        """Record explicit user feedback (thumbs up/down) on generated MIDI."""
        for event in self.profile.midi_generations:
            if event.generation_id == generation_id:
                event.accepted = accepted
                self._save_profile()
                return
        # If not found, create a new event (shouldn't happen, but handle gracefully)
        print(f"Warning: Generation ID {generation_id} not found for feedback")

    def record_midi_modification(
        self,
        generation_id: str,
        parameter_name: str,
        old_value: float,
        new_value: float
    ):
        """Record a parameter modification after MIDI generation."""
        for event in self.profile.midi_generations:
            if event.generation_id == generation_id:
                adjustment = ParameterAdjustment(
                    parameter_name=parameter_name,
                    old_value=old_value,
                    new_value=new_value
                )
                event.modifications.append(adjustment)
                self._save_profile()
                return
        print(f"Warning: Generation ID {generation_id} not found for modification")

    def record_rule_break_modification(
        self,
        rule_break: str,
        action: str,  # "added" or "removed"
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a rule-break addition or removal."""
        modification = RuleBreakModification(
            rule_break=rule_break,
            action=action,
            context=context or {}
        )
        self.profile.rule_break_modifications.append(modification)
        self._save_profile()

    def get_parameter_statistics(self) -> Dict[str, Any]:
        """
        Get statistical analysis of parameter adjustments.

        Returns:
            Dictionary with statistics for each parameter:
            - most_used_range: (min, max) tuple of most common range
            - average_value: Average final value
            - adjustment_frequency: How often this parameter is adjusted
            - preferred_zones: List of preferred value ranges
        """
        stats = {}

        # Group adjustments by parameter
        by_parameter = defaultdict(list)
        for adj in self.profile.parameter_adjustments:
            by_parameter[adj.parameter_name].append(adj.new_value)

        for param_name, values in by_parameter.items():
            if not values:
                continue

            stats[param_name] = {
                "average_value": statistics.mean(values),
                "median_value": statistics.median(values),
                "min_value": min(values),
                "max_value": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "adjustment_count": len(values),
            }

            # Find most common range (bin into 0.1 increments)
            bins = defaultdict(int)
            for val in values:
                bin_key = round(val * 10) / 10  # Round to 0.1
                bins[bin_key] += 1

            if bins:
                most_common_bin = max(bins.items(), key=lambda x: x[1])[0]
                stats[param_name]["most_used_range"] = (most_common_bin, most_common_bin + 0.1)

        return stats

    def get_emotion_preferences(self) -> Dict[str, int]:
        """Get frequency of emotion selections."""
        emotion_counts = defaultdict(int)
        for selection in self.profile.emotion_selections:
            emotion_counts[selection.emotion_name] += 1
        return dict(emotion_counts)

    def get_acceptance_rate(self) -> float:
        """Get MIDI generation acceptance rate (0.0 to 1.0)."""
        feedbacks = [
            event.accepted
            for event in self.profile.midi_generations
            if event.accepted is not None
        ]
        if not feedbacks:
            return 0.5  # Default to neutral if no feedback yet
        return sum(feedbacks) / len(feedbacks)

    def get_preferred_parameter_ranges(self) -> Dict[str, tuple]:
        """
        Get preferred parameter ranges based on accepted generations.

        Returns:
            Dictionary mapping parameter name to (min, max) preferred range
        """
        # Get parameters from accepted generations
        accepted_params = []
        for event in self.profile.midi_generations:
            if event.accepted is True:
                accepted_params.append(event.parameters)

        if not accepted_params:
            return {}

        # Aggregate ranges
        ranges = {}
        param_names = set()
        for params in accepted_params:
            param_names.update(params.keys())

        for param_name in param_names:
            values = [p[param_name] for p in accepted_params if param_name in p]
            if values:
                ranges[param_name] = (min(values), max(values))

        return ranges

    def clear_preferences(self):
        """Clear all learned preferences (reset to defaults)."""
        self.profile = UserPreferenceProfile(user_id=self.user_id)
        self._save_profile()

    def get_profile(self) -> UserPreferenceProfile:
        """Get the current user profile."""
        return self.profile
