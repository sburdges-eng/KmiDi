"""GUI module - Qt widgets only, no business logic."""

from kmidi_gui.gui.main_window import MainWindow
from kmidi_gui.gui.docks import AIAssistantDock, LogsDock
from kmidi_gui.gui.parameter_panel import EmotionParameterPanel, TechnicalParameterPanel
from kmidi_gui.gui.preferences_dialog import PreferencesDialog

__version__ = "1.0.0"

__all__ = [
    "MainWindow",
    "AIAssistantDock",
    "LogsDock",
    "EmotionParameterPanel",
    "TechnicalParameterPanel",
    "PreferencesDialog",
]
