"""Preferences dialog for KmiDi GUI.

Settings:
- API URL
- Theme preferences
- Logging level
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal


class PreferencesDialog(QDialog):
    """Preferences dialog."""
    
    # Signal emitted when preferences are saved
    preferences_saved = Signal(dict)
    
    def __init__(self, parent=None, current_prefs: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # API Settings
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout(api_group)
        
        api_url_layout = QHBoxLayout()
        api_url_label = QLabel("API URL:")
        api_url_layout.addWidget(api_url_label)
        
        self.api_url_edit = QLineEdit()
        self.api_url_edit.setText(current_prefs.get("api_url", "http://127.0.0.1:8000") if current_prefs else "http://127.0.0.1:8000")
        self.api_url_edit.setPlaceholderText("http://127.0.0.1:8000")
        api_url_layout.addWidget(self.api_url_edit)
        
        api_layout.addLayout(api_url_layout)
        layout.addWidget(api_group)
        
        # Appearance Settings
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)
        
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Audio Dark", "System Default"])
        current_theme = current_prefs.get("theme", "Audio Dark") if current_prefs else "Audio Dark"
        index = self.theme_combo.findText(current_theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        theme_layout.addWidget(self.theme_combo)
        
        appearance_layout.addLayout(theme_layout)
        layout.addWidget(appearance_group)
        
        # Logging Settings
        logging_group = QGroupBox("Logging")
        logging_layout = QVBoxLayout(logging_group)
        
        log_level_layout = QHBoxLayout()
        log_level_label = QLabel("Log Level:")
        log_level_layout.addWidget(log_level_label)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        current_level = current_prefs.get("log_level", "INFO") if current_prefs else "INFO"
        index = self.log_level_combo.findText(current_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        log_level_layout.addWidget(self.log_level_combo)
        
        logging_layout.addLayout(log_level_layout)
        layout.addWidget(logging_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _on_ok(self):
        """Handle OK button click."""
        prefs = {
            "api_url": self.api_url_edit.text(),
            "theme": self.theme_combo.currentText(),
            "log_level": self.log_level_combo.currentText(),
        }
        self.preferences_saved.emit(prefs)
        self.accept()
    
    def get_preferences(self) -> dict:
        """Get current preferences.
        
        Returns:
            Dictionary with preference values
        """
        return {
            "api_url": self.api_url_edit.text(),
            "theme": self.theme_combo.currentText(),
            "log_level": self.log_level_combo.currentText(),
        }
