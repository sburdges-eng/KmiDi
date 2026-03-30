"""Preset dialog UI component.

Simple list view with load/save buttons. No modals - uses dock or sidebar.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QMessageBox, QInputDialog
)
from PySide6.QtCore import Signal, Qt
from pathlib import Path
from typing import Optional

from kmidi_gui.core.preset import Preset
from kmidi_gui.core.preset_manager import PresetManager


class PresetDialog(QWidget):
    """Preset management dialog.

    Displays list of presets with load/save/delete buttons.
    Designed to be embedded in a dock or sidebar, not a modal dialog.
    """

    # Signals
    preset_loaded = Signal(Preset)  # Emitted when preset is loaded
    preset_saved = Signal(Preset)  # Emitted when preset is saved

    def __init__(self, project_path: Optional[Path] = None, parent=None):
        """Initialize preset dialog.

        Args:
            project_path: Path to project file or directory
            parent: Parent widget
        """
        super().__init__(parent)
        self.project_path = project_path
        self.preset_manager = PresetManager(project_path)
        self.current_preset: Optional[Preset] = None

        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("Presets")
        title.setObjectName("ParamTitle")
        layout.addWidget(title)

        # Preset list
        self.preset_list = QListWidget()
        self.preset_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.preset_list)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load_clicked)
        button_layout.addWidget(self.load_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        button_layout.addWidget(self.delete_btn)

        layout.addLayout(button_layout)

    def _refresh_list(self):
        """Refresh preset list from disk."""
        self.preset_list.clear()
        presets = self.preset_manager.list_presets()

        for preset in presets:
            item = QListWidgetItem(preset.name or f"Preset {preset.id[:8]}")
            item.setData(Qt.ItemDataRole.UserRole, preset.id)
            item.setToolTip(
                f"Created: {preset.timestamp.strftime('%Y-%m-%d %H:%M')}")
            self.preset_list.addItem(item)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on preset item."""
        self._load_preset_by_id(item.data(Qt.ItemDataRole.UserRole))

    def _on_save_clicked(self):
        """Handle save button click."""
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            text=self.current_preset.name if self.current_preset else ""
        )

        if not ok or not name:
            return

        # Get current state from parent (will be implemented in integration)
        # For now, create a basic preset
        preset = Preset(
            name=name,
            # TODO: Capture current state from application
            # emotion_state=current_emotion_state,
            # intent_schema=current_intent_schema,
            # ml_settings=current_ml_settings,
            # ml_visualization=current_ml_visualization,
            # teaching_mode=current_teaching_mode,
            # trust_settings=current_trust_settings,
        )

        try:
            self.preset_manager.save_preset(preset, self.project_path)
            self._refresh_list()
            self.preset_saved.emit(preset)
            QMessageBox.information(self, "Preset Saved",
                                     f"Preset '{name}' saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed",
                                  f"Failed to save preset: {e}")

    def _on_load_clicked(self):
        """Handle load button click."""
        current_item = self.preset_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection",
                                "Please select a preset to load.")
            return

        preset_id = current_item.data(Qt.ItemDataRole.UserRole)
        self._load_preset_by_id(preset_id)

    def _load_preset_by_id(self, preset_id: str):
        """Load preset by ID."""
        preset = self.preset_manager.load_preset(preset_id, self.project_path)

        if not preset:
            QMessageBox.critical(self, "Load Failed", "Failed to load preset.")
            return

        self.current_preset = preset
        self.preset_loaded.emit(preset)
        QMessageBox.information(self, "Preset Loaded", f"Preset '{preset.name}' loaded successfully.")

    def _on_delete_clicked(self):
        """Handle delete button click."""
        current_item = self.preset_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a preset to delete.")
            return

        preset_id = current_item.data(Qt.ItemDataRole.UserRole)
        preset_name = current_item.text()

        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Are you sure you want to delete '{preset_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.preset_manager.delete_preset(preset_id, self.project_path):
                self._refresh_list()
            else:
                QMessageBox.critical(self, "Delete Failed", "Failed to delete preset.")

    def set_project_path(self, project_path: Path):
        """Update project path and refresh presets.

        Args:
            project_path: New project path
        """
        self.project_path = project_path
        self.preset_manager = PresetManager(project_path)
        self._refresh_list()

    def capture_current_state(
        self,
        emotion_state=None,
        intent_schema=None,
        ml_settings=None,
        ml_visualization=None,
        teaching_mode=False,
        trust_settings=None
    ):
        """Capture current application state for preset.

        This method should be called before saving to capture the current state.
        The captured state will be used when Save is clicked.

        Args:
            emotion_state: Current EmotionState
            intent_schema: Current CompleteSongIntent
            ml_settings: Current ML enable/disable settings
            ml_visualization: Current ML visualization toggles
            teaching_mode: Current teaching mode state
            trust_settings: Current trust settings
        """
        self.current_preset = Preset(
            emotion_state=emotion_state,
            intent_schema=intent_schema,
            ml_settings=ml_settings or {},
            ml_visualization=ml_visualization or {},
            teaching_mode=teaching_mode,
            trust_settings=trust_settings,
        )
