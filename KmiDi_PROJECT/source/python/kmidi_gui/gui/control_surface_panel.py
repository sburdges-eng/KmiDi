"""Control surface panel UI.

Device list, mapping table, and learn mode interface.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from pathlib import Path
from typing import Optional

from kmidi_gui.core.control_surface import ControlSurfaceManager, ControlMapping


class ControlSurfacePanel(QWidget):
    """Control surface configuration panel.

    Shows MIDI devices, parameter mappings, and learn mode controls.
    """

    # Signals
    mapping_added = Signal(ControlMapping)
    mapping_removed = Signal(str)  # parameter name

    def __init__(self, project_path: Optional[Path] = None, parent=None):
        """Initialize control surface panel.

        Args:
            project_path: Path to project file or directory
            parent: Parent widget
        """
        super().__init__(parent)
        self.project_path = project_path
        self.control_manager = ControlSurfaceManager()

        # Load existing mappings if project path provided
        if project_path:
            self.control_manager.load_mappings(project_path)

        self._setup_ui()
        self._refresh_devices()
        self._refresh_mappings()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("Control Surface")
        title.setObjectName("ParamTitle")
        layout.addWidget(title)

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("MIDI Device:"))

        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_layout.addWidget(self.device_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_devices)
        device_layout.addWidget(refresh_btn)

        layout.addLayout(device_layout)

        # Mappings table
        mappings_label = QLabel("Parameter Mappings:")
        layout.addWidget(mappings_label)

        self.mappings_table = QTableWidget()
        self.mappings_table.setColumnCount(4)
        self.mappings_table.setHorizontalHeaderLabels(["Parameter", "Type", "Control", "Range"])
        self.mappings_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.mappings_table)

        # Learn mode
        learn_layout = QHBoxLayout()

        self.learn_btn = QPushButton("Learn")
        self.learn_btn.setCheckable(True)
        self.learn_btn.clicked.connect(self._on_learn_clicked)
        learn_layout.addWidget(self.learn_btn)

        self.learn_label = QLabel("Click Learn, then move a MIDI control")
        self.learn_label.setVisible(False)
        learn_layout.addWidget(self.learn_label)

        learn_layout.addStretch()
        layout.addLayout(learn_layout)

        # Buttons
        button_layout = QHBoxLayout()

        remove_btn = QPushButton("Remove Mapping")
        remove_btn.clicked.connect(self._on_remove_clicked)
        button_layout.addWidget(remove_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _refresh_devices(self):
        """Refresh MIDI device list."""
        self.device_combo.clear()
        devices = self.control_manager.discover_devices()

        if not devices:
            self.device_combo.addItem("No MIDI devices found")
            return

        for device in devices:
            self.device_combo.addItem(device["name"], device["id"])

    def _refresh_mappings(self):
        """Refresh mappings table."""
        self.mappings_table.setRowCount(len(self.control_manager.mappings))

        for row, mapping in enumerate(self.control_manager.mappings):
            self.mappings_table.setItem(row, 0, QTableWidgetItem(mapping.parameter))

            type_str = f"{mapping.midi_type.upper()}"
            self.mappings_table.setItem(row, 1, QTableWidgetItem(type_str))

            control_str = ""
            if mapping.midi_type == "cc":
                control_str = f"CC {mapping.cc_number}"
            else:
                control_str = f"Note {mapping.note_number}"
            self.mappings_table.setItem(row, 2, QTableWidgetItem(control_str))

            range_str = f"{mapping.min_value:.2f} - {mapping.max_value:.2f}"
            self.mappings_table.setItem(row, 3, QTableWidgetItem(range_str))

        self.mappings_table.resizeColumnsToContents()

    def _on_device_changed(self, index: int):
        """Handle device selection change.

        Args:
            index: Selected device index
        """
        if index < 0:
            return

        device_id = self.device_combo.itemData(index)
        if device_id:
            self.control_manager.connect_device(str(device_id))

    def _on_learn_clicked(self, checked: bool):
        """Handle learn button click.

        Args:
            checked: Whether learn mode is active
        """
        if checked:
            # Get parameter from user
            from PySide6.QtWidgets import QInputDialog
            parameter, ok = QInputDialog.getText(
                self, "Learn Parameter", "Parameter to map:",
                text="intent.vulnerability_scale"
            )

            if ok and parameter:
                self.control_manager.start_learn_mode(parameter, self._on_mapping_learned)
                self.learn_label.setVisible(True)
                self.learn_label.setText(f"Learning: {parameter} - Move a MIDI control now")
            else:
                self.learn_btn.setChecked(False)
        else:
            self.control_manager.stop_learn_mode()
            self.learn_label.setVisible(False)

    def _on_mapping_learned(self, mapping: ControlMapping):
        """Handle mapping learned callback.

        Args:
            mapping: Learned mapping
        """
        self.learn_btn.setChecked(False)
        self.learn_label.setVisible(False)
        self._refresh_mappings()
        self.mapping_added.emit(mapping)

        # Save mappings
        if self.project_path:
            self.control_manager.save_mappings(self.project_path)

        QMessageBox.information(self, "Mapping Learned", f"Mapped {mapping.parameter} to {mapping.midi_type.upper()} {mapping.cc_number or mapping.note_number}")

    def _on_remove_clicked(self):
        """Handle remove mapping button click."""
        current_row = self.mappings_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a mapping to remove.")
            return

        parameter_item = self.mappings_table.item(current_row, 0)
        if not parameter_item:
            return

        parameter = parameter_item.text()

        reply = QMessageBox.question(
            self, "Remove Mapping",
            f"Remove mapping for '{parameter}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.control_manager.remove_mapping(parameter):
                self._refresh_mappings()
                self.mapping_removed.emit(parameter)

                # Save mappings
                if self.project_path:
                    self.control_manager.save_mappings(self.project_path)
            else:
                QMessageBox.critical(self, "Remove Failed", "Failed to remove mapping.")

    def set_project_path(self, project_path: Path):
        """Update project path and reload mappings.

        Args:
            project_path: New project path
        """
        self.project_path = project_path
        self.control_manager.load_mappings(project_path)
        self._refresh_mappings()
