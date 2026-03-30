"""Export dialog for MIDI, Intent Schema, and ML annotations.

Uses native macOS save dialog. Non-blocking with progress indicator.
"""

from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QProgressBar, QMessageBox, QFileDialog
)
from PySide6.QtCore import Signal

from kmidi_gui.core.export import ExportManager
from music_brain.session.intent_schema import CompleteSongIntent


class ExportDialog(QDialog):
    """Export dialog with format selection and progress indicator."""

    export_complete = Signal(bool, str)  # success, message

    def __init__(self, parent=None):
        """Initialize export dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.export_manager = ExportManager()
        self.export_manager.export_finished.connect(self._on_export_finished)
        self.export_manager.export_progress.connect(self._on_export_progress)

        self.current_intent: Optional[CompleteSongIntent] = None
        self.current_midi_data: Optional[dict] = None
        self.current_annotations: Optional[dict] = None

        # Track active export workers
        self.active_workers = []  # List of QThread workers
        self.export_results = []  # List of (success, message) tuples

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        self.setWindowTitle("Export")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Format selection
        format_label = QLabel("Export Formats:")
        layout.addWidget(format_label)

        self.midi_checkbox = QCheckBox("MIDI File")
        self.midi_checkbox.setChecked(True)
        layout.addWidget(self.midi_checkbox)

        self.intent_checkbox = QCheckBox("Intent Schema JSON")
        self.intent_checkbox.setChecked(True)
        layout.addWidget(self.intent_checkbox)

        self.annotations_checkbox = QCheckBox("ML Annotations JSON")
        self.annotations_checkbox.setChecked(False)
        layout.addWidget(self.annotations_checkbox)

        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(self.export_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def set_data(
        self,
        intent: Optional[CompleteSongIntent] = None,
        midi_data: Optional[dict] = None,
        annotations: Optional[dict] = None
    ):
        """Set data to export.

        Args:
            intent: Intent schema to export
            midi_data: MIDI data to export
            annotations: ML annotations to export
        """
        self.current_intent = intent
        self.current_midi_data = midi_data
        self.current_annotations = annotations

    def _on_export_clicked(self):
        """Handle export button click."""
        # Check if at least one format is selected
        if not any([
            self.midi_checkbox.isChecked(),
            self.intent_checkbox.isChecked(),
            self.annotations_checkbox.isChecked()
        ]):
            QMessageBox.warning(self, "No Format Selected",
                                "Please select at least one export format.")
            return

        # Get base directory from save dialog
        base_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", str(Path.home() / "Desktop")
        )

        if not base_dir:
            return

        base_path = Path(base_dir)

        # Reset tracking for new export session
        self.active_workers.clear()
        self.export_results.clear()

        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.export_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        # Export selected formats
        exports_started = 0

        if self.midi_checkbox.isChecked() and self.current_midi_data:
            midi_path = base_path / "export.mid"
            worker = self.export_manager.export_midi(midi_path, self.current_midi_data)
            self.active_workers.append(worker)
            exports_started += 1

        if self.intent_checkbox.isChecked() and self.current_intent:
            intent_path = base_path / "intent_schema.json"
            worker = self.export_manager.export_intent(intent_path, self.current_intent)
            self.active_workers.append(worker)
            exports_started += 1

        if self.annotations_checkbox.isChecked() and self.current_annotations:
            annotations_path = base_path / "ml_annotations.json"
            worker = self.export_manager.export_annotations(annotations_path, self.current_annotations)
            self.active_workers.append(worker)
            exports_started += 1

        if exports_started == 0:
            QMessageBox.warning(self, "No Data",
                                "No data available for selected formats.")
            self.progress_bar.setVisible(False)
            self.status_label.setVisible(False)
            self.export_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)

    def _on_export_progress(self, message: str):
        """Handle export progress update.

        Args:
            message: Progress message
        """
        self.status_label.setText(message)

    def _on_export_finished(self, success: bool, message: str):
        """Handle export completion for a single export.

        Tracks completion of individual exports and only shows final
        completion dialog when all exports have finished.

        Args:
            success: Whether this export succeeded
            message: Completion message for this export
        """
        # Record this export's result
        self.export_results.append((success, message))

        # Check if all exports have completed
        if len(self.export_results) >= len(self.active_workers):
            # All exports finished - show final result
            all_success = all(result[0] for result in self.export_results)
            failed_count = sum(1 for result in self.export_results if not result[0])

            # Hide progress indicators
            self.progress_bar.setVisible(False)
            self.status_label.setVisible(False)
            self.export_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)

            # Prepare final message
            if all_success:
                total = len(self.export_results)
                final_message = f"Successfully exported {total} file(s)."
                QMessageBox.information(self, "Export Complete", final_message)
                self.export_complete.emit(True, final_message)
                self.accept()
            else:
                total = len(self.export_results)
                final_message = (
                    f"Export completed with {failed_count} failure(s) "
                    f"out of {total} file(s)."
                )
                QMessageBox.warning(self, "Export Partially Failed", final_message)
                self.export_complete.emit(False, final_message)
