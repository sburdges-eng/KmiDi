"""Export pipeline for MIDI, Intent Schema, and ML annotations.

Export is asynchronous and non-blocking. Uses QThread for background operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from PySide6.QtCore import QThread, Signal, QObject

from music_brain.session.intent_schema import CompleteSongIntent

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """Background worker for export operations."""

    finished = Signal(bool, str)  # success, message
    progress = Signal(str)  # progress message

    def __init__(self, exporter, path: Path, data: Any):
        """Initialize export worker.

        Args:
            exporter: Exporter instance
            path: Export path
            data: Data to export
        """
        super().__init__()
        self.exporter = exporter
        self.path = path
        self.data = data

    def run(self):
        """Run export in background thread."""
        try:
            self.progress.emit(f"Exporting to {self.path.name}...")
            self.exporter.export(self.path, self.data)
            self.finished.emit(True, f"Exported to {self.path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.finished.emit(False, f"Export failed: {e}")


class MIDIExporter:
    """Exports MIDI files."""

    def export(self, path: Path, midi_data: Any) -> None:
        """Export MIDI data to file.

        Args:
            path: Output file path
            midi_data: MIDI data (format depends on source)

        Raises:
            IOError: If file cannot be written
        """
        # TODO: Implement actual MIDI export
        # For now, create a placeholder file
        if isinstance(midi_data, dict) and "file_path" in midi_data:
            # If midi_data contains a file path, copy it
            from shutil import copy2
            source_path = Path(midi_data["file_path"])
            if source_path.exists():
                copy2(source_path, path)
                logger.info(f"Exported MIDI: {path}")
                return

        # Placeholder: create empty file
        path.write_bytes(b"")
        logger.warning(
            f"MIDI export not fully implemented, created placeholder: {path}")


class IntentExporter:
    """Exports Intent Schema JSON."""

    def export(self, path: Path, intent: CompleteSongIntent) -> None:
        """Export intent schema to JSON file.

        Args:
            path: Output file path
            intent: CompleteSongIntent to export

        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(intent.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Exported intent schema: {path}")
        except IOError as e:
            logger.error(f"Failed to export intent schema: {e}")
            raise


class AnnotationExporter:
    """Exports ML annotation metadata."""

    def export(self, path: Path, annotations: Dict[str, Any]) -> None:
        """Export ML annotations to JSON file.

        Args:
            path: Output file path
            annotations: Annotation data dictionary

        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported annotations: {path}")
        except IOError as e:
            logger.error(f"Failed to export annotations: {e}")
            raise


class ExportManager(QObject):
    """Manages async export operations.

    All exports run in background threads to keep UI responsive.
    """

    export_finished = Signal(bool, str)  # success, message
    export_progress = Signal(str)  # progress message

    def __init__(self):
        """Initialize export manager."""
        super().__init__()
        self.midi_exporter = MIDIExporter()
        self.intent_exporter = IntentExporter()
        self.annotation_exporter = AnnotationExporter()
        # Note: Multiple workers can run concurrently
        # The dialog tracks all active workers, not this manager

    def export_midi(self, path: Path, midi_data: Any) -> QThread:
        """Export MIDI file asynchronously.

        Args:
            path: Output path
            midi_data: MIDI data to export

        Returns:
            QThread worker (can be used to track progress)
        """
        worker = ExportWorker(self.midi_exporter, path, midi_data)
        worker.finished.connect(self._on_export_finished)
        worker.progress.connect(self._on_export_progress)
        worker.start()
        return worker

    def export_intent(self, path: Path, intent: CompleteSongIntent) -> QThread:
        """Export Intent Schema JSON asynchronously.

        Args:
            path: Output file path
            intent: CompleteSongIntent to export

        Returns:
            QThread worker
        """
        worker = ExportWorker(self.intent_exporter, path, intent)
        worker.finished.connect(self._on_export_finished)
        worker.progress.connect(self._on_export_progress)
        worker.start()
        return worker

    def export_annotations(self, path: Path, annotations: Dict[str, Any]) -> QThread:
        """Export ML annotations asynchronously.

        Args:
            path: Output file path
            annotations: Annotation data

        Returns:
            QThread worker
        """
        worker = ExportWorker(self.annotation_exporter, path, annotations)
        worker.finished.connect(self._on_export_finished)
        worker.progress.connect(self._on_export_progress)
        worker.start()
        return worker

    def _on_export_finished(self, success: bool, message: str):
        """Handle export completion.

        Args:
            success: Whether export succeeded
            message: Completion message
        """
        # Emit signal - dialog will track multiple concurrent exports
        self.export_finished.emit(success, message)

    def _on_export_progress(self, message: str):
        """Handle export progress update.

        Args:
            message: Progress message
        """
        self.export_progress.emit(message)
