"""Action controllers - bridge GUI events to core logic.

This layer:
- Receives GUI signals
- Calls core logic functions
- Updates GUI via signals/slots
- Manages worker threads for long operations
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal, QThread, Qt

from kmidi_gui.core.engine import get_engine, set_engine, MusicEngine
from kmidi_gui.core.models import EmotionIntent, GenerationResult
from kmidi_gui.core.history import HistoryManager, HistoryComponent
from kmidi_gui.gui.preferences_dialog import PreferencesDialog

logger = logging.getLogger(__name__)


class GenerationWorker(QThread):
    """Background worker for music generation.

    Runs generation in a separate thread to keep UI responsive.
    """

    finished = Signal(GenerationResult)
    error = Signal(str)

    def __init__(self, intent: EmotionIntent):
        super().__init__()
        self.intent = intent
        self.engine = get_engine()

    def run(self):
        """Run generation in background thread."""
        try:
            result = self.engine.generate_music(self.intent)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Generation worker error: {e}")
            self.error.emit(str(e))


class ActionController(QObject):
    """Main action controller.

    Connects GUI signals to core logic and updates GUI with results.
    """

    # Signals (Controller → GUI)
    status_changed = Signal(str)
    results_ready = Signal(str)
    generation_started = Signal()
    generation_finished = Signal(bool)  # success
    error_occurred = Signal(str)
    log_message = Signal(str, str)  # message, level
    project_loaded = Signal(dict)  # project data
    project_saved = Signal(str)  # file path

    def __init__(self, main_window):
        """Initialize controller.

        Args:
            main_window: MainWindow instance to control
        """
        super().__init__()
        self.main_window = main_window
        self.engine = get_engine()
        self.current_worker: Optional[GenerationWorker] = None
        self.history_manager = HistoryManager()
        self.preferences: Dict[str, Any] = {
            "api_url": "http://127.0.0.1:8000",
            "theme": "Audio Dark",
            "log_level": "INFO",
        }

        # Connect GUI signals to controller methods
        main_window.generate_requested.connect(self.handle_generate)
        main_window.preview_requested.connect(self.handle_preview)
        main_window.export_requested.connect(self.handle_export)
        main_window.new_project_requested.connect(self.handle_new_project)
        main_window.open_project_requested.connect(self.handle_open_project)
        main_window.save_project_requested.connect(self.handle_save_project)
        main_window.preferences_requested.connect(self.handle_preferences)
        main_window.ai_analysis_requested.connect(self.handle_ai_analysis)
        main_window.batch_process_requested.connect(self.handle_batch_process)
        main_window.presets_requested.connect(self.handle_presets)
        main_window.undo_requested.connect(self.handle_undo)
        main_window.redo_requested.connect(self.handle_redo)

        # Connect controller signals to GUI
        self.log_message.connect(self._on_log_message)

        # Update undo/redo menu states periodically
        self._update_history_menu_states()

        # Initial status check
        self._check_api_status()

    def handle_generate(self, emotion_text: str, emotion_params: dict, technical_params: dict):
        """Handle generate request from GUI.

        Args:
            emotion_text: User's emotional intent text
            emotion_params: Emotion parameters (valence, arousal, intensity)
            technical_params: Technical parameters (key, bpm, genre)
        """
        logger.info(f"Generate requested: {emotion_text[:50]}...")
        self.log_message.emit("Starting music generation...", "INFO")

        # Create intent from text and parameters
        intent = EmotionIntent(
            core_event=emotion_text,
            mood_primary="grief" if "grief" in emotion_text.lower() else None,
            technical_key=technical_params.get("key"),
            technical_bpm=technical_params.get("bpm"),
            technical_genre=technical_params.get("genre"),
        )

        # Start background worker
        self.generation_started.emit()
        self.status_changed.emit("Generating music...")

        self.current_worker = GenerationWorker(intent)
        self.current_worker.finished.connect(self._on_generation_finished)
        self.current_worker.error.connect(self._on_generation_error)
        self.current_worker.start()

    def handle_preview(self):
        """Handle preview request from GUI."""
        logger.info("Preview requested")

        # Check if we have a recent generation result
        if not hasattr(self, 'last_result') or not self.last_result:
            self.error_occurred.emit(
                "No music generated yet. Generate music first.")
            return

        try:
            from pathlib import Path
            import platform

            # Get MIDI file path
            midi_path = self.last_result.midi_path
            if not midi_path or not Path(midi_path).exists():
                # Try to use system default MIDI player
                if platform.system() == "Darwin":  # macOS
                    # Use afplay or open with default app
                    self.status_changed.emit(
                        "Preview: Playing with default MIDI player")
                    # Could launch external player here
                elif platform.system() == "Linux":
                    self.status_changed.emit(
                        "Preview: Use timidity or fluidsynth")
                else:  # Windows
                    self.status_changed.emit("Preview: Use Windows Media Player")

                self.status_changed.emit(
                    "Preview: MIDI file not available for playback")
            else:
                # Preview available
                self.status_changed.emit(
                    f"Preview ready: {Path(midi_path).name}")
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            self.error_occurred.emit(f"Preview error: {str(e)}")

    def handle_export(self):
        """Handle export request from GUI."""
        from kmidi_gui.gui.export_dialog import ExportDialog

        logger.info("Export requested")

        # Check if we have a recent generation result
        if not hasattr(self, 'last_result') or not self.last_result:
            self.error_occurred.emit(
                "No music generated yet. Generate music first.")
            return

        # Create export dialog
        dialog = ExportDialog(self.main_window)

        # Set export data
        midi_data = None
        if self.last_result.midi_path:
            midi_data = {"file_path": self.last_result.midi_path}

        # TODO: Get current intent schema from application state
        intent_schema = None

        # TODO: Get ML annotations from application state
        annotations = None

        dialog.set_data(
            intent=intent_schema,
            midi_data=midi_data,
            annotations=annotations
        )

        dialog.export_complete.connect(self._on_export_complete)
        dialog.exec()

    def _on_export_complete(self, success: bool, message: str):
        """Handle export completion.

        Args:
            success: Whether export succeeded
            message: Completion message
        """
        if success:
            self.status_changed.emit(message)
            self.log_message.emit(message, "INFO")
        else:
            self.error_occurred.emit(message)
            self.log_message.emit(message, "ERROR")

    def _on_generation_finished(self, result: GenerationResult):
        """Handle generation completion.

        Args:
            result: Generation result
        """
        self.current_worker = None
        self.last_result = result  # Store for preview/export

        if result.success:
            # Format results for display
            results_text = "Generation successful!\n\n"
            if result.chords:
                results_text += f"Chords: {' - '.join(result.chords)}\n"
            if result.key:
                results_text += f"Key: {result.key}\n"
            if result.tempo:
                results_text += f"Tempo: {result.tempo} BPM\n"
            if result.midi_path:
                results_text += f"MIDI: {result.midi_path}\n"

            self.results_ready.emit(results_text)
            self.status_changed.emit("Generation complete")
            self.generation_finished.emit(True)
        else:
            error_msg = result.error or "Unknown error"
            self.error_occurred.emit(f"Generation failed: {error_msg}")
            self.status_changed.emit("Generation failed")
            self.generation_finished.emit(False)

    def _on_generation_error(self, error_msg: str):
        """Handle generation error.

        Args:
            error_msg: Error message
        """
        self.current_worker = None
        self.error_occurred.emit(f"Error: {error_msg}")
        self.status_changed.emit("Error occurred")
        self.generation_finished.emit(False)
        self.log_message.emit(f"Generation failed: {error_msg}", "ERROR")

    def handle_new_project(self):
        """Handle new project request."""
        self.log_message.emit("New project created", "INFO")
        self.status_changed.emit("Ready")

    def handle_open_project(self, file_path: str):
        """Handle open project request.

        Args:
            file_path: Path to project file
        """
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)

            # Load project data into UI
            if "emotion_text" in project_data:
                self.main_window.emotion_input.setPlainText(
                    project_data["emotion_text"])

            if "emotion_params" in project_data:
                params = project_data["emotion_params"]
                self.main_window.emotion_params.valence_slider.setValue(
                    int(params.get("valence", 0) * 100))
                self.main_window.emotion_params.arousal_slider.setValue(
                    int(params.get("arousal", 0.5) * 100))
                self.main_window.emotion_params.intensity_slider.setValue(
                    int(params.get("intensity", 0.5) * 100))

            if "technical_params" in project_data:
                params = project_data["technical_params"]
                if params.get("key"):
                    index = self.main_window.technical_params.key_combo.findText(
                        params["key"])
                    if index >= 0:
                        self.main_window.technical_params.key_combo.setCurrentIndex(index)
                if params.get("bpm"):
                    self.main_window.technical_params.bpm_spinbox.setValue(
                        params["bpm"])
                if params.get("genre"):
                    index = self.main_window.technical_params.genre_combo.findText(
                        params["genre"])
                    if index >= 0:
                        self.main_window.technical_params.genre_combo.setCurrentIndex(index)

            self.main_window.set_project_path(Path(file_path))
            self.log_message.emit(
                f"Project loaded: {Path(file_path).name}", "INFO")
            self.status_changed.emit("Project loaded")
        except Exception as e:
            logger.error(f"Failed to open project: {e}")
            self.error_occurred.emit(f"Failed to open project: {str(e)}")
            self.log_message.emit(f"Failed to open project: {str(e)}", "ERROR")

    def handle_save_project(self, file_path: str):
        """Handle save project request.

        Args:
            file_path: Path to save project file
        """
        try:
            # Collect project data
            project_data = {
                "version": "1.0.0",
                "emotion_text": self.main_window.emotion_input.toPlainText(),
                "emotion_params": self.main_window.emotion_params.get_values(),
                "technical_params": self.main_window.technical_params.get_values(),
                "results": (self.main_window.results_display.toPlainText()
                            if hasattr(self, 'last_result') else ""),
            }

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)

            self.main_window.set_project_path(Path(file_path))
            self.log_message.emit(
                f"Project saved: {Path(file_path).name}", "INFO")
            self.status_changed.emit("Project saved")
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            self.error_occurred.emit(f"Failed to save project: {str(e)}")
            self.log_message.emit(f"Failed to save project: {str(e)}", "ERROR")

    def handle_preferences(self):
        """Handle preferences request."""
        dialog = PreferencesDialog(self.main_window, self.preferences)
        dialog.preferences_saved.connect(self._on_preferences_saved)
        dialog.exec()

    def _on_preferences_saved(self, prefs: dict):
        """Handle preferences saved.

        Args:
            prefs: New preferences dictionary
        """
        self.preferences = prefs

        # Update engine API URL if changed
        if prefs.get("api_url") != self.engine.api_url:
            self.engine = MusicEngine(music_brain_api_url=prefs["api_url"])
            set_engine(self.engine)
            self.log_message.emit(f"API URL updated: {prefs['api_url']}", "INFO")

        # Update logging level
        import logging
        log_level = getattr(logging, prefs.get("log_level", "INFO"), logging.INFO)
        logging.getLogger().setLevel(log_level)

        self.log_message.emit("Preferences saved", "INFO")
        self._check_api_status()

    def handle_ai_analysis(self):
        """Handle AI analysis request."""
        self.log_message.emit(
            "AI analysis requested (not yet implemented)", "INFO")
        # TODO: Implement AI analysis
        self.status_changed.emit("AI analysis: Coming soon")

    def handle_batch_process(self):
        """Handle batch process request."""
        self.log_message.emit(
            "Batch process requested (not yet implemented)", "INFO")
        # TODO: Implement batch processing
        self.status_changed.emit("Batch process: Coming soon")

    def handle_presets(self):
        """Handle presets request."""
        from kmidi_gui.gui.preset_dialog import PresetDialog
        from PySide6.QtWidgets import QDockWidget

        # Create or show preset dock
        preset_dock = None
        for dock in self.main_window.findChildren(QDockWidget):
            if dock.windowTitle() == "Presets":
                preset_dock = dock
                break

        if not preset_dock:
            preset_dock = QDockWidget("Presets", self.main_window)
            preset_dialog = PresetDialog(
                project_path=self.main_window.current_project_path,
                parent=preset_dock
            )
            preset_dialog.preset_loaded.connect(self._on_preset_loaded)
            preset_dialog.preset_saved.connect(self._on_preset_saved)
            preset_dock.setWidget(preset_dialog)
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, preset_dock)

        preset_dock.setVisible(True)
        preset_dock.raise_()

    def _on_preset_loaded(self, preset):
        """Handle preset loaded signal.

        Args:
            preset: Loaded preset
        """
        logger.info(f"Preset loaded: {preset.name}")
        # TODO: Apply preset atomically
        # 1. Snapshot current state
        # 2. Apply preset state
        # 3. Update UI
        self.log_message.emit(f"Preset '{preset.name}' loaded", "INFO")
        self.status_changed.emit(f"Preset loaded: {preset.name}")

    def _on_preset_saved(self, preset):
        """Handle preset saved signal.

        Args:
            preset: Saved preset
        """
        logger.info(f"Preset saved: {preset.name}")
        self.log_message.emit(f"Preset '{preset.name}' saved", "INFO")
        self.status_changed.emit(f"Preset saved: {preset.name}")

    def handle_undo(self, component: str = HistoryComponent.INTENT.value):
        """Handle undo request.

        Args:
            component: Component to undo (defaults to intent)
        """
        action = self.history_manager.undo(component)
        if action:
            self.log_message.emit(f"Undid: {action.description}", "INFO")
            self._update_history_menu_states()
        else:
            self.log_message.emit("Nothing to undo", "INFO")

    def handle_redo(self, component: str = HistoryComponent.INTENT.value):
        """Handle redo request.

        Args:
            component: Component to redo (defaults to intent)
        """
        action = self.history_manager.redo(component)
        if action:
            self.log_message.emit(f"Redid: {action.description}", "INFO")
            self._update_history_menu_states()
        else:
            self.log_message.emit("Nothing to redo", "INFO")

    def _update_history_menu_states(self):
        """Update undo/redo menu item enabled states."""
        if (not hasattr(self.main_window, "undo_action") or
                not hasattr(self.main_window, "redo_action")):
            return

        component = HistoryComponent.INTENT.value
        can_undo = self.history_manager.can_undo(component)
        can_redo = self.history_manager.can_redo(component)
        self.main_window.undo_action.setEnabled(can_undo)
        self.main_window.redo_action.setEnabled(can_redo)

    def _check_api_status(self):
        """Check API connection status."""
        try:
            import requests
            api_url = self.preferences.get("api_url", "http://127.0.0.1:8000")
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                self.main_window.set_api_status("Online", True)
                self.log_message.emit("API connection: Online", "INFO")
            else:
                self.main_window.set_api_status("Offline", False)
                self.log_message.emit("API connection: Offline", "WARNING")
        except Exception as e:
            self.main_window.set_api_status("Offline", False)
            self.log_message.emit(f"API connection: {str(e)}", "WARNING")

    def _on_log_message(self, message: str, level: str):
        """Handle log message from controller.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        logs_dock = self.main_window.get_logs_dock()
        if logs_dock:
            logs_dock.add_log(message, level)

