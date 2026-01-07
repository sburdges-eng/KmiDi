"""Action controllers - bridge GUI events to core logic.

This layer:
- Receives GUI signals
- Calls core logic functions
- Updates GUI via signals/slots
- Manages worker threads for long operations
"""

import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal, QThread

from kmidi_gui.core.engine import get_engine
from kmidi_gui.core.models import EmotionIntent, GenerationResult

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
    
    def __init__(self, main_window):
        """Initialize controller.
        
        Args:
            main_window: MainWindow instance to control
        """
        super().__init__()
        self.main_window = main_window
        self.engine = get_engine()
        self.current_worker: Optional[GenerationWorker] = None
        
        # Connect GUI signals to controller methods
        main_window.generate_requested.connect(self.handle_generate)
        main_window.preview_requested.connect(self.handle_preview)
        main_window.export_requested.connect(self.handle_export)
    
    def handle_generate(self, emotion_text: str):
        """Handle generate request from GUI.
        
        Args:
            emotion_text: User's emotional intent text
        """
        logger.info(f"Generate requested: {emotion_text[:50]}...")
        
        # Create intent from text (simplified - would parse more in real implementation)
        intent = EmotionIntent(
            core_event=emotion_text,
            mood_primary="grief" if "grief" in emotion_text.lower() else None,
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
            self.error_occurred.emit("No music generated yet. Generate music first.")
            return
        
        try:
            from pathlib import Path
            import subprocess
            import platform
            
            # Get MIDI file path
            midi_path = self.last_result.midi_path
            if not midi_path or not Path(midi_path).exists():
                # Try to use system default MIDI player
                if platform.system() == "Darwin":  # macOS
                    # Use afplay or open with default app
                    self.status_changed.emit("Preview: Playing with default MIDI player")
                    # Could launch external player here
                elif platform.system() == "Linux":
                    self.status_changed.emit("Preview: Use timidity or fluidsynth")
                else:  # Windows
                    self.status_changed.emit("Preview: Use Windows Media Player")
                
                self.status_changed.emit("Preview: MIDI file not available for playback")
            else:
                # Preview available
                self.status_changed.emit(f"Preview ready: {Path(midi_path).name}")
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            self.error_occurred.emit(f"Preview error: {str(e)}")
    
    def handle_export(self):
        """Handle export request from GUI."""
        logger.info("Export requested")
        
        # Check if we have a recent generation result
        if not hasattr(self, 'last_result') or not self.last_result:
            self.error_occurred.emit("No music generated yet. Generate music first.")
            return
        
        try:
            from pathlib import Path
            from datetime import datetime
            
            # Get export directory
            export_dir = Path.home() / "Desktop" / "KmiDi_Exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"kmidi_export_{timestamp}"
            
            # Export MIDI if available
            if self.last_result.midi_path and Path(self.last_result.midi_path).exists():
                import shutil
                midi_export_path = export_dir / f"{base_name}.mid"
                shutil.copy2(self.last_result.midi_path, midi_export_path)
                self.status_changed.emit(f"Exported MIDI: {midi_export_path.name}")
            
            # Export metadata as JSON
            import json
            metadata_export_path = export_dir / f"{base_name}_metadata.json"
            with open(metadata_export_path, 'w') as f:
                json.dump(self.last_result.to_dict(), f, indent=2)
            
            self.status_changed.emit(f"Exported to: {export_dir}")
            self.results_ready.emit(f"Export complete!\n\nLocation: {export_dir}\nFiles: {base_name}.mid, {base_name}_metadata.json")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.error_occurred.emit(f"Export error: {str(e)}")
    
    def _on_generation_finished(self, result: GenerationResult):
        """Handle generation completion.
        
        Args:
            result: Generation result
        """
        self.current_worker = None
        self.last_result = result  # Store for preview/export
        
        if result.success:
            # Format results for display
            results_text = f"Generation successful!\n\n"
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

