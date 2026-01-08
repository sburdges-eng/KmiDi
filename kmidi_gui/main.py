"""Main entry point for KmiDi Qt application."""

import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from kmidi_gui.gui.main_window import MainWindow
from kmidi_gui.controllers.actions import ActionController
from kmidi_gui.themes.palette import apply_audio_palette

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("KmiDi")
    app.setOrganizationName("KmiDi")
    
    # Apply audio tool theme
    app.setStyle("Fusion")  # Use Fusion style (neutral base)
    apply_audio_palette(app)
    
    # Load QSS stylesheet
    theme_path = Path(__file__).parent / "themes" / "audio_dark.qss"
    if theme_path.exists():
        with open(theme_path, "r") as f:
            app.setStyleSheet(f.read())
        logger.info("Audio theme loaded")
    else:
        logger.warning(f"Theme file not found: {theme_path}")
    
    # Create main window
    window = MainWindow()
    
    # Create controller (connects GUI to core logic)
    controller = ActionController(window)
    
    # Connect controller signals to GUI updates
    controller.status_changed.connect(window.set_status)
    controller.results_ready.connect(window.set_results)
    controller.generation_started.connect(lambda: window.set_generating(True))
    controller.generation_finished.connect(lambda success: window.set_generating(False))
    controller.error_occurred.connect(lambda msg: window.set_status(f"Error: {msg}"))
    
    # Show window
    window.show()
    
    # Set initial status
    window.set_api_status("Checking...", online=None)
    # TODO: Check API status in background
    
    logger.info("KmiDi application started")
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

