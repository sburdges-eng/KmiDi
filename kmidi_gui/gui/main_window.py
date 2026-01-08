"""Main window for KmiDi Qt application.

This is the root GUI component. It contains only UI elements
and delegates all logic to controllers.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QStatusBar, QMenuBar, QMenu,
    QToolBar, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QFont

# Import theme
from kmidi_gui.themes.palette import apply_audio_palette, AudioColors


class MainWindow(QMainWindow):
    """Main application window.
    
    Layout:
    - Menu bar
    - Toolbar
    - Central widget (emotion input + results)
    - Status bar
    """
    
    # Signals (GUI → Controller)
    generate_requested = Signal(str)  # emotion text
    preview_requested = Signal()
    export_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KmiDi - Intelligent Digital Audio Workstation")
        self.setMinimumSize(800, 600)
        
        # Setup UI
        self._setup_menu()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()
        
        # Apply audio theme
        self._apply_theme()
    
    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Project", self._on_new_project)
        file_menu.addAction("Open Project", self._on_open_project)
        file_menu.addAction("Save Project", self._on_save_project)
        file_menu.addSeparator()
        file_menu.addAction("Export MIDI", self._on_export)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo", self._on_undo)
        edit_menu.addAction("Redo", self._on_redo)
        edit_menu.addSeparator()
        edit_menu.addAction("Preferences", self._on_preferences)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Show AI Assistant", self._on_toggle_ai_dock)
        view_menu.addAction("Show Logs", self._on_toggle_logs_dock)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("AI Analysis", self._on_ai_analysis)
        tools_menu.addAction("Batch Process", self._on_batch_process)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self._on_about)
        help_menu.addAction("Documentation", self._on_documentation)
    
    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Generate button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        toolbar.addWidget(self.generate_btn)
        
        # Preview button
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._on_preview_clicked)
        toolbar.addWidget(self.preview_btn)
        
        # Export button
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._on_export_clicked)
        toolbar.addWidget(self.export_btn)
        
        toolbar.addSeparator()
        
        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._on_preferences)
        toolbar.addWidget(settings_btn)
    
    def _setup_central_widget(self):
        """Setup central widget."""
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Emotion input label
        input_label = QLabel("What's on your heart?")
        input_label.setObjectName("ParamTitle")
        layout.addWidget(input_label)
        
        # Emotion input text area
        self.emotion_input = QTextEdit()
        self.emotion_input.setPlaceholderText(
            "Describe what you're feeling...\n"
            "Example: 'I'm feeling grief hidden as love'"
        )
        self.emotion_input.setMinimumHeight(100)
        layout.addWidget(self.emotion_input)
        
        # Results display (placeholder)
        results_label = QLabel("Results")
        results_label.setObjectName("ParamTitle")
        layout.addWidget(results_label)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Generated music will appear here...")
        layout.addWidget(self.results_display)
        
        self.setCentralWidget(central)
    
    def _setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status labels
        self.status_label = QLabel("Ready")
        self.api_status_label = QLabel("API: Checking...")
        self.file_count_label = QLabel("Files: 0")
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.api_status_label)
        self.status_bar.addPermanentWidget(self.file_count_label)
    
    def _apply_theme(self):
        """Apply audio tool theme."""
        # Load QSS
        theme_path = Path(__file__).parent.parent / "themes" / "audio_dark.qss"
        if theme_path.exists():
            with open(theme_path, "r") as f:
                self.setStyleSheet(f.read())
        
        # Apply palette (done at application level, but ensure consistency)
        app = QApplication.instance()
        if app:
            apply_audio_palette(app)
    
    # Event handlers (these emit signals, no logic)
    
    def _on_generate_clicked(self):
        """Handle generate button click."""
        text = self.emotion_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Input", "Please enter your emotional intent.")
            return
        self.generate_requested.emit(text)
    
    def _on_preview_clicked(self):
        """Handle preview button click."""
        self.preview_requested.emit()
    
    def _on_export_clicked(self):
        """Handle export button click."""
        self.export_requested.emit()
    
    # Menu handlers (placeholders - will be connected to controllers)
    
    def _on_new_project(self):
        """Handle new project menu action."""
        pass  # Will be connected to controller
    
    def _on_open_project(self):
        """Handle open project menu action."""
        pass  # Will be connected to controller
    
    def _on_save_project(self):
        """Handle save project menu action."""
        pass  # Will be connected to controller
    
    def _on_export(self):
        """Handle export menu action."""
        self.export_requested.emit()
    
    def _on_undo(self):
        """Handle undo menu action."""
        pass  # Will be connected to controller
    
    def _on_redo(self):
        """Handle redo menu action."""
        pass  # Will be connected to controller
    
    def _on_preferences(self):
        """Handle preferences menu action."""
        pass  # Will be connected to controller
    
    def _on_toggle_ai_dock(self):
        """Handle toggle AI dock menu action."""
        pass  # Will be connected to controller
    
    def _on_toggle_logs_dock(self):
        """Handle toggle logs dock menu action."""
        pass  # Will be connected to controller
    
    def _on_ai_analysis(self):
        """Handle AI analysis menu action."""
        pass  # Will be connected to controller
    
    def _on_batch_process(self):
        """Handle batch process menu action."""
        pass  # Will be connected to controller
    
    def _on_about(self):
        """Handle about menu action."""
        QMessageBox.about(
            self,
            "About KmiDi",
            "KmiDi - Intelligent Digital Audio Workstation\n\n"
            "Version 1.0.0\n"
            "Professional music generation tool"
        )
    
    def _on_documentation(self):
        """Handle documentation menu action."""
        pass  # Will open documentation
    
    # Public methods for controller to update UI
    
    def set_status(self, message: str):
        """Update status bar message."""
        self.status_label.setText(message)
    
    def set_api_status(self, status: str, online: bool = True):
        """Update API status indicator."""
        color = AudioColors.STATUS_OK if online else AudioColors.STATUS_ERROR
        self.api_status_label.setText(f'<span style="color: {color}">API: {status}</span>')
    
    def set_file_count(self, count: int):
        """Update file count."""
        self.file_count_label.setText(f"Files: {count}")
    
    def set_results(self, text: str):
        """Update results display."""
        self.results_display.setPlainText(text)
    
    def set_generating(self, generating: bool):
        """Update generate button state."""
        self.generate_btn.setEnabled(not generating)
        self.generate_btn.setText("Generating..." if generating else "Generate")

