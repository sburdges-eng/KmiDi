"""Main window for KmiDi Qt application.

This is the root GUI component. It contains only UI elements
and delegates all logic to controllers.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QStatusBar,
    QToolBar,
    QMessageBox,
    QFileDialog,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal

# Import theme
from kmidi_gui.themes.palette import apply_audio_palette, AudioColors
from kmidi_gui.gui.docks import AIAssistantDock, LogsDock
from kmidi_gui.gui.parameter_panel import EmotionParameterPanel, TechnicalParameterPanel
from kmidi_gui.gui.control_surface_panel import ControlSurfacePanel


class MainWindow(QMainWindow):
    """Main application window.

    Layout:
    - Menu bar
    - Toolbar
    - Central widget (emotion input + results)
    - Status bar
    """

    # Signals (GUI → Controller)
    generate_requested = Signal(str, dict, dict)  # emotion text, emotion params, technical params
    preview_requested = Signal()
    export_requested = Signal()
    new_project_requested = Signal()
    open_project_requested = Signal(str)  # file path
    save_project_requested = Signal(str)  # file path
    preferences_requested = Signal()
    ai_analysis_requested = Signal()
    batch_process_requested = Signal()
    presets_requested = Signal()
    undo_requested = Signal(str)  # component name
    redo_requested = Signal(str)  # component name

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KmiDi - Intelligent Digital Audio Workstation")
        self.setMinimumSize(1000, 700)

        # Current project path
        self.current_project_path: Path = None

        # Setup UI
        self._setup_menu()
        self._setup_toolbar()
        self._setup_docks()
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
        self.undo_action = edit_menu.addAction("Undo", self._on_undo)
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action = edit_menu.addAction("Redo", self._on_redo)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addSeparator()
        edit_menu.addAction("Preferences", self._on_preferences)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Show AI Assistant", self._on_toggle_ai_dock)
        view_menu.addAction("Show Logs", self._on_toggle_logs_dock)
        view_menu.addAction("Show Control Surface", self._on_toggle_control_surface)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("AI Analysis", self._on_ai_analysis)
        tools_menu.addAction("Batch Process", self._on_batch_process)

        # Presets menu
        presets_menu = menubar.addMenu("Presets")
        presets_menu.addAction("Manage Presets", self._on_manage_presets)

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

    def _setup_docks(self):
        """Setup dock widgets."""
        # AI Assistant dock
        self.ai_dock = AIAssistantDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)
        self.ai_dock.setVisible(False)  # Hidden by default

        # Logs dock
        self.logs_dock = LogsDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.logs_dock)
        self.logs_dock.setVisible(False)  # Hidden by default

        # Control Surface dock
        from PySide6.QtWidgets import QDockWidget
        self.control_surface_dock = QDockWidget("Control Surface", self)
        self.control_surface_panel = ControlSurfacePanel(
            project_path=self.current_project_path,
            parent=self.control_surface_dock
        )
        self.control_surface_dock.setWidget(self.control_surface_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.control_surface_dock)
        self.control_surface_dock.setVisible(False)  # Hidden by default

    def _setup_central_widget(self):
        """Setup central widget with splitter for parameters and content."""
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Left panel: Parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(12)

        # Emotion parameter panel
        self.emotion_params = EmotionParameterPanel()
        left_layout.addWidget(self.emotion_params)

        # Technical parameter panel
        self.technical_params = TechnicalParameterPanel()
        left_layout.addWidget(self.technical_params)

        left_layout.addStretch()

        # Right panel: Input and Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Emotion input label
        input_label = QLabel("What's on your heart?")
        input_label.setObjectName("ParamTitle")
        right_layout.addWidget(input_label)

        # Emotion input text area
        self.emotion_input = QTextEdit()
        self.emotion_input.setPlaceholderText(
            "Describe what you're feeling...\n"
            "Example: 'I'm feeling grief hidden as love'"
        )
        self.emotion_input.setMinimumHeight(120)
        right_layout.addWidget(self.emotion_input)

        # Results display
        results_label = QLabel("Results")
        results_label.setObjectName("ParamTitle")
        right_layout.addWidget(results_label)

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText(
            "Generated music will appear here...")
        right_layout.addWidget(self.results_display)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)  # Parameters don't stretch
        splitter.setStretchFactor(1, 1)  # Content stretches
        splitter.setSizes([280, 720])  # Initial sizes

        main_layout.addWidget(splitter)

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
            QMessageBox.warning(self, "No Input",
                                "Please enter your emotional intent.")
            return

        # Get parameter values
        emotion_params = self.emotion_params.get_values()
        technical_params = self.technical_params.get_values()

        self.generate_requested.emit(text, emotion_params, technical_params)

    def _on_preview_clicked(self):
        """Handle preview button click."""
        self.preview_requested.emit()

    def _on_export_clicked(self):
        """Handle export button click."""
        self.export_requested.emit()

    # Menu handlers

    def _on_new_project(self):
        """Handle new project menu action."""
        self.current_project_path = None
        self.emotion_input.clear()
        self.results_display.clear()
        self.setWindowTitle("KmiDi - Intelligent Digital Audio Workstation")
        self.new_project_requested.emit()

    def _on_open_project(self):
        """Handle open project menu action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(Path.home() / "Documents"),
            "KmiDi Project (*.kmidi);;All Files (*)",
        )
        if file_path:
            self.current_project_path = Path(file_path)
            self.open_project_requested.emit(file_path)

    def _on_save_project(self):
        """Handle save project menu action."""
        if not self.current_project_path:
            # Save as
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Project",
                str(Path.home() / "Documents"),
                "KmiDi Project (*.kmidi);;All Files (*)",
            )
            if file_path:
                if not file_path.endswith(".kmidi"):
                    file_path += ".kmidi"
                self.current_project_path = Path(file_path)
                self.save_project_requested.emit(file_path)
        else:
            self.save_project_requested.emit(str(self.current_project_path))

    def _on_export(self):
        """Handle export menu action."""
        self.export_requested.emit()

    def _on_undo(self):
        """Handle undo menu action."""
        from kmidi_gui.core.history import HistoryComponent
        self.undo_requested.emit(HistoryComponent.INTENT.value)

    def _on_redo(self):
        """Handle redo menu action."""
        from kmidi_gui.core.history import HistoryComponent
        self.redo_requested.emit(HistoryComponent.INTENT.value)

    def _on_preferences(self):
        """Handle preferences menu action."""
        self.preferences_requested.emit()

    def _on_toggle_ai_dock(self):
        """Handle toggle AI dock menu action."""
        self.ai_dock.setVisible(not self.ai_dock.isVisible())

    def _on_toggle_logs_dock(self):
        """Handle toggle logs dock menu action."""
        self.logs_dock.setVisible(not self.logs_dock.isVisible())

    def _on_toggle_control_surface(self):
        """Handle toggle control surface dock menu action."""
        self.control_surface_dock.setVisible(not self.control_surface_dock.isVisible())

    def _on_ai_analysis(self):
        """Handle AI analysis menu action."""
        self.ai_analysis_requested.emit()

    def _on_batch_process(self):
        """Handle batch process menu action."""
        self.batch_process_requested.emit()

    def _on_manage_presets(self):
        """Handle manage presets menu action."""
        self.presets_requested.emit()

    def _on_about(self):
        """Handle about menu action."""
        QMessageBox.about(
            self,
            "About KmiDi",
            "KmiDi - Intelligent Digital Audio Workstation\n\n"
            "Version 1.0.0\n"
            "Professional music generation tool",
        )

    def _on_documentation(self):
        """Handle documentation menu action."""
        QMessageBox.information(
            self,
            "Documentation",
            "KmiDi Documentation\n\n"
            "Visit: https://github.com/kmidi/kmidi/docs\n\n"
            "Or check the docs/ folder in the project directory.",
        )

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

    # Public methods for dock widgets

    def get_logs_dock(self) -> LogsDock:
        """Get logs dock widget."""
        return self.logs_dock

    def get_ai_dock(self) -> AIAssistantDock:
        """Get AI assistant dock widget."""
        return self.ai_dock

    def set_project_path(self, path: Path):
        """Set current project path and update window title."""
        self.current_project_path = path
        if path:
            self.setWindowTitle(f"KmiDi - {path.name}")
        else:
            self.setWindowTitle("KmiDi - Intelligent Digital Audio Workstation")
