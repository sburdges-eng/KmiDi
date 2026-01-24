"""Dock widgets for KmiDi GUI.

Dock widgets for:
- AI Assistant (analysis results, confidence indicators)
- Logs (operation logs, errors)
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase

from kmidi_gui.themes.palette import AudioColors


class AIAssistantDock(QDockWidget):
    """AI Assistant dock widget.

    Shows AI analysis results, confidence indicators, and action buttons.
    """

    # Signals
    preview_requested = Signal()
    apply_requested = Signal()
    ignore_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("AI Assistant", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(300)

        # Create main widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Analysis summary
        summary_label = QLabel("Analysis Summary")
        summary_label.setObjectName("ParamTitle")
        layout.addWidget(summary_label)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setPlaceholderText(
            "AI analysis results will appear here...")
        layout.addWidget(self.summary_text)

        # Confidence indicator
        confidence_label = QLabel("Confidence")
        confidence_label.setObjectName("ParamTitle")
        layout.addWidget(confidence_label)

        self.confidence_label = QLabel("N/A")
        self.confidence_label.setObjectName("ParamValue")
        layout.addWidget(self.confidence_label)

        # Action buttons
        buttons_layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_requested.emit)
        buttons_layout.addWidget(self.preview_btn)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_requested.emit)
        buttons_layout.addWidget(self.apply_btn)

        self.ignore_btn = QPushButton("Ignore")
        self.ignore_btn.clicked.connect(self.ignore_requested.emit)
        buttons_layout.addWidget(self.ignore_btn)

        layout.addLayout(buttons_layout)

        layout.addStretch()

        self.setWidget(widget)

    def set_analysis(self, summary: str, confidence: float = None):
        """Set analysis results.

        Args:
            summary: Analysis summary text
            confidence: Confidence score (0.0-1.0) or None
        """
        self.summary_text.setPlainText(summary)

        if confidence is not None:
            confidence_pct = int(confidence * 100)
            color = AudioColors.STATUS_OK if confidence >= 0.7 else AudioColors.STATUS_WARN if confidence >= 0.5 else AudioColors.STATUS_ERROR
            self.confidence_label.setText(
                f'<span style="color: {color}">{confidence_pct}%</span>'
            )
        else:
            self.confidence_label.setText("N/A")

    def clear(self):
        """Clear analysis results."""
        self.summary_text.clear()
        self.confidence_label.setText("N/A")


class LogsDock(QDockWidget):
    """Logs dock widget.

    Shows operation logs, errors, and debug messages.
    """

    def __init__(self, parent=None):
        super().__init__("Logs", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.setMinimumHeight(150)

        # Create main widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font_db = QFontDatabase()
        available_fonts = font_db.families()
        if "Monaco" in available_fonts:
            self.log_text.setFont(QFont("Monaco", 10))
        else:
            self.log_text.setFont(QFont("Courier", 10))
        self.log_text.setPlaceholderText("Operation logs will appear here...")
        layout.addWidget(self.log_text)

        # Clear button
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.log_text.clear)
        buttons_layout.addWidget(clear_btn)

        layout.addLayout(buttons_layout)

        self.setWidget(widget)

    def add_log(self, message: str, level: str = "INFO"):
        """Add log message.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color by level
        color_map = {
            "INFO": AudioColors.STATUS_OK,
            "WARNING": AudioColors.STATUS_WARN,
            "ERROR": AudioColors.STATUS_ERROR,
            "DEBUG": "#888888",
        }
        color = color_map.get(level, "#d0d0d0")

        formatted = f'<span style="color: {color}">[{timestamp}] [{level}]</span> {message}'
        self.log_text.append(formatted)

        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear all logs."""
        self.log_text.clear()
