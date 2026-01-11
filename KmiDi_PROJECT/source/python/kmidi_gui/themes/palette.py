"""Audio tool color palette for Qt application."""

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


def apply_audio_palette(app: QApplication) -> None:
    """Apply professional audio tool palette to Qt application.
    
    This palette provides:
    - Dark, neutral backgrounds (not pure black)
    - Soft gray text (not white)
    - Muted blue accent
    - Eye-fatigue-resistant colors
    """
    p = QPalette()

    # Window colors
    p.setColor(QPalette.Window, QColor("#1e1e1e"))
    p.setColor(QPalette.WindowText, QColor("#d0d0d0"))

    # Base colors (for text inputs, lists)
    p.setColor(QPalette.Base, QColor("#181818"))
    p.setColor(QPalette.AlternateBase, QColor("#202020"))

    # Text colors
    p.setColor(QPalette.Text, QColor("#d0d0d0"))
    p.setColor(QPalette.PlaceholderText, QColor("#808080"))

    # Button colors
    p.setColor(QPalette.Button, QColor("#2a2a2a"))
    p.setColor(QPalette.ButtonText, QColor("#d0d0d0"))

    # Highlight (selection, focus)
    p.setColor(QPalette.Highlight, QColor("#3a6ea5"))  # muted blue
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))

    # Tooltip colors
    p.setColor(QPalette.ToolTipBase, QColor("#2a2a2a"))
    p.setColor(QPalette.ToolTipText, QColor("#e0e0e0"))

    # Disabled colors
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#777777"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#777777"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#777777"))

    app.setPalette(p)


# Color constants for programmatic use
class AudioColors:
    """Audio tool color constants."""
    
    # Backgrounds
    BG_WINDOW = "#1e1e1e"
    BG_PANEL = "#202020"
    BG_BASE = "#181818"
    
    # Text
    TEXT_PRIMARY = "#d0d0d0"
    TEXT_SECONDARY = "#a8a8a8"
    TEXT_PLACEHOLDER = "#808080"
    
    # Accent
    ACCENT_BLUE = "#3a6ea5"
    ACCENT_BLUE_LIGHT = "#5a8fcf"
    ACCENT_BLUE_HOVER = "#6aa0e0"
    
    # Buttons
    BUTTON_BG = "#2a2a2a"
    BUTTON_BG_HOVER = "#333333"
    BUTTON_BG_PRESSED = "#242424"
    
    # Borders
    BORDER = "#2a2a2a"
    BORDER_LIGHT = "#3a3a3a"
    
    # Status colors
    STATUS_OK = "#7fc97f"
    STATUS_WARN = "#e6b450"
    STATUS_ERROR = "#e06c75"
    
    # Confidence colors
    CONFIDENCE_HIGH = "#7fc97f"
    CONFIDENCE_MEDIUM = "#e6b450"
    CONFIDENCE_LOW = "#e06c75"

