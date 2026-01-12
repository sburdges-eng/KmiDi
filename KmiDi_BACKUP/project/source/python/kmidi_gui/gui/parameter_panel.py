"""Parameter panels for emotion controls.

Parameter panels for:
- Emotional parameters (valence, arousal, intensity)
- Technical parameters (key, BPM, genre)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QComboBox, QGroupBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal


class EmotionParameterPanel(QWidget):
    """Panel for emotional parameters.
    
    Controls:
    - Valence (negative to positive)
    - Arousal (low to high)
    - Intensity (weak to strong)
    """
    
    # Signals
    valence_changed = Signal(float)  # -1.0 to 1.0
    arousal_changed = Signal(float)  # 0.0 to 1.0
    intensity_changed = Signal(float)  # 0.0 to 1.0
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ParamPanel")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("Emotional Parameters")
        title.setObjectName("ParamTitle")
        layout.addWidget(title)
        
        # Valence slider
        valence_layout = QHBoxLayout()
        valence_label = QLabel("Valence:")
        valence_label.setObjectName("ParamLabel")
        valence_label.setMinimumWidth(80)
        valence_layout.addWidget(valence_label)
        
        self.valence_slider = QSlider(Qt.Horizontal)
        self.valence_slider.setMinimum(-100)
        self.valence_slider.setMaximum(100)
        self.valence_slider.setValue(0)
        self.valence_slider.valueChanged.connect(self._on_valence_changed)
        valence_layout.addWidget(self.valence_slider)
        
        self.valence_value = QLabel("0.0")
        self.valence_value.setObjectName("ParamValue")
        self.valence_value.setMinimumWidth(40)
        valence_layout.addWidget(self.valence_value)
        
        layout.addLayout(valence_layout)
        
        # Arousal slider
        arousal_layout = QHBoxLayout()
        arousal_label = QLabel("Arousal:")
        arousal_label.setObjectName("ParamLabel")
        arousal_label.setMinimumWidth(80)
        arousal_layout.addWidget(arousal_label)
        
        self.arousal_slider = QSlider(Qt.Horizontal)
        self.arousal_slider.setMinimum(0)
        self.arousal_slider.setMaximum(100)
        self.arousal_slider.setValue(50)
        self.arousal_slider.valueChanged.connect(self._on_arousal_changed)
        arousal_layout.addWidget(self.arousal_slider)
        
        self.arousal_value = QLabel("0.5")
        self.arousal_value.setObjectName("ParamValue")
        self.arousal_value.setMinimumWidth(40)
        arousal_layout.addWidget(self.arousal_value)
        
        layout.addLayout(arousal_layout)
        
        # Intensity slider
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Intensity:")
        intensity_label.setObjectName("ParamLabel")
        intensity_label.setMinimumWidth(80)
        intensity_layout.addWidget(intensity_label)
        
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)
        
        self.intensity_value = QLabel("0.5")
        self.intensity_value.setObjectName("ParamValue")
        self.intensity_value.setMinimumWidth(40)
        intensity_layout.addWidget(self.intensity_value)
        
        layout.addLayout(intensity_layout)
        
        layout.addStretch()
    
    def _on_valence_changed(self, value: int):
        """Handle valence slider change."""
        valence = value / 100.0
        self.valence_value.setText(f"{valence:.2f}")
        self.valence_changed.emit(valence)
    
    def _on_arousal_changed(self, value: int):
        """Handle arousal slider change."""
        arousal = value / 100.0
        self.arousal_value.setText(f"{arousal:.2f}")
        self.arousal_changed.emit(arousal)
    
    def _on_intensity_changed(self, value: int):
        """Handle intensity slider change."""
        intensity = value / 100.0
        self.intensity_value.setText(f"{intensity:.2f}")
        self.intensity_changed.emit(intensity)
    
    def get_values(self) -> dict:
        """Get current parameter values.
        
        Returns:
            Dictionary with valence, arousal, intensity
        """
        return {
            "valence": self.valence_slider.value() / 100.0,
            "arousal": self.arousal_slider.value() / 100.0,
            "intensity": self.intensity_slider.value() / 100.0,
        }


class TechnicalParameterPanel(QWidget):
    """Panel for technical parameters.
    
    Controls:
    - Key (musical key)
    - BPM (beats per minute)
    - Genre (musical genre)
    """
    
    # Signals
    key_changed = Signal(str)
    bpm_changed = Signal(int)
    genre_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ParamPanel")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("Technical Parameters")
        title.setObjectName("ParamTitle")
        layout.addWidget(title)
        
        # Key selector
        key_layout = QHBoxLayout()
        key_label = QLabel("Key:")
        key_label.setObjectName("ParamLabel")
        key_label.setMinimumWidth(80)
        key_layout.addWidget(key_label)
        
        self.key_combo = QComboBox()
        self.key_combo.addItems([
            "Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
        ])
        self.key_combo.currentTextChanged.connect(self.key_changed.emit)
        key_layout.addWidget(self.key_combo)
        
        layout.addLayout(key_layout)
        
        # BPM spinbox
        bpm_layout = QHBoxLayout()
        bpm_label = QLabel("BPM:")
        bpm_label.setObjectName("ParamLabel")
        bpm_label.setMinimumWidth(80)
        bpm_layout.addWidget(bpm_label)
        
        self.bpm_spinbox = QSpinBox()
        self.bpm_spinbox.setMinimum(40)
        self.bpm_spinbox.setMaximum(200)
        self.bpm_spinbox.setValue(120)
        self.bpm_spinbox.valueChanged.connect(self.bpm_changed.emit)
        bpm_layout.addWidget(self.bpm_spinbox)
        
        layout.addLayout(bpm_layout)
        
        # Genre selector
        genre_layout = QHBoxLayout()
        genre_label = QLabel("Genre:")
        genre_label.setObjectName("ParamLabel")
        genre_label.setMinimumWidth(80)
        genre_layout.addWidget(genre_label)
        
        self.genre_combo = QComboBox()
        self.genre_combo.addItems([
            "Auto", "Ambient", "Blues", "Classical", "Electronic", "Folk",
            "Funk", "Jazz", "Pop", "Rock", "Soul", "Other"
        ])
        self.genre_combo.currentTextChanged.connect(self.genre_changed.emit)
        genre_layout.addWidget(self.genre_combo)
        
        layout.addLayout(genre_layout)
        
        layout.addStretch()
    
    def get_values(self) -> dict:
        """Get current parameter values.
        
        Returns:
            Dictionary with key, bpm, genre
        """
        return {
            "key": None if self.key_combo.currentText() == "Auto" else self.key_combo.currentText(),
            "bpm": self.bpm_spinbox.value(),
            "genre": None if self.genre_combo.currentText() == "Auto" else self.genre_combo.currentText(),
        }
