# music_brain/effects/base.py
"""
Base classes and core types for the guitar effects engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import json
import math


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class EffectCategory(Enum):
    """Effect categories."""
    DISTORTION = "distortion"
    MODULATION = "modulation"
    TIME = "time"
    DYNAMICS = "dynamics"
    FILTER = "filter"
    PITCH = "pitch"
    AMP = "amp"
    SPECIAL = "special"
    UTILITY = "utility"


class WaveShape(Enum):
    """LFO/oscillator wave shapes."""
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAW_UP = "saw_up"
    SAW_DOWN = "saw_down"
    RANDOM = "random"
    SAMPLE_HOLD = "sample_hold"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    PULSE = "pulse"
    STEPPED = "stepped"


class FilterType(Enum):
    """Filter types."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    ALLPASS = "allpass"
    PEAK = "peak"
    LOWSHELF = "lowshelf"
    HIGHSHELF = "highshelf"
    COMB = "comb"
    FORMANT = "formant"


class DistortionType(Enum):
    """Distortion circuit types."""
    TUBE = "tube"
    TRANSISTOR = "transistor"
    DIODE = "diode"
    FUZZ = "fuzz"
    RECTIFIER = "rectifier"
    BITCRUSH = "bitcrush"
    WAVEFOLD = "wavefold"
    TAPE = "tape"


class DelayType(Enum):
    """Delay types."""
    DIGITAL = "digital"
    ANALOG = "analog"
    TAPE = "tape"
    REVERSE = "reverse"
    PING_PONG = "ping_pong"
    MULTI_TAP = "multi_tap"
    GRANULAR = "granular"
    SHIMMER = "shimmer"
    DUCKING = "ducking"
    MODULATED = "modulated"
    DIFFUSED = "diffused"
    SWELL = "swell"


class ReverbType(Enum):
    """Reverb algorithm types."""
    ROOM = "room"
    HALL = "hall"
    PLATE = "plate"
    SPRING = "spring"
    CHAMBER = "chamber"
    CATHEDRAL = "cathedral"
    CAVE = "cave"
    SHIMMER = "shimmer"
    REVERSE = "reverse"
    GATED = "gated"
    CONVOLUTION = "convolution"
    GRANULAR = "granular"
    INFINITE = "infinite"
    MODULATED = "modulated"
    NONLINEAR = "nonlinear"


class AmpModel(Enum):
    """Amp simulation models."""
    CLEAN_TWIN = "clean_twin"
    CLEAN_JAZZ = "clean_jazz"
    CLEAN_ACOUSTIC = "clean_acoustic"
    CRUNCH_BRIT = "crunch_brit"
    CRUNCH_PLEXI = "crunch_plexi"
    CRUNCH_BLUES = "crunch_blues"
    DRIVE_MODERN = "drive_modern"
    DRIVE_RECTIFIER = "drive_rectifier"
    DRIVE_5150 = "drive_5150"
    HIGH_GAIN = "high_gain"
    METAL = "metal"
    DJENT = "djent"
    BASS_SVT = "bass_svt"
    BASS_B15 = "bass_b15"
    ACOUSTIC_SIM = "acoustic_sim"


class CabinetType(Enum):
    """Cabinet/IR types."""
    CAB_1X8 = "1x8"
    CAB_1X10 = "1x10"
    CAB_1X12 = "1x12"
    CAB_2X10 = "2x10"
    CAB_2X12 = "2x12"
    CAB_4X10 = "4x10"
    CAB_4X12 = "4x12"
    CAB_8X10 = "8x10"
    OPEN_BACK = "open_back"
    CLOSED_BACK = "closed_back"
    IR_CUSTOM = "ir_custom"


# =============================================================================
# PARAMETER SYSTEM
# =============================================================================

@dataclass
class Parameter:
    """
    A single effect parameter with full modulation support.
    """
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 1.0
    default: float = 0.5
    unit: str = ""
    description: str = ""
    
    # Modulation
    mod_amount: float = 0.0  # -1.0 to 1.0
    mod_source: Optional[str] = None
    
    # Curve/response
    curve: str = "linear"  # linear, exponential, logarithmic, s-curve
    
    # Quantization
    steps: Optional[int] = None  # If set, quantize to N steps
    
    # Randomization range
    random_min: Optional[float] = None
    random_max: Optional[float] = None
    
    def get_normalized(self) -> float:
        """Get value normalized to 0-1."""
        return (self.value - self.min_val) / (self.max_val - self.min_val)
    
    def set_normalized(self, val: float):
        """Set value from 0-1 normalized input."""
        self.value = self.min_val + val * (self.max_val - self.min_val)
    
    def apply_curve(self, val: float) -> float:
        """Apply response curve to value."""
        if self.curve == "exponential":
            return val ** 2
        elif self.curve == "logarithmic":
            return math.sqrt(val) if val >= 0 else 0
        elif self.curve == "s-curve":
            return 0.5 * (1 + math.tanh(3 * (val - 0.5)))
        return val  # linear
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "min": self.min_val,
            "max": self.max_val,
            "default": self.default,
            "unit": self.unit,
            "mod_amount": self.mod_amount,
            "mod_source": self.mod_source,
            "curve": self.curve,
        }


# =============================================================================
# BASE EFFECT CLASS
# =============================================================================

class BaseEffect(ABC):
    """
    Abstract base class for all effects.
    """
    
    def __init__(
        self,
        name: str,
        category: EffectCategory,
        bypass: bool = False,
    ):
        self.name = name
        self.category = category
        self.bypass = bypass
        self.parameters: Dict[str, Parameter] = {}
        self.input_gain: float = 1.0
        self.output_gain: float = 1.0
        self.wet_dry_mix: float = 1.0  # 0 = dry, 1 = wet
        self._init_parameters()
    
    @abstractmethod
    def _init_parameters(self):
        """Initialize effect-specific parameters."""
        pass
    
    @abstractmethod
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        """Process audio samples through the effect."""
        pass
    
    def add_parameter(
        self,
        name: str,
        default: float = 0.5,
        min_val: float = 0.0,
        max_val: float = 1.0,
        unit: str = "",
        description: str = "",
        curve: str = "linear",
    ) -> Parameter:
        """Add a parameter to this effect."""
        param = Parameter(
            name=name,
            value=default,
            min_val=min_val,
            max_val=max_val,
            default=default,
            unit=unit,
            description=description,
            curve=curve,
        )
        self.parameters[name] = param
        return param
    
    def get_param(self, name: str) -> float:
        """Get parameter value."""
        if name in self.parameters:
            return self.parameters[name].value
        return 0.0
    
    def set_param(self, name: str, value: float):
        """Set parameter value."""
        if name in self.parameters:
            param = self.parameters[name]
            param.value = max(param.min_val, min(param.max_val, value))
    
    def set_param_normalized(self, name: str, value: float):
        """Set parameter from normalized 0-1 value."""
        if name in self.parameters:
            self.parameters[name].set_normalized(value)
    
    def randomize(self, amount: float = 1.0):
        """Randomize all parameters within their ranges."""
        import random
        for param in self.parameters.values():
            if param.random_min is not None and param.random_max is not None:
                rng = param.random_max - param.random_min
            else:
                rng = param.max_val - param.min_val
            
            center = param.value
            deviation = rng * amount * 0.5
            new_val = center + random.uniform(-deviation, deviation)
            param.value = max(param.min_val, min(param.max_val, new_val))
    
    def to_dict(self) -> Dict:
        """Serialize effect to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "bypass": self.bypass,
            "input_gain": self.input_gain,
            "output_gain": self.output_gain,
            "wet_dry_mix": self.wet_dry_mix,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
        }
    
    def from_dict(self, data: Dict):
        """Load effect settings from dictionary."""
        self.bypass = data.get("bypass", False)
        self.input_gain = data.get("input_gain", 1.0)
        self.output_gain = data.get("output_gain", 1.0)
        self.wet_dry_mix = data.get("wet_dry_mix", 1.0)
        
        for name, pdata in data.get("parameters", {}).items():
            if name in self.parameters:
                self.parameters[name].value = pdata.get("value", self.parameters[name].default)
                self.parameters[name].mod_amount = pdata.get("mod_amount", 0.0)
                self.parameters[name].mod_source = pdata.get("mod_source")


# =============================================================================
# MODULATION SOURCES
# =============================================================================

class ModulationSource(ABC):
    """Base class for modulation sources."""
    
    def __init__(self, name: str):
        self.name = name
        self.output: float = 0.0
    
    @abstractmethod
    def update(self, delta_time: float, **kwargs) -> float:
        """Update and return current modulation value (-1 to 1)."""
        pass


@dataclass
class LFOSource(ModulationSource):
    """Low Frequency Oscillator modulation source."""
    
    rate: float = 1.0  # Hz
    shape: WaveShape = WaveShape.SINE
    phase: float = 0.0
    depth: float = 1.0
    offset: float = 0.0
    sync_to_tempo: bool = False
    tempo_division: str = "1/4"  # 1/1, 1/2, 1/4, 1/8, 1/16, 1/32
    retrigger: bool = False
    
    _phase_accum: float = field(default=0.0, init=False)
    
    def __init__(self, name: str = "LFO", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._phase_accum = self.phase
    
    def update(self, delta_time: float, tempo_bpm: float = 120.0, **kwargs) -> float:
        """Generate LFO output."""
        if self.sync_to_tempo:
            # Calculate rate from tempo division
            divisions = {
                "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5,
                "1/16": 0.25, "1/32": 0.125, "1/2T": 4/3, "1/4T": 2/3,
                "1/8T": 1/3, "1/16T": 1/6, "1/2D": 3.0, "1/4D": 1.5,
            }
            beats = divisions.get(self.tempo_division, 1.0)
            self.rate = tempo_bpm / 60.0 / beats
        
        self._phase_accum += self.rate * delta_time
        if self._phase_accum >= 1.0:
            self._phase_accum -= 1.0
        
        phase = self._phase_accum * 2 * math.pi
        
        if self.shape == WaveShape.SINE:
            self.output = math.sin(phase)
        elif self.shape == WaveShape.TRIANGLE:
            self.output = 2 * abs(2 * (self._phase_accum - 0.5)) - 1
        elif self.shape == WaveShape.SQUARE:
            self.output = 1.0 if self._phase_accum < 0.5 else -1.0
        elif self.shape == WaveShape.SAW_UP:
            self.output = 2 * self._phase_accum - 1
        elif self.shape == WaveShape.SAW_DOWN:
            self.output = 1 - 2 * self._phase_accum
        elif self.shape == WaveShape.RANDOM:
            import random
            self.output = random.uniform(-1, 1)
        elif self.shape == WaveShape.EXPONENTIAL:
            self.output = math.exp(self._phase_accum * 2) / math.e - 1
        else:
            self.output = math.sin(phase)
        
        return (self.output * self.depth) + self.offset


@dataclass
class EnvelopeFollower(ModulationSource):
    """Envelope follower - tracks input amplitude."""
    
    attack: float = 0.01  # seconds
    release: float = 0.1  # seconds
    sensitivity: float = 1.0
    threshold: float = 0.0
    
    _envelope: float = field(default=0.0, init=False)
    
    def __init__(self, name: str = "EnvFollow", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def update(self, delta_time: float, input_level: float = 0.0, **kwargs) -> float:
        """Track input envelope."""
        target = abs(input_level) * self.sensitivity
        
        if target > self._envelope:
            coef = 1.0 - math.exp(-delta_time / max(0.001, self.attack))
        else:
            coef = 1.0 - math.exp(-delta_time / max(0.001, self.release))
        
        self._envelope += coef * (target - self._envelope)
        
        self.output = max(0, (self._envelope - self.threshold) / (1 - self.threshold))
        return self.output * 2 - 1  # Convert to -1 to 1


@dataclass 
class StepSequencer(ModulationSource):
    """Step sequencer modulation source."""
    
    steps: List[float] = field(default_factory=lambda: [0.0] * 8)
    num_steps: int = 8
    rate: float = 1.0  # Hz
    sync_to_tempo: bool = True
    tempo_division: str = "1/8"
    glide: float = 0.0  # 0-1, smoothing between steps
    direction: str = "forward"  # forward, backward, bounce, random
    
    _current_step: int = field(default=0, init=False)
    _step_accum: float = field(default=0.0, init=False)
    _bounce_dir: int = field(default=1, init=False)
    
    def __init__(self, name: str = "StepSeq", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if len(self.steps) < self.num_steps:
            self.steps.extend([0.0] * (self.num_steps - len(self.steps)))
    
    def update(self, delta_time: float, tempo_bpm: float = 120.0, **kwargs) -> float:
        """Advance step sequencer."""
        if self.sync_to_tempo:
            divisions = {
                "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5,
                "1/16": 0.25, "1/32": 0.125,
            }
            beats = divisions.get(self.tempo_division, 0.5)
            self.rate = tempo_bpm / 60.0 / beats
        
        self._step_accum += self.rate * delta_time
        
        if self._step_accum >= 1.0:
            self._step_accum -= 1.0
            self._advance_step()
        
        target = self.steps[self._current_step % len(self.steps)]
        
        if self.glide > 0:
            glide_coef = 1.0 - math.exp(-delta_time / max(0.001, self.glide * 0.5))
            self.output += glide_coef * (target - self.output)
        else:
            self.output = target
        
        return self.output
    
    def _advance_step(self):
        import random
        
        if self.direction == "forward":
            self._current_step = (self._current_step + 1) % self.num_steps
        elif self.direction == "backward":
            self._current_step = (self._current_step - 1) % self.num_steps
        elif self.direction == "bounce":
            self._current_step += self._bounce_dir
            if self._current_step >= self.num_steps - 1 or self._current_step <= 0:
                self._bounce_dir *= -1
        elif self.direction == "random":
            self._current_step = random.randint(0, self.num_steps - 1)


@dataclass
class RandomSource(ModulationSource):
    """Random/noise modulation source."""
    
    rate: float = 1.0  # Changes per second
    smoothing: float = 0.0  # 0-1
    range_min: float = -1.0
    range_max: float = 1.0
    
    _target: float = field(default=0.0, init=False)
    _accum: float = field(default=0.0, init=False)
    
    def __init__(self, name: str = "Random", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def update(self, delta_time: float, **kwargs) -> float:
        import random
        
        self._accum += self.rate * delta_time
        
        if self._accum >= 1.0:
            self._accum -= 1.0
            self._target = random.uniform(self.range_min, self.range_max)
        
        if self.smoothing > 0:
            coef = 1.0 - math.exp(-delta_time / max(0.001, self.smoothing))
            self.output += coef * (self._target - self.output)
        else:
            self.output = self._target
        
        return self.output


@dataclass
class ExpressionInput(ModulationSource):
    """Expression pedal/controller input."""
    
    input_value: float = 0.0
    curve: str = "linear"  # linear, exponential, logarithmic
    invert: bool = False
    min_output: float = -1.0
    max_output: float = 1.0
    
    def __init__(self, name: str = "Expression", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def update(self, delta_time: float, expression_value: float = None, **kwargs) -> float:
        if expression_value is not None:
            self.input_value = expression_value
        
        val = self.input_value
        
        if self.curve == "exponential":
            val = val ** 2
        elif self.curve == "logarithmic":
            val = math.sqrt(val) if val >= 0 else 0
        
        if self.invert:
            val = 1.0 - val
        
        self.output = self.min_output + val * (self.max_output - self.min_output)
        return self.output


@dataclass
class MIDISource(ModulationSource):
    """MIDI CC/Note modulation source."""
    
    cc_number: int = 1  # Modulation wheel default
    channel: int = 0  # All channels
    learn_mode: bool = False
    smoothing: float = 0.0
    
    _cc_value: float = field(default=0.0, init=False)
    
    def __init__(self, name: str = "MIDI", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def update(self, delta_time: float, midi_cc: Dict[int, float] = None, **kwargs) -> float:
        if midi_cc and self.cc_number in midi_cc:
            self._cc_value = midi_cc[self.cc_number] / 127.0
        
        if self.smoothing > 0:
            coef = 1.0 - math.exp(-delta_time / max(0.001, self.smoothing))
            self.output += coef * (self._cc_value - self.output)
        else:
            self.output = self._cc_value
        
        return self.output * 2 - 1  # Convert to -1 to 1


@dataclass
class EmotionSource(ModulationSource):
    """
    DAiW-specific: Modulation driven by detected emotion.
    Maps emotional states to modulation values.
    """
    
    emotion_map: Dict[str, float] = field(default_factory=lambda: {
        "grief": -0.8,
        "rage": 0.9,
        "fear": 0.5,
        "nostalgia": -0.3,
        "defiance": 0.7,
        "tenderness": -0.5,
        "dissociation": 0.0,
        "awe": 0.4,
        "confusion": 0.6,
        "longing": -0.4,
        "hope": 0.3,
    })
    current_emotion: str = "neutral"
    intensity: float = 1.0
    transition_time: float = 2.0  # Seconds to transition between emotions
    
    _target: float = field(default=0.0, init=False)
    
    def __init__(self, name: str = "Emotion", **kwargs):
        super().__init__(name)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def set_emotion(self, emotion: str, intensity: float = 1.0):
        """Set current emotion."""
        self.current_emotion = emotion.lower()
        self.intensity = intensity
        self._target = self.emotion_map.get(self.current_emotion, 0.0) * intensity
    
    def update(self, delta_time: float, emotion: str = None, **kwargs) -> float:
        if emotion:
            self.set_emotion(emotion, kwargs.get("intensity", 1.0))
        
        if self.transition_time > 0:
            coef = 1.0 - math.exp(-delta_time / max(0.001, self.transition_time))
            self.output += coef * (self._target - self.output)
        else:
            self.output = self._target
        
        return self.output

