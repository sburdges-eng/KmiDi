# music_brain/effects/guitar_fx.py
"""
Guitar FX Engine - The complete effects system.
Combines all effects, modulation, routing, and presets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import json
import math
from pathlib import Path

from music_brain.effects.base import (
    BaseEffect,
    EffectCategory,
    Parameter,
    ModulationSource,
    LFOSource,
    EnvelopeFollower,
    StepSequencer,
    RandomSource,
    ExpressionInput,
    MIDISource,
    EmotionSource,
    WaveShape,
    DistortionType,
    DelayType,
    ReverbType,
    FilterType,
    AmpModel,
    CabinetType,
)

from music_brain.effects.effects import (
    DistortionEffect,
    OverdriveEffect,
    FuzzEffect,
    ChorusEffect,
    FlangerEffect,
    PhaserEffect,
    TremoloEffect,
    VibratoEffect,
    RotaryEffect,
    RingModEffect,
    UnivibeEffect,
    DelayEffect,
    ReverbEffect,
    CompressorEffect,
    NoiseGateEffect,
    EQEffect,
    WahEffect,
    FilterEffect,
    PitchShiftEffect,
    HarmonizerEffect,
    OctaverEffect,
    AmpSimEffect,
    CabinetSimEffect,
    LooperEffect,
    GranularEffect,
    ShimmerEffect,
    FreezeEffect,
    SlicerEffect,
    BitcrusherEffect,
)


# =============================================================================
# CONSTANTS
# =============================================================================

EFFECT_CATEGORIES = {
    EffectCategory.DISTORTION: ["Distortion", "Overdrive", "Fuzz"],
    EffectCategory.MODULATION: ["Chorus", "Flanger", "Phaser", "Tremolo", "Vibrato", "Rotary", "Ring Mod", "Univibe"],
    EffectCategory.TIME: ["Delay", "Reverb"],
    EffectCategory.DYNAMICS: ["Compressor", "Noise Gate"],
    EffectCategory.FILTER: ["EQ", "Wah", "Filter"],
    EffectCategory.PITCH: ["Pitch Shift", "Harmonizer", "Octaver"],
    EffectCategory.AMP: ["Amp Sim", "Cabinet"],
    EffectCategory.SPECIAL: ["Looper", "Granular", "Shimmer", "Freeze", "Slicer", "Bitcrusher"],
}

ALL_EFFECTS = {
    "Distortion": DistortionEffect,
    "Overdrive": OverdriveEffect,
    "Fuzz": FuzzEffect,
    "Chorus": ChorusEffect,
    "Flanger": FlangerEffect,
    "Phaser": PhaserEffect,
    "Tremolo": TremoloEffect,
    "Vibrato": VibratoEffect,
    "Rotary": RotaryEffect,
    "Ring Mod": RingModEffect,
    "Univibe": UnivibeEffect,
    "Delay": DelayEffect,
    "Reverb": ReverbEffect,
    "Compressor": CompressorEffect,
    "Noise Gate": NoiseGateEffect,
    "EQ": EQEffect,
    "Wah": WahEffect,
    "Filter": FilterEffect,
    "Pitch Shift": PitchShiftEffect,
    "Harmonizer": HarmonizerEffect,
    "Octaver": OctaverEffect,
    "Amp Sim": AmpSimEffect,
    "Cabinet": CabinetSimEffect,
    "Looper": LooperEffect,
    "Granular": GranularEffect,
    "Shimmer": ShimmerEffect,
    "Freeze": FreezeEffect,
    "Slicer": SlicerEffect,
    "Bitcrusher": BitcrusherEffect,
}

# Emotion to effect mapping for DAiW integration
EMOTION_EFFECT_MAP = {
    "grief": {
        "reverb": {"decay": 5.0, "mix": 0.6, "type": "hall"},
        "delay": {"time": 500, "feedback": 0.5, "mix": 0.4},
        "filter": {"cutoff": 800, "resonance": 0.3},
        "tremolo": {"rate": 2.0, "depth": 0.3},
    },
    "rage": {
        "distortion": {"drive": 0.9, "tone": 0.7, "type": "fuzz"},
        "compressor": {"threshold": -30, "ratio": 8},
        "eq": {"low_mid_gain": 3, "high_mid_gain": 4},
        "noise_gate": {"threshold": -40},
    },
    "fear": {
        "tremolo": {"rate": 8.0, "depth": 0.7},
        "phaser": {"rate": 0.3, "depth": 0.8, "feedback": 0.6},
        "reverb": {"decay": 3.0, "mix": 0.5, "type": "chamber"},
        "filter": {"cutoff": 2000, "lfo_amount": 0.5, "lfo_rate": 4.0},
    },
    "nostalgia": {
        "chorus": {"rate": 0.8, "depth": 0.5, "mix": 0.4},
        "delay": {"time": 300, "feedback": 0.3, "type": "tape"},
        "reverb": {"decay": 2.0, "mix": 0.3, "type": "spring"},
        "eq": {"high_shelf_gain": -3, "low_shelf_gain": 2},
    },
    "defiance": {
        "distortion": {"drive": 0.7, "tone": 0.8, "type": "tube"},
        "compressor": {"threshold": -20, "ratio": 4},
        "delay": {"time": 200, "feedback": 0.2, "mix": 0.2},
        "eq": {"high_mid_gain": 5, "presence": 0.7},
    },
    "tenderness": {
        "chorus": {"rate": 0.5, "depth": 0.3, "mix": 0.3},
        "reverb": {"decay": 2.5, "mix": 0.4, "type": "room"},
        "eq": {"high_shelf_gain": -2, "low_shelf_gain": -1},
        "compressor": {"threshold": -15, "ratio": 2, "attack": 30},
    },
    "dissociation": {
        "shimmer": {"decay": 10.0, "shimmer": 0.7, "pitch": 12},
        "granular": {"grain_size": 200, "density": 0.3, "pitch_random": 5},
        "reverb": {"decay": 8.0, "mix": 0.7, "type": "infinite"},
        "filter": {"cutoff": 1500, "resonance": 0.5},
    },
    "awe": {
        "shimmer": {"decay": 8.0, "shimmer": 0.5, "pitch": 12},
        "reverb": {"decay": 6.0, "mix": 0.5, "type": "cathedral"},
        "chorus": {"rate": 0.3, "depth": 0.4, "voices": 4},
        "delay": {"time": 400, "feedback": 0.4, "mix": 0.3},
    },
    "confusion": {
        "phaser": {"rate": 0.5, "depth": 0.9, "feedback": 0.7},
        "flanger": {"rate": 0.2, "depth": 0.8, "feedback": 0.6},
        "delay": {"time": 333, "feedback": 0.5, "type": "ping_pong"},
        "bitcrusher": {"bits": 8, "sample_rate": 22050, "mix": 0.3},
    },
    "longing": {
        "reverb": {"decay": 4.0, "mix": 0.5, "type": "hall"},
        "delay": {"time": 450, "feedback": 0.4, "mix": 0.35},
        "chorus": {"rate": 0.6, "depth": 0.4},
        "filter": {"cutoff": 3000, "env_amount": 0.3},
    },
}


# =============================================================================
# SIGNAL ROUTING
# =============================================================================

@dataclass
class SignalPath:
    """A single signal path in the routing matrix."""
    source: str  # "input", effect name, or "aux_N"
    destination: str  # Effect name or "output"
    gain: float = 1.0
    pan: float = 0.5  # 0 = left, 1 = right
    enabled: bool = True


class SignalRouter:
    """
    Flexible signal routing with parallel/serial support.
    """
    
    def __init__(self):
        self.paths: List[SignalPath] = []
        self.aux_sends: Dict[str, float] = {}  # aux_N -> level
        self.aux_returns: Dict[str, float] = {}
        self.master_volume: float = 1.0
        
    def add_path(self, source: str, destination: str, gain: float = 1.0):
        """Add a signal path."""
        path = SignalPath(source=source, destination=destination, gain=gain)
        self.paths.append(path)
        return path
    
    def remove_path(self, source: str, destination: str):
        """Remove a signal path."""
        self.paths = [p for p in self.paths if not (p.source == source and p.destination == destination)]
    
    def create_serial_chain(self, effects: List[str]):
        """Create a simple serial effect chain."""
        self.paths.clear()
        
        if not effects:
            self.add_path("input", "output")
            return
        
        # Input -> first effect
        self.add_path("input", effects[0])
        
        # Chain effects
        for i in range(len(effects) - 1):
            self.add_path(effects[i], effects[i + 1])
        
        # Last effect -> output
        self.add_path(effects[-1], "output")
    
    def create_parallel_chain(self, effects: List[str], blend: float = 0.5):
        """Create parallel effect paths."""
        self.paths.clear()
        
        for effect in effects:
            self.add_path("input", effect, gain=blend / len(effects))
            self.add_path(effect, "output")
        
        # Dry path
        self.add_path("input", "output", gain=1 - blend)
    
    def to_dict(self) -> Dict:
        return {
            "paths": [{"source": p.source, "destination": p.destination, "gain": p.gain, "pan": p.pan} for p in self.paths],
            "aux_sends": self.aux_sends,
            "aux_returns": self.aux_returns,
            "master_volume": self.master_volume,
        }
    
    def from_dict(self, data: Dict):
        self.paths = [SignalPath(**p) for p in data.get("paths", [])]
        self.aux_sends = data.get("aux_sends", {})
        self.aux_returns = data.get("aux_returns", {})
        self.master_volume = data.get("master_volume", 1.0)


# =============================================================================
# MODULATION MATRIX
# =============================================================================

@dataclass
class ModulationRoute:
    """A single modulation routing."""
    source_name: str  # Name of ModulationSource
    target_effect: str  # Effect name
    target_param: str  # Parameter name
    amount: float = 0.5  # -1 to 1
    enabled: bool = True


class ModulationMatrix:
    """
    Full modulation matrix - any source can modulate any parameter.
    """
    
    def __init__(self):
        self.sources: Dict[str, ModulationSource] = {}
        self.routes: List[ModulationRoute] = []
        
        # Create default sources
        self._create_default_sources()
    
    def _create_default_sources(self):
        """Create standard modulation sources."""
        # LFOs
        self.add_source(LFOSource("LFO1", rate=1.0, shape=WaveShape.SINE))
        self.add_source(LFOSource("LFO2", rate=0.5, shape=WaveShape.TRIANGLE))
        self.add_source(LFOSource("LFO3", rate=2.0, shape=WaveShape.SQUARE))
        self.add_source(LFOSource("LFO4", rate=0.25, shape=WaveShape.SAW_UP))
        
        # Envelope followers
        self.add_source(EnvelopeFollower("EnvFollow1", attack=0.01, release=0.1))
        self.add_source(EnvelopeFollower("EnvFollow2", attack=0.05, release=0.5))
        
        # Step sequencers
        self.add_source(StepSequencer("StepSeq1", num_steps=8))
        self.add_source(StepSequencer("StepSeq2", num_steps=16))
        
        # Random
        self.add_source(RandomSource("Random1", rate=1.0, smoothing=0.1))
        self.add_source(RandomSource("Random2", rate=4.0, smoothing=0.0))
        
        # Expression
        self.add_source(ExpressionInput("Expression"))
        
        # MIDI
        self.add_source(MIDISource("MIDI_CC1", cc_number=1))
        self.add_source(MIDISource("MIDI_CC11", cc_number=11))
        
        # Emotion (DAiW specific!)
        self.add_source(EmotionSource("Emotion"))
    
    def add_source(self, source: ModulationSource):
        """Add a modulation source."""
        self.sources[source.name] = source
    
    def remove_source(self, name: str):
        """Remove a modulation source."""
        if name in self.sources:
            del self.sources[name]
            self.routes = [r for r in self.routes if r.source_name != name]
    
    def add_route(
        self,
        source_name: str,
        target_effect: str,
        target_param: str,
        amount: float = 0.5
    ) -> ModulationRoute:
        """Add a modulation route."""
        route = ModulationRoute(
            source_name=source_name,
            target_effect=target_effect,
            target_param=target_param,
            amount=amount,
        )
        self.routes.append(route)
        return route
    
    def remove_route(self, source_name: str, target_effect: str, target_param: str):
        """Remove a modulation route."""
        self.routes = [
            r for r in self.routes
            if not (r.source_name == source_name and r.target_effect == target_effect and r.target_param == target_param)
        ]
    
    def update_sources(self, delta_time: float, **kwargs):
        """Update all modulation sources."""
        for source in self.sources.values():
            source.update(delta_time, **kwargs)
    
    def get_modulation(self, effect_name: str, param_name: str) -> float:
        """Get total modulation amount for a parameter."""
        total = 0.0
        
        for route in self.routes:
            if route.enabled and route.target_effect == effect_name and route.target_param == param_name:
                if route.source_name in self.sources:
                    total += self.sources[route.source_name].output * route.amount
        
        return max(-1, min(1, total))
    
    def to_dict(self) -> Dict:
        return {
            "sources": {name: {"type": type(s).__name__, "params": vars(s)} for name, s in self.sources.items()},
            "routes": [{"source": r.source_name, "effect": r.target_effect, "param": r.target_param, "amount": r.amount} for r in self.routes],
        }


# =============================================================================
# EFFECT CHAIN
# =============================================================================

class EffectChain:
    """
    A chain of effects with ordering and bypass.
    """
    
    def __init__(self, name: str = "Default"):
        self.name = name
        self.effects: Dict[str, BaseEffect] = {}
        self.order: List[str] = []
        self.global_bypass: bool = False
        self.input_level: float = 1.0
        self.output_level: float = 1.0
    
    def add_effect(self, effect: BaseEffect, position: Optional[int] = None):
        """Add an effect to the chain."""
        self.effects[effect.name] = effect
        
        if position is not None and 0 <= position <= len(self.order):
            self.order.insert(position, effect.name)
        else:
            self.order.append(effect.name)
    
    def remove_effect(self, name: str):
        """Remove an effect from the chain."""
        if name in self.effects:
            del self.effects[name]
            self.order = [n for n in self.order if n != name]
    
    def move_effect(self, name: str, new_position: int):
        """Move an effect to a new position."""
        if name in self.order:
            self.order.remove(name)
            self.order.insert(max(0, min(new_position, len(self.order))), name)
    
    def get_effect(self, name: str) -> Optional[BaseEffect]:
        """Get an effect by name."""
        return self.effects.get(name)
    
    def set_param(self, effect_name: str, param_name: str, value: float):
        """Set a parameter on an effect."""
        if effect_name in self.effects:
            self.effects[effect_name].set_param(param_name, value)
    
    def process(self, samples: List[float], sample_rate: int, modulation: Optional[ModulationMatrix] = None) -> List[float]:
        """Process audio through the chain."""
        if self.global_bypass:
            return samples
        
        # Apply input level
        output = [s * self.input_level for s in samples]
        
        # Process through each effect in order
        for effect_name in self.order:
            if effect_name in self.effects:
                effect = self.effects[effect_name]
                
                # Apply modulation to parameters
                if modulation:
                    for param_name, param in effect.parameters.items():
                        mod_amount = modulation.get_modulation(effect_name, param_name)
                        if mod_amount != 0:
                            # Apply modulation
                            mod_range = param.max_val - param.min_val
                            param.value = param.default + mod_amount * mod_range * param.mod_amount
                            param.value = max(param.min_val, min(param.max_val, param.value))
                
                output = effect.process(output, sample_rate)
        
        # Apply output level
        output = [s * self.output_level for s in output]
        
        return output
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "effects": {name: eff.to_dict() for name, eff in self.effects.items()},
            "order": self.order,
            "global_bypass": self.global_bypass,
            "input_level": self.input_level,
            "output_level": self.output_level,
        }
    
    def from_dict(self, data: Dict):
        self.name = data.get("name", "Default")
        self.order = data.get("order", [])
        self.global_bypass = data.get("global_bypass", False)
        self.input_level = data.get("input_level", 1.0)
        self.output_level = data.get("output_level", 1.0)
        
        # Recreate effects
        self.effects = {}
        for name, eff_data in data.get("effects", {}).items():
            effect_class = ALL_EFFECTS.get(name)
            if effect_class:
                effect = effect_class()
                effect.from_dict(eff_data)
                self.effects[name] = effect


# =============================================================================
# PRESET SYSTEM
# =============================================================================

@dataclass
class EffectPreset:
    """
    A complete effect preset including chain, modulation, and routing.
    """
    name: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    emotion: Optional[str] = None  # DAiW emotion association
    
    chain_data: Dict = field(default_factory=dict)
    modulation_data: Dict = field(default_factory=dict)
    routing_data: Dict = field(default_factory=dict)
    
    def save(self, path: str):
        """Save preset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "emotion": self.emotion,
            "chain": self.chain_data,
            "modulation": self.modulation_data,
            "routing": self.routing_data,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EffectPreset":
        """Load preset from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            emotion=data.get("emotion"),
            chain_data=data.get("chain", {}),
            modulation_data=data.get("modulation", {}),
            routing_data=data.get("routing", {}),
        )


# =============================================================================
# MAIN ENGINE
# =============================================================================

class GuitarFXEngine:
    """
    The complete guitar effects engine.
    
    Features:
    - 28+ effect types
    - Full modulation matrix
    - Flexible signal routing
    - Preset system
    - DAiW emotion integration
    """
    
    def __init__(self):
        self.chain = EffectChain("Main")
        self.modulation = ModulationMatrix()
        self.router = SignalRouter()
        
        self.tempo_bpm: float = 120.0
        self.sample_rate: int = 44100
        
        # Input/output
        self.input_level: float = 1.0
        self.output_level: float = 1.0
        self.noise_gate_active: bool = False
        
        # Tuner
        self.tuner_active: bool = False
        self.detected_pitch: float = 0.0
        self.detected_note: str = ""
        
        # Current emotion (DAiW)
        self.current_emotion: Optional[str] = None
        self.emotion_intensity: float = 1.0
    
    def add_effect(self, effect_type: str, position: Optional[int] = None) -> BaseEffect:
        """Add an effect by type name."""
        if effect_type not in ALL_EFFECTS:
            raise ValueError(f"Unknown effect type: {effect_type}")
        
        effect = ALL_EFFECTS[effect_type]()
        self.chain.add_effect(effect, position)
        return effect
    
    def remove_effect(self, name: str):
        """Remove an effect by name."""
        self.chain.remove_effect(name)
    
    def get_effect(self, name: str) -> Optional[BaseEffect]:
        """Get an effect by name."""
        return self.chain.get_effect(name)
    
    def set_param(self, effect_name: str, param_name: str, value: float):
        """Set a parameter on an effect."""
        self.chain.set_param(effect_name, param_name, value)
    
    def add_modulation(
        self,
        source_name: str,
        effect_name: str,
        param_name: str,
        amount: float = 0.5
    ):
        """Add a modulation route."""
        self.modulation.add_route(source_name, effect_name, param_name, amount)
    
    def set_emotion(self, emotion: str, intensity: float = 1.0):
        """
        Set the current emotion for DAiW-aware processing.
        This affects the Emotion modulation source and can trigger preset suggestions.
        """
        self.current_emotion = emotion.lower()
        self.emotion_intensity = intensity
        
        # Update emotion modulation source
        if "Emotion" in self.modulation.sources:
            emotion_source = self.modulation.sources["Emotion"]
            if isinstance(emotion_source, EmotionSource):
                emotion_source.set_emotion(emotion, intensity)
    
    def get_emotion_suggestions(self) -> Dict[str, Any]:
        """Get effect suggestions based on current emotion."""
        if not self.current_emotion:
            return {}
        
        return EMOTION_EFFECT_MAP.get(self.current_emotion, {})
    
    def apply_emotion_preset(self, emotion: str, blend: float = 1.0):
        """
        Apply effect settings based on emotion.
        This is the DAiW magic - emotion → effects mapping.
        """
        suggestions = EMOTION_EFFECT_MAP.get(emotion.lower(), {})
        
        for effect_type, params in suggestions.items():
            # Add effect if not present
            effect_name = effect_type.replace("_", " ").title()
            if effect_name not in self.chain.effects:
                try:
                    self.add_effect(effect_name)
                except ValueError:
                    continue
            
            # Set parameters with blend
            effect = self.chain.get_effect(effect_name)
            if effect:
                for param_name, value in params.items():
                    if param_name == "type":
                        continue  # Handle type separately
                    if param_name in effect.parameters:
                        current = effect.parameters[param_name].value
                        blended = current + (value - current) * blend
                        effect.set_param(param_name, blended)
    
    def process(self, samples: List[float]) -> List[float]:
        """Process audio through the complete effects chain."""
        # Update modulation sources
        delta_time = len(samples) / self.sample_rate
        self.modulation.update_sources(
            delta_time,
            tempo_bpm=self.tempo_bpm,
            emotion=self.current_emotion,
            intensity=self.emotion_intensity,
        )
        
        # Apply input level
        samples = [s * self.input_level for s in samples]
        
        # Process through chain
        output = self.chain.process(samples, self.sample_rate, self.modulation)
        
        # Apply output level
        output = [s * self.output_level for s in output]
        
        return output
    
    def save_preset(self, name: str, path: str, description: str = ""):
        """Save current state as a preset."""
        preset = EffectPreset(
            name=name,
            description=description,
            emotion=self.current_emotion,
            chain_data=self.chain.to_dict(),
            modulation_data=self.modulation.to_dict(),
            routing_data=self.router.to_dict(),
        )
        preset.save(path)
        return preset
    
    def load_preset(self, path: str):
        """Load a preset from file."""
        preset = EffectPreset.load(path)
        
        self.chain.from_dict(preset.chain_data)
        # Modulation and routing would need similar from_dict methods
        
        if preset.emotion:
            self.set_emotion(preset.emotion)
        
        return preset
    
    def get_state(self) -> Dict:
        """Get complete engine state."""
        return {
            "tempo_bpm": self.tempo_bpm,
            "sample_rate": self.sample_rate,
            "input_level": self.input_level,
            "output_level": self.output_level,
            "current_emotion": self.current_emotion,
            "emotion_intensity": self.emotion_intensity,
            "chain": self.chain.to_dict(),
            "modulation": self.modulation.to_dict(),
            "routing": self.router.to_dict(),
        }
    
    def list_effects(self) -> List[str]:
        """List all available effect types."""
        return list(ALL_EFFECTS.keys())
    
    def list_effects_by_category(self) -> Dict[str, List[str]]:
        """List effects organized by category."""
        return {cat.value: effects for cat, effects in EFFECT_CATEGORIES.items()}
    
    def get_effect_params(self, effect_name: str) -> Dict[str, Parameter]:
        """Get all parameters for an effect."""
        effect = self.chain.get_effect(effect_name)
        if effect:
            return effect.parameters
        return {}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_preset_from_emotion(emotion: str, intensity: float = 1.0) -> EffectPreset:
    """
    Create a complete preset based on emotion.
    This is the core DAiW → effects translation.
    """
    engine = GuitarFXEngine()
    engine.apply_emotion_preset(emotion, intensity)
    
    preset = EffectPreset(
        name=f"{emotion.title()} Preset",
        description=f"Auto-generated preset for {emotion} emotion at {intensity:.0%} intensity",
        tags=[emotion, "auto-generated", "daiw"],
        emotion=emotion,
        chain_data=engine.chain.to_dict(),
        modulation_data=engine.modulation.to_dict(),
        routing_data=engine.router.to_dict(),
    )
    
    return preset


def get_effect_suggestions(emotion: str) -> List[Dict[str, Any]]:
    """Get effect suggestions for an emotion."""
    suggestions = EMOTION_EFFECT_MAP.get(emotion.lower(), {})
    
    result = []
    for effect_type, params in suggestions.items():
        result.append({
            "effect": effect_type,
            "params": params,
            "reason": f"Complements {emotion} emotional quality",
        })
    
    return result

