"""
Kelly Companion Engines

All music generation engines for the Kelly MIDI Companion system.
"""

# Import only the engines that exist
from .bass_engine import BassEngine, BassConfig, BassOutput
from .counter_melody_engine import CounterMelodyEngine, CounterMelodyConfig, CounterMelodyOutput
from .dynamics_engine import DynamicsEngine, DynamicsProfile
from .fill_engine import FillEngine, FillConfig, FillOutput
from .melody_engine import MelodyEngine, MelodyConfig, MelodyOutput
from .orchestration import OrchestrationEngine, OrchestrationSession
from .pad_engine import PadEngine, PadConfig, PadOutput
from .rhythm_engine import RhythmEngine, RhythmConfig, RhythmOutput
from .string_engine import StringEngine, StringConfig, StringOutput
from .tension_engine import TensionEngine, TensionProfile
from .transition_engine import TransitionEngine, TransitionPlan
from .variation_engine import VariationEngine, VariationPlan

__all__ = [
    # Engines
    "BassEngine",
    "CounterMelodyEngine",
    "DynamicsEngine",
    "FillEngine",
    "MelodyEngine",
    "OrchestrationEngine",
    "PadEngine",
    "RhythmEngine",
    "StringEngine",
    "TensionEngine",
    "TransitionEngine",
    "VariationEngine",
    # Configs
    "BassConfig",
    "CounterMelodyConfig",
    "FillConfig",
    "MelodyConfig",
    "PadConfig",
    "RhythmConfig",
    "StringConfig",
    # Outputs
    "BassOutput",
    "CounterMelodyOutput",
    "FillOutput",
    "MelodyOutput",
    "PadOutput",
    "RhythmOutput",
    "StringOutput",
    # Other
    "OrchestrationSession",
    "DynamicsProfile",
    "TensionProfile",
    "TransitionPlan",
    "VariationPlan",
]
