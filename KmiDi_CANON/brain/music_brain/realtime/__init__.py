"""
Real-time MIDI Processing - Live MIDI input/output and analysis.

This module provides real-time MIDI processing capabilities:
- MIDI input capture from hardware/software ports
- Real-time chord detection
- Real-time groove analysis
- MIDI transformation and routing
- MIDI output to hardware/software ports
- Realtime engine for streaming note events
- Tempo-aware clock for musical timing
- Event scheduling for lookahead processing
"""

# Core MIDI processor
from music_brain.realtime.midi_processor import (
    RealtimeMidiProcessor,
    MidiProcessorConfig,
    ChordDetectionCallback,
    GrooveAnalysisCallback,
    MidiTransformCallback,
)

# Transformers
from music_brain.realtime.transformers import (
    create_transpose_transformer,
    create_velocity_scale_transformer,
    create_chord_generator_transformer,
    create_arpeggiator_transformer,
    create_humanize_transformer,
    create_channel_router_transformer,
    create_filter_transformer,
)

# Realtime engine components
from music_brain.realtime.engine import RealtimeEngine
from music_brain.realtime.clock import RealtimeClock, ClockSnapshot
from music_brain.realtime.scheduler import EventScheduler
from music_brain.realtime.events import (
    ScheduledEvent,
    ControlEvent,
    ControlEventType,
    MetricEvent,
)
from music_brain.realtime.transport import BaseTransport, MidiTransport, OscTransport

__all__ = [
    # Core MIDI processor
    "RealtimeMidiProcessor",
    "MidiProcessorConfig",
    "ChordDetectionCallback",
    "GrooveAnalysisCallback",
    "MidiTransformCallback",
    # Transformers
    "create_transpose_transformer",
    "create_velocity_scale_transformer",
    "create_chord_generator_transformer",
    "create_arpeggiator_transformer",
    "create_humanize_transformer",
    "create_channel_router_transformer",
    "create_filter_transformer",
    # Realtime engine
    "RealtimeEngine",
    "RealtimeClock",
    "ClockSnapshot",
    "EventScheduler",
    # Events
    "ScheduledEvent",
    "ControlEvent",
    "ControlEventType",
    "MetricEvent",
    # Transport
    "BaseTransport",
    "MidiTransport",
    "OscTransport",
]

