"""
DAW Integration - Logic Pro and other DAW integration utilities.

Provides bridges for working with different DAW file formats,
MIDI export/import, DAW-specific features, timeline markers, and
comprehensive DAW functions reference.
"""

from music_brain.daw.logic import (
    LogicProject,
    export_to_logic,
    import_from_logic,
    create_logic_template,
)
from music_brain.daw.markers import (
    MarkerEvent,
    EmotionalSection,
    export_markers_midi,
    export_sections_midi,
    merge_markers_with_midi,
    get_standard_structure,
    get_emotional_structure,
)
from music_brain.daw.functions import (
    Transport,
    TransportState,
    Track,
    TrackType,
    MIDINote,
    MIDIEditor,
    MixerChannel,
    DAWProject,
    QuantizeValue,
    get_daw_function_reference,
    get_transport_reference,
    get_track_reference,
    get_midi_editing_reference,
    get_mixing_reference,
    get_project_reference,
    DAW_FUNCTIONS_REFERENCE,
)

__all__ = [
    # Logic integration
    "LogicProject",
    "export_to_logic",
    "import_from_logic",
    "create_logic_template",
    # Markers
    "MarkerEvent",
    "EmotionalSection",
    "export_markers_midi",
    "export_sections_midi",
    "merge_markers_with_midi",
    "get_standard_structure",
    "get_emotional_structure",
    # DAW Functions Reference
    "Transport",
    "TransportState",
    "Track",
    "TrackType",
    "MIDINote",
    "MIDIEditor",
    "MixerChannel",
    "DAWProject",
    "QuantizeValue",
    "get_daw_function_reference",
    "get_transport_reference",
    "get_track_reference",
    "get_midi_editing_reference",
    "get_mixing_reference",
    "get_project_reference",
    "DAW_FUNCTIONS_REFERENCE",
]
