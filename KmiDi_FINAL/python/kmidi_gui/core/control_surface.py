"""External control surface support for MIDI CC/Note mapping.

Supports MIDI CC and Note messages mapped to application parameters.
Includes learn mode for easy mapping setup.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class MIDIType(str, Enum):
    """MIDI message type."""
    CC = "cc"
    NOTE = "note"


@dataclass
class ControlMapping:
    """Mapping from MIDI control to application parameter.

    Maps a MIDI CC or Note to a parameter with value range.
    """

    parameter: str  # e.g., "intent.vulnerability_scale"
    midi_type: str  # "cc" or "note"
    channel: int  # 0-15
    cc_number: Optional[int] = None  # For CC messages
    note_number: Optional[int] = None  # For Note messages
    min_value: float = 0.0
    max_value: float = 1.0
    device_id: str = ""  # MIDI device identifier

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "parameter": self.parameter,
            "midi_type": self.midi_type,
            "channel": self.channel,
            "cc_number": self.cc_number,
            "note_number": self.note_number,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "device_id": self.device_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ControlMapping":
        """Create from dictionary."""
        return cls(
            parameter=data["parameter"],
            midi_type=data["midi_type"],
            channel=data["channel"],
            cc_number=data.get("cc_number"),
            note_number=data.get("note_number"),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            device_id=data.get("device_id", ""),
        )

    def matches(self, midi_type: str, channel: int, cc_or_note: int,
                 device_id: str = "") -> bool:
        """Check if MIDI message matches this mapping.

        Args:
            midi_type: "cc" or "note"
            channel: MIDI channel (0-15)
            cc_or_note: CC number or note number
            device_id: Device ID (optional, matches any if empty)

        Returns:
            True if message matches
        """
        if self.midi_type != midi_type:
            return False
        if self.channel != channel:
            return False
        if device_id and self.device_id and device_id != self.device_id:
            return False

        if midi_type == "cc":
            return self.cc_number == cc_or_note
        else:
            return self.note_number == cc_or_note


class ControlSurfaceManager:
    """Manages MIDI input devices and parameter mappings.

    Handles device discovery, connection, message routing, and learn mode.
    """

    def __init__(self):
        """Initialize control surface manager."""
        self.mappings: List[ControlMapping] = []
        self.learn_mode_active = False
        self.learn_parameter: Optional[str] = None
        self.learn_callback: Optional[Callable[[ControlMapping], None]] = None

        # MIDI input will be initialized when rtmidi is available
        self.midi_input = None
        self._init_midi()

        # Parameter update callbacks
        self.parameter_callbacks: Dict[str, Callable[[float], None]] = {}

    def _init_midi(self):
        """Initialize MIDI input (if rtmidi is available)."""
        try:
            import rtmidi  # type: ignore[import]
            self.midi_input = rtmidi.MidiIn()
            logger.info("MIDI input initialized")
        except ImportError:
            logger.warning(
                "python-rtmidi not available, control surface disabled")
            self.midi_input = None

    def discover_devices(self) -> List[Dict[str, str]]:
        """Discover available MIDI input devices.

        Returns:
            List of device info dictionaries with 'id' and 'name' keys
        """
        if not self.midi_input:
            return []

        try:
            devices = []
            port_count = self.midi_input.get_port_count()
            for i in range(port_count):
                name = self.midi_input.get_port_name(i)
                devices.append({"id": str(i), "name": name})
            return devices
        except Exception as e:
            logger.error(f"Failed to discover MIDI devices: {e}")
            return []

    def connect_device(self, device_id: str) -> bool:
        """Connect to MIDI input device.

        Args:
            device_id: Device identifier from discover_devices()

        Returns:
            True if connected successfully
        """
        if not self.midi_input:
            logger.warning("MIDI input not available")
            return False

        try:
            port_index = int(device_id)
            self.midi_input.open_port(port_index)
            self.midi_input.set_callback(self._on_midi_message)
            logger.info(f"Connected to MIDI device: {device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MIDI device {device_id}: {e}")
            return False

    def disconnect_device(self, device_id: str) -> None:
        """Disconnect from MIDI input device.

        Args:
            device_id: Device identifier
        """
        if not self.midi_input:
            return

        try:
            self.midi_input.close_port()
            logger.info(f"Disconnected from MIDI device: {device_id}")
        except Exception as e:
            logger.error(f"Failed to disconnect from MIDI device: {e}")

    def start_learn_mode(self, parameter: str, callback: Optional[Callable[[ControlMapping], None]] = None) -> None:
        """Start learn mode for a parameter.

        When learn mode is active, the next MIDI message will be mapped to the parameter.

        Args:
            parameter: Parameter to map (e.g., "intent.vulnerability_scale")
            callback: Optional callback when mapping is learned
        """
        self.learn_mode_active = True
        self.learn_parameter = parameter
        self.learn_callback = callback
        logger.info(f"Learn mode started for parameter: {parameter}")

    def stop_learn_mode(self) -> None:
        """Stop learn mode."""
        self.learn_mode_active = False
        self.learn_parameter = None
        self.learn_callback = None
        logger.info("Learn mode stopped")

    def add_mapping(self, mapping: ControlMapping) -> None:
        """Add parameter mapping.

        Args:
            mapping: Control mapping to add
        """
        # Remove existing mapping for same parameter if any
        self.mappings = [m for m in self.mappings if m.parameter != mapping.parameter]
        self.mappings.append(mapping)
        logger.info(
            f"Added mapping: {mapping.parameter} <- "
            f"{mapping.midi_type} {mapping.cc_number or mapping.note_number}")

    def remove_mapping(self, parameter: str) -> bool:
        """Remove parameter mapping.

        Args:
            parameter: Parameter to unmap

        Returns:
            True if mapping was removed
        """
        before_count = len(self.mappings)
        self.mappings = [m for m in self.mappings if m.parameter != parameter]
        removed = len(self.mappings) < before_count
        if removed:
            logger.info(f"Removed mapping for parameter: {parameter}")
        return removed

    def get_mapping(self, parameter: str) -> Optional[ControlMapping]:
        """Get mapping for parameter.

        Args:
            parameter: Parameter name

        Returns:
            Mapping or None if not found
        """
        for mapping in self.mappings:
            if mapping.parameter == parameter:
                return mapping
        return None

    def register_parameter_callback(self, parameter: str, callback: Callable[[float], None]) -> None:
        """Register callback for parameter updates.

        Args:
            parameter: Parameter name
            callback: Function to call with new value
        """
        self.parameter_callbacks[parameter] = callback

    def _on_midi_message(self, message_data):
        """Handle incoming MIDI message.

        Args:
            message_data: Tuple of (message, delta_time) from rtmidi
        """
        if not message_data:
            return

        message, _ = message_data

        if len(message) < 2:
            return

        status = message[0]
        channel = status & 0x0F
        message_type = (status >> 4) & 0x0F

        # Handle learn mode
        if self.learn_mode_active and self.learn_parameter:
            if message_type == 0xB:  # CC
                cc_number = message[1]
                mapping = ControlMapping(
                    parameter=self.learn_parameter,
                    midi_type="cc",
                    channel=channel,
                    cc_number=cc_number,
                    min_value=0.0,
                    max_value=1.0,
                )
                self.add_mapping(mapping)
                if self.learn_callback:
                    self.learn_callback(mapping)
                self.stop_learn_mode()
                return
            elif message_type == 0x9:  # Note On
                note_number = message[1]
                mapping = ControlMapping(
                    parameter=self.learn_parameter,
                    midi_type="note",
                    channel=channel,
                    note_number=note_number,
                    min_value=0.0,
                    max_value=1.0,
                )
                self.add_mapping(mapping)
                if self.learn_callback:
                    self.learn_callback(mapping)
                self.stop_learn_mode()
                return

        # Handle normal message routing
        device_id = ""  # TODO: Get actual device ID from rtmidi

        if message_type == 0xB:  # CC
            cc_number = message[1]
            value = message[2] if len(message) > 2 else 0

            # Find matching mapping
            for mapping in self.mappings:
                if mapping.matches("cc", channel, cc_number, device_id):
                    # Map MIDI value (0-127) to parameter range
                    normalized = (value / 127.0) * (mapping.max_value - mapping.min_value) + mapping.min_value
                    self._update_parameter(mapping.parameter, normalized)
                    return

        elif message_type == 0x9:  # Note On
            note_number = message[1]
            velocity = message[2] if len(message) > 2 else 0

            # Find matching mapping
            for mapping in self.mappings:
                if mapping.matches("note", channel, note_number, device_id):
                    # Map velocity (0-127) to parameter range
                    normalized = (velocity / 127.0) * (mapping.max_value - mapping.min_value) + mapping.min_value
                    self._update_parameter(mapping.parameter, normalized)
                    return

    def _update_parameter(self, parameter: str, value: float) -> None:
        """Update parameter value.

        Args:
            parameter: Parameter name
            value: New value
        """
        if parameter in self.parameter_callbacks:
            try:
                self.parameter_callbacks[parameter](value)
                logger.debug(f"Updated parameter {parameter} = {value}")
            except Exception as e:
                logger.error(f"Failed to update parameter {parameter}: {e}")

    def save_mappings(self, project_path: Path) -> None:
        """Save mappings to project file.

        Args:
            project_path: Path to project file or directory
        """
        if project_path.is_file():
            project_path = project_path.parent

        mappings_file = project_path / "control_mappings.json"

        try:
            data = {
                "mappings": [m.to_dict() for m in self.mappings],
                "version": "1.0.0",
            }
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved control mappings to {mappings_file}")
        except Exception as e:
            logger.error(f"Failed to save control mappings: {e}")

    def load_mappings(self, project_path: Path) -> None:
        """Load mappings from project file.

        Args:
            project_path: Path to project file or directory
        """
        if project_path.is_file():
            project_path = project_path.parent

        mappings_file = project_path / "control_mappings.json"

        if not mappings_file.exists():
            logger.debug(f"No control mappings file found: {mappings_file}")
            return

        try:
            with open(mappings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.mappings = [
                ControlMapping.from_dict(m) for m in data.get("mappings", [])]
            logger.info(
                f"Loaded {len(self.mappings)} control mappings from {mappings_file}")
        except Exception as e:
            logger.error(f"Failed to load control mappings: {e}")
