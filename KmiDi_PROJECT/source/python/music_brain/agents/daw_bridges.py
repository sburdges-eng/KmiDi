#!/usr/bin/env python3
"""
DAW Bridge Implementations.

Concrete implementations of DAWProtocol for different DAWs:
- AbletonDAWBridge: Wraps existing AbletonBridge
- LogicProBridge: AppleScript + MIDI for Logic Pro X
- ReaperBridge: OSC (via ReaOSC) for Reaper
- BitwigBridge: OSC for Bitwig Studio

Each bridge auto-registers with the DAWRegistry on import.
"""

from __future__ import annotations

import platform
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .daw_protocol import (
    BaseDAWBridge,
    DAWCapabilities,
    DAWRegistry,
    DAWType,
    TrackInfo,
    TransportState,
)


# =============================================================================
# Voice Control CC Mappings (shared across bridges)
# =============================================================================


class VoiceCC:
    """MIDI CC mappings for voice synthesis control."""

    VOWEL = 20
    FORMANT_SHIFT = 21
    BREATHINESS = 22
    VIBRATO_RATE = 23
    VIBRATO_DEPTH = 24
    PITCH_BEND = 25
    JITTER = 26
    SHIMMER = 27
    NASALITY = 28


VOWEL_FORMANTS = {
    "A": (800, 1200),
    "E": (400, 2200),
    "I": (300, 2800),
    "O": (500, 900),
    "U": (350, 700),
}


# =============================================================================
# Ableton Bridge (wraps existing implementation)
# =============================================================================


@dataclass
class AbletonConfig:
    """Ableton bridge configuration."""

    osc_host: str = "127.0.0.1"
    osc_send_port: int = 9000
    osc_receive_port: int = 9001
    midi_port: str = "DAiW Voice"
    virtual_midi: bool = True


class AbletonDAWBridge(BaseDAWBridge):
    """
    DAW bridge for Ableton Live.

    Wraps the existing AbletonBridge to conform to DAWProtocol.
    Uses OSC for transport control and MIDI for notes/CC.
    """

    def __init__(self, config: Optional[AbletonConfig] = None):
        super().__init__()
        self.config = config or AbletonConfig()

        # Lazy import to avoid circular dependency
        self._osc_bridge = None
        self._midi_bridge = None

        self._capabilities = DAWCapabilities(
            has_transport=True,
            has_tempo_control=True,
            has_loop_control=True,
            can_create_tracks=True,
            can_arm_tracks=True,
            can_mute_solo=True,
            has_clip_launcher=True,  # Ableton's signature feature
            has_arrangement=True,
            has_midi_output=True,
            has_midi_input=True,
            has_osc=True,
            has_automation=True,
            has_markers=True,
            can_control_plugins=False,
        )

    @property
    def daw_type(self) -> DAWType:
        return DAWType.ABLETON

    def _do_connect(self) -> bool:
        """Connect to Ableton via OSC and MIDI."""
        from .ableton_bridge import (
            AbletonMIDIBridge,
            AbletonOSCBridge,
            MIDIConfig,
            OSCConfig,
        )

        osc_config = OSCConfig(
            host=self.config.osc_host,
            send_port=self.config.osc_send_port,
            receive_port=self.config.osc_receive_port,
        )
        midi_config = MIDIConfig(
            output_port=self.config.midi_port,
            virtual=self.config.virtual_midi,
        )

        self._osc_bridge = AbletonOSCBridge(osc_config)
        self._midi_bridge = AbletonMIDIBridge(midi_config)

        osc_ok = self._osc_bridge.connect()
        midi_ok = self._midi_bridge.connect()

        return osc_ok or midi_ok

    def _do_disconnect(self) -> None:
        if self._midi_bridge:
            self._midi_bridge.disconnect()
        if self._osc_bridge:
            self._osc_bridge.disconnect()

    # Transport
    def play(self) -> None:
        if self._osc_bridge:
            self._osc_bridge.play()

    def stop(self) -> None:
        if self._osc_bridge:
            self._osc_bridge.stop()

    def record(self) -> None:
        if self._osc_bridge:
            self._osc_bridge.record()

    def set_tempo(self, bpm: float) -> None:
        if self._osc_bridge:
            self._osc_bridge.set_tempo(bpm)

    def set_position(self, bars: float, beats: float = 0.0) -> None:
        if self._osc_bridge:
            self._osc_bridge.set_position(bars, beats)

    # Track control
    def create_track(self, track_type: str = "midi", name: str = "") -> int:
        if self._osc_bridge:
            return self._osc_bridge.create_track(track_type)
        return -1

    def arm_track(self, index: int, armed: bool = True) -> None:
        if self._osc_bridge:
            self._osc_bridge.arm_track(index, armed)

    def mute_track(self, index: int, muted: bool = True) -> None:
        if self._osc_bridge:
            self._osc_bridge.mute_track(index, muted)

    def solo_track(self, index: int, soloed: bool = True) -> None:
        if self._osc_bridge:
            self._osc_bridge.solo_track(index, soloed)

    def set_track_volume(self, index: int, volume_db: float) -> None:
        if self._osc_bridge:
            self._osc_bridge.set_track_volume(index, volume_db)

    def set_track_pan(self, index: int, pan: float) -> None:
        if self._osc_bridge:
            self._osc_bridge.set_track_pan(index, pan)

    # Clips
    def fire_clip(self, track: int, clip: int) -> None:
        if self._osc_bridge:
            self._osc_bridge.fire_clip(track, clip)

    def stop_clip(self, track: int, clip: int) -> None:
        if self._osc_bridge:
            self._osc_bridge.stop_clip(track, clip)

    # MIDI
    def send_note_on(self, note: int, velocity: int = 100, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.send_note_on(note, velocity, channel)

    def send_note_off(self, note: int, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.send_note_off(note, channel)

    def send_note(
        self,
        note: int,
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        if self._midi_bridge:
            self._midi_bridge.send_note(note, velocity, duration_ms, channel)

    def send_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        if self._midi_bridge:
            self._midi_bridge.send_chord(notes, velocity, duration_ms, channel)

    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.send_cc(cc, value, channel)

    def send_pitch_bend(self, value: int, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.send_pitch_bend(value, channel)

    def all_notes_off(self, channel: Optional[int] = None) -> None:
        if self._midi_bridge:
            self._midi_bridge.all_notes_off(channel)

    # Voice control
    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.set_vowel(vowel, channel)

    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.set_breathiness(amount, channel)

    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        if self._midi_bridge:
            self._midi_bridge.set_vibrato(rate, depth, channel)


# =============================================================================
# Logic Pro Bridge (AppleScript + MIDI)
# =============================================================================


@dataclass
class LogicProConfig:
    """Logic Pro bridge configuration."""

    midi_port: str = "DAiW Voice"
    virtual_midi: bool = True
    use_midi_for_transport: bool = False  # Use AppleScript by default


class LogicProBridge(BaseDAWBridge):
    """
    DAW bridge for Logic Pro X.

    Uses AppleScript for transport control and MIDI for notes/CC.
    macOS only.
    """

    def __init__(self, config: Optional[LogicProConfig] = None):
        super().__init__()
        self.config = config or LogicProConfig()
        self._midi_output = None

        self._capabilities = DAWCapabilities(
            has_transport=True,
            has_tempo_control=True,
            has_loop_control=True,
            can_create_tracks=True,
            can_arm_tracks=True,
            can_mute_solo=True,
            has_clip_launcher=False,  # Logic doesn't have clip launcher
            has_arrangement=True,
            has_midi_output=True,
            has_midi_input=True,
            has_osc=False,  # Logic uses AppleScript, not OSC
            has_automation=True,
            has_markers=True,
            can_control_plugins=True,
        )

    @property
    def daw_type(self) -> DAWType:
        return DAWType.LOGIC_PRO

    def _do_connect(self) -> bool:
        if platform.system() != "Darwin":
            print("Logic Pro bridge requires macOS")
            return False

        # Check if Logic Pro is running
        if not self._is_logic_running():
            print("Logic Pro is not running")
            return False

        # Connect MIDI
        try:
            import mido

            if self.config.virtual_midi:
                self._midi_output = mido.open_output(
                    self.config.midi_port, virtual=True
                )
            else:
                ports = mido.get_output_names()
                matching = [p for p in ports if self.config.midi_port in p]
                if matching:
                    self._midi_output = mido.open_output(matching[0])

            return self._midi_output is not None

        except ImportError:
            print("mido not installed. Run: pip install mido python-rtmidi")
            return False
        except Exception as e:
            print(f"MIDI connection failed: {e}")
            return False

    def _do_disconnect(self) -> None:
        if self._midi_output:
            try:
                self.all_notes_off()
                time.sleep(0.05)
                self._midi_output.close()
            except Exception:
                pass
            self._midi_output = None

    def _is_logic_running(self) -> bool:
        """Check if Logic Pro is running."""
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to (name of processes) contains "Logic Pro"',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "true" in result.stdout.lower()
        except Exception:
            return False

    def _run_applescript(self, script: str) -> Optional[str]:
        """Run AppleScript and return result."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            print(f"AppleScript error: {e}")
            return None

    # Transport (AppleScript)
    def play(self) -> None:
        self._run_applescript(
            'tell application "Logic Pro" to set playhead state to playing'
        )

    def stop(self) -> None:
        self._run_applescript(
            'tell application "Logic Pro" to set playhead state to stopped'
        )

    def record(self) -> None:
        # Logic uses key commands for record
        self._run_applescript(
            '''
            tell application "Logic Pro" to activate
            tell application "System Events"
                key code 15 -- "R" key
            end tell
            '''
        )

    def set_tempo(self, bpm: float) -> None:
        self._run_applescript(f'tell application "Logic Pro" to set tempo to {bpm}')

    def set_position(self, bars: float, beats: float = 0.0) -> None:
        # Logic uses locator in beats
        position = (bars - 1) * 4 + beats  # Assuming 4/4
        self._run_applescript(
            f'tell application "Logic Pro" to set locator position to {position}'
        )

    # Track control (limited via AppleScript)
    def create_track(self, track_type: str = "midi", name: str = "") -> int:
        # Use keyboard shortcut
        key = "36" if track_type == "midi" else "36"  # Option+Cmd+N for new track
        self._run_applescript(
            f'''
            tell application "Logic Pro" to activate
            tell application "System Events"
                key code {key} using {{option down, command down}}
            end tell
            '''
        )
        return -1  # Can't get track index via AppleScript

    def arm_track(self, index: int, armed: bool = True) -> None:
        # Would need to select track first, then toggle arm
        pass

    def mute_track(self, index: int, muted: bool = True) -> None:
        pass

    def solo_track(self, index: int, soloed: bool = True) -> None:
        pass

    def set_track_volume(self, index: int, volume_db: float) -> None:
        pass

    def set_track_pan(self, index: int, pan: float) -> None:
        pass

    # MIDI
    def send_note_on(self, note: int, velocity: int = 100, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("note_on", note=note, velocity=velocity, channel=channel)
            self._midi_output.send(msg)

    def send_note_off(self, note: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("note_off", note=note, velocity=0, channel=channel)
            self._midi_output.send(msg)

    def send_note(
        self,
        note: int,
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        self.send_note_on(note, velocity, channel)

        def off_later():
            time.sleep(duration_ms / 1000.0)
            self.send_note_off(note, channel)

        threading.Thread(target=off_later, daemon=True).start()

    def send_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        for note in notes:
            self.send_note_on(note, velocity, channel)

        def off_later():
            time.sleep(duration_ms / 1000.0)
            for note in notes:
                self.send_note_off(note, channel)

        threading.Thread(target=off_later, daemon=True).start()

    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("control_change", control=cc, value=value, channel=channel)
            self._midi_output.send(msg)

    def send_pitch_bend(self, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("pitchwheel", pitch=value, channel=channel)
            self._midi_output.send(msg)

    def all_notes_off(self, channel: Optional[int] = None) -> None:
        if self._midi_output:
            import mido

            channels = [channel] if channel is not None else range(16)
            for ch in channels:
                msg = mido.Message("control_change", control=123, value=0, channel=ch)
                self._midi_output.send(msg)

    # Voice control
    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        vowel_map = {"A": 0, "E": 32, "I": 64, "O": 96, "U": 127}
        value = vowel_map.get(vowel.upper(), 64)
        self.send_cc(VoiceCC.VOWEL, value, channel)

    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        self.send_cc(VoiceCC.BREATHINESS, int(amount * 127), channel)

    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        self.send_cc(VoiceCC.VIBRATO_RATE, int(rate * 127), channel)
        self.send_cc(VoiceCC.VIBRATO_DEPTH, int(depth * 127), channel)


# =============================================================================
# Reaper Bridge (OSC via ReaOSC)
# =============================================================================


@dataclass
class ReaperConfig:
    """Reaper bridge configuration."""

    osc_host: str = "127.0.0.1"
    osc_send_port: int = 8000
    osc_receive_port: int = 9000
    midi_port: str = "DAiW Voice"
    virtual_midi: bool = True


class ReaperBridge(BaseDAWBridge):
    """
    DAW bridge for Reaper.

    Uses OSC (requires ReaControlMIDI or similar) for transport
    and MIDI for notes/CC.
    """

    def __init__(self, config: Optional[ReaperConfig] = None):
        super().__init__()
        self.config = config or ReaperConfig()
        self._osc_client = None
        self._midi_output = None

        self._capabilities = DAWCapabilities(
            has_transport=True,
            has_tempo_control=True,
            has_loop_control=True,
            can_create_tracks=True,
            can_arm_tracks=True,
            can_mute_solo=True,
            has_clip_launcher=False,
            has_arrangement=True,
            has_midi_output=True,
            has_midi_input=True,
            has_osc=True,
            has_automation=True,
            has_markers=True,
            can_control_plugins=True,
            custom_features={"has_reascript": True},
        )

    @property
    def daw_type(self) -> DAWType:
        return DAWType.REAPER

    def _do_connect(self) -> bool:
        try:
            from pythonosc import udp_client

            self._osc_client = udp_client.SimpleUDPClient(
                self.config.osc_host, self.config.osc_send_port
            )
        except ImportError:
            print("python-osc not installed. Run: pip install python-osc")
            return False

        # Connect MIDI
        try:
            import mido

            if self.config.virtual_midi:
                self._midi_output = mido.open_output(
                    self.config.midi_port, virtual=True
                )
            else:
                ports = mido.get_output_names()
                matching = [p for p in ports if self.config.midi_port in p]
                if matching:
                    self._midi_output = mido.open_output(matching[0])
        except ImportError:
            print("mido not installed. Run: pip install mido python-rtmidi")
        except Exception as e:
            print(f"MIDI connection failed: {e}")

        return self._osc_client is not None

    def _do_disconnect(self) -> None:
        if self._midi_output:
            try:
                self.all_notes_off()
                time.sleep(0.05)
                self._midi_output.close()
            except Exception:
                pass
            self._midi_output = None
        self._osc_client = None

    def _send_osc(self, address: str, *args) -> None:
        if self._osc_client:
            self._osc_client.send_message(address, list(args) if args else [])

    # Transport (Reaper OSC)
    def play(self) -> None:
        self._send_osc("/action/1007")  # Transport: Play

    def stop(self) -> None:
        self._send_osc("/action/1016")  # Transport: Stop

    def record(self) -> None:
        self._send_osc("/action/1013")  # Transport: Record

    def set_tempo(self, bpm: float) -> None:
        self._send_osc("/tempo/raw", bpm)

    def set_position(self, bars: float, beats: float = 0.0) -> None:
        # Reaper uses time in seconds or beats
        self._send_osc("/time", bars * 4 + beats)  # Simplified

    def set_loop(self, enabled: bool, start: float = 0.0, end: float = 4.0) -> None:
        self._send_osc("/repeat", 1 if enabled else 0)

    # Track control
    def create_track(self, track_type: str = "midi", name: str = "") -> int:
        self._send_osc("/action/40001")  # Insert new track
        return -1

    def arm_track(self, index: int, armed: bool = True) -> None:
        self._send_osc(f"/track/{index + 1}/recarm", 1 if armed else 0)

    def mute_track(self, index: int, muted: bool = True) -> None:
        self._send_osc(f"/track/{index + 1}/mute", 1 if muted else 0)

    def solo_track(self, index: int, soloed: bool = True) -> None:
        self._send_osc(f"/track/{index + 1}/solo", 1 if soloed else 0)

    def set_track_volume(self, index: int, volume_db: float) -> None:
        self._send_osc(f"/track/{index + 1}/volume/db", volume_db)

    def set_track_pan(self, index: int, pan: float) -> None:
        self._send_osc(f"/track/{index + 1}/pan", pan)

    # MIDI (same as Logic)
    def send_note_on(self, note: int, velocity: int = 100, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("note_on", note=note, velocity=velocity, channel=channel)
            self._midi_output.send(msg)

    def send_note_off(self, note: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("note_off", note=note, velocity=0, channel=channel)
            self._midi_output.send(msg)

    def send_note(
        self,
        note: int,
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        self.send_note_on(note, velocity, channel)

        def off_later():
            time.sleep(duration_ms / 1000.0)
            self.send_note_off(note, channel)

        threading.Thread(target=off_later, daemon=True).start()

    def send_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        for n in notes:
            self.send_note_on(n, velocity, channel)

        def off_later():
            time.sleep(duration_ms / 1000.0)
            for n in notes:
                self.send_note_off(n, channel)

        threading.Thread(target=off_later, daemon=True).start()

    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("control_change", control=cc, value=value, channel=channel)
            self._midi_output.send(msg)

    def send_pitch_bend(self, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("pitchwheel", pitch=value, channel=channel)
            self._midi_output.send(msg)

    def all_notes_off(self, channel: Optional[int] = None) -> None:
        if self._midi_output:
            import mido

            channels = [channel] if channel is not None else range(16)
            for ch in channels:
                msg = mido.Message("control_change", control=123, value=0, channel=ch)
                self._midi_output.send(msg)

    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        vowel_map = {"A": 0, "E": 32, "I": 64, "O": 96, "U": 127}
        self.send_cc(VoiceCC.VOWEL, vowel_map.get(vowel.upper(), 64), channel)

    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        self.send_cc(VoiceCC.BREATHINESS, int(amount * 127), channel)

    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        self.send_cc(VoiceCC.VIBRATO_RATE, int(rate * 127), channel)
        self.send_cc(VoiceCC.VIBRATO_DEPTH, int(depth * 127), channel)


# =============================================================================
# Bitwig Bridge (OSC)
# =============================================================================


@dataclass
class BitwigConfig:
    """Bitwig bridge configuration."""

    osc_host: str = "127.0.0.1"
    osc_send_port: int = 8000
    osc_receive_port: int = 9000
    midi_port: str = "DAiW Voice"
    virtual_midi: bool = True


class BitwigBridge(BaseDAWBridge):
    """
    DAW bridge for Bitwig Studio.

    Uses OSC (requires Bitwig OSC extension) for transport/clips
    and MIDI for notes/CC.
    """

    def __init__(self, config: Optional[BitwigConfig] = None):
        super().__init__()
        self.config = config or BitwigConfig()
        self._osc_client = None
        self._midi_output = None

        self._capabilities = DAWCapabilities(
            has_transport=True,
            has_tempo_control=True,
            has_loop_control=True,
            can_create_tracks=True,
            can_arm_tracks=True,
            can_mute_solo=True,
            has_clip_launcher=True,  # Like Ableton
            has_arrangement=True,
            has_midi_output=True,
            has_midi_input=True,
            has_osc=True,
            has_automation=True,
            has_markers=True,
            can_control_plugins=True,
        )

    @property
    def daw_type(self) -> DAWType:
        return DAWType.BITWIG

    def _do_connect(self) -> bool:
        try:
            from pythonosc import udp_client

            self._osc_client = udp_client.SimpleUDPClient(
                self.config.osc_host, self.config.osc_send_port
            )
        except ImportError:
            print("python-osc not installed. Run: pip install python-osc")
            return False

        # Connect MIDI
        try:
            import mido

            if self.config.virtual_midi:
                self._midi_output = mido.open_output(
                    self.config.midi_port, virtual=True
                )
            else:
                ports = mido.get_output_names()
                matching = [p for p in ports if self.config.midi_port in p]
                if matching:
                    self._midi_output = mido.open_output(matching[0])
        except ImportError:
            print("mido not installed. Run: pip install mido python-rtmidi")
        except Exception as e:
            print(f"MIDI connection failed: {e}")

        return self._osc_client is not None

    def _do_disconnect(self) -> None:
        if self._midi_output:
            try:
                self.all_notes_off()
                time.sleep(0.05)
                self._midi_output.close()
            except Exception:
                pass
            self._midi_output = None
        self._osc_client = None

    def _send_osc(self, address: str, *args) -> None:
        if self._osc_client:
            self._osc_client.send_message(address, list(args) if args else [])

    # Transport (Bitwig OSC)
    def play(self) -> None:
        self._send_osc("/transport/play")

    def stop(self) -> None:
        self._send_osc("/transport/stop")

    def record(self) -> None:
        self._send_osc("/transport/record")

    def set_tempo(self, bpm: float) -> None:
        self._send_osc("/transport/tempo/raw", bpm)

    def set_position(self, bars: float, beats: float = 0.0) -> None:
        self._send_osc("/transport/position", bars, beats)

    def set_loop(self, enabled: bool, start: float = 0.0, end: float = 4.0) -> None:
        self._send_osc("/transport/loop", 1 if enabled else 0)
        self._send_osc("/transport/loop/start", start)
        self._send_osc("/transport/loop/end", end)

    # Track control
    def create_track(self, track_type: str = "midi", name: str = "") -> int:
        self._send_osc("/track/add", track_type)
        return -1

    def arm_track(self, index: int, armed: bool = True) -> None:
        self._send_osc(f"/track/{index}/arm", 1 if armed else 0)

    def mute_track(self, index: int, muted: bool = True) -> None:
        self._send_osc(f"/track/{index}/mute", 1 if muted else 0)

    def solo_track(self, index: int, soloed: bool = True) -> None:
        self._send_osc(f"/track/{index}/solo", 1 if soloed else 0)

    def set_track_volume(self, index: int, volume_db: float) -> None:
        self._send_osc(f"/track/{index}/volume", volume_db)

    def set_track_pan(self, index: int, pan: float) -> None:
        self._send_osc(f"/track/{index}/pan", pan)

    # Clips (Bitwig has clip launcher like Ableton)
    def fire_clip(self, track: int, clip: int) -> None:
        self._send_osc(f"/track/{track}/clip/{clip}/launch")

    def stop_clip(self, track: int, clip: int) -> None:
        self._send_osc(f"/track/{track}/clip/{clip}/stop")

    # MIDI (same pattern)
    def send_note_on(self, note: int, velocity: int = 100, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("note_on", note=note, velocity=velocity, channel=channel)
            self._midi_output.send(msg)

    def send_note_off(self, note: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("note_off", note=note, velocity=0, channel=channel)
            self._midi_output.send(msg)

    def send_note(
        self,
        note: int,
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        self.send_note_on(note, velocity, channel)

        def off_later():
            time.sleep(duration_ms / 1000.0)
            self.send_note_off(note, channel)

        threading.Thread(target=off_later, daemon=True).start()

    def send_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: int = 500,
        channel: int = 0,
    ) -> None:
        for n in notes:
            self.send_note_on(n, velocity, channel)

        def off_later():
            time.sleep(duration_ms / 1000.0)
            for n in notes:
                self.send_note_off(n, channel)

        threading.Thread(target=off_later, daemon=True).start()

    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("control_change", control=cc, value=value, channel=channel)
            self._midi_output.send(msg)

    def send_pitch_bend(self, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido

            msg = mido.Message("pitchwheel", pitch=value, channel=channel)
            self._midi_output.send(msg)

    def all_notes_off(self, channel: Optional[int] = None) -> None:
        if self._midi_output:
            import mido

            channels = [channel] if channel is not None else range(16)
            for ch in channels:
                msg = mido.Message("control_change", control=123, value=0, channel=ch)
                self._midi_output.send(msg)

    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        vowel_map = {"A": 0, "E": 32, "I": 64, "O": 96, "U": 127}
        self.send_cc(VoiceCC.VOWEL, vowel_map.get(vowel.upper(), 64), channel)

    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        self.send_cc(VoiceCC.BREATHINESS, int(amount * 127), channel)

    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        self.send_cc(VoiceCC.VIBRATO_RATE, int(rate * 127), channel)
        self.send_cc(VoiceCC.VIBRATO_DEPTH, int(depth * 127), channel)


# =============================================================================
# Register all bridges
# =============================================================================


def _register_bridges():
    """Register all DAW bridges with the registry."""
    DAWRegistry.register(DAWType.ABLETON, AbletonDAWBridge)
    DAWRegistry.register(DAWType.LOGIC_PRO, LogicProBridge)
    DAWRegistry.register(DAWType.REAPER, ReaperBridge)
    DAWRegistry.register(DAWType.BITWIG, BitwigBridge)

    # Set Ableton as default (can be changed by user)
    DAWRegistry.set_default(DAWType.ABLETON)


# Auto-register on import
_register_bridges()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Configs
    "AbletonConfig",
    "LogicProConfig",
    "ReaperConfig",
    "BitwigConfig",
    # Bridges
    "AbletonDAWBridge",
    "LogicProBridge",
    "ReaperBridge",
    "BitwigBridge",
    # Voice CC
    "VoiceCC",
    "VOWEL_FORMANTS",
]

