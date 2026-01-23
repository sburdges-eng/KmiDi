"""
DAiW Brain Server - OSC Bridge for JUCE Plugin Integration
==========================================================

This server bridges the Python DAiW brain with C++ JUCE plugins via OSC.
It receives generation requests from plugins and responds with MIDI data.

OSC Protocol:
- Receive: /daiw/generate (text, motivation, chaos, vulnerability)
- Send: /daiw/result (JSON with MIDI events)
- Receive: /daiw/ping
- Send: /daiw/pong

Philosophy: "Interrogate Before Generate" - The tool shouldn't finish art
for people. It should make them braver.
"""

import json
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from pythonosc import osc_server, udp_client, dispatcher
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("[WARNING] pythonosc not installed. Install with: pip install python-osc")

from music_brain.structure.comprehensive_engine import (
    TherapySession,
    HarmonyPlan,
    NoteEvent,
    render_plan_to_midi,
)


class DAiWBrainServer:
    """
    OSC server that receives requests from JUCE plugins and generates MIDI.
    
    The server listens for OSC messages and processes them using the
    comprehensive engine, then responds with MIDI data in JSON format.
    """
    
    def __init__(self, listen_port: int = 9000, send_port: int = 9001):
        """
        Initialize the brain server.
        
        Args:
            listen_port: Port to listen for incoming OSC messages (from plugin)
            send_port: Port to send OSC responses (to plugin)
        """
        if not OSC_AVAILABLE:
            raise RuntimeError(
                "pythonosc is required. Install with: pip install python-osc"
            )
        
        self.listen_port = listen_port
        self.send_port = send_port
        self.client = udp_client.SimpleUDPClient("127.0.0.1", send_port)
        self.dispatcher = dispatcher.Dispatcher()
        self.server: Optional[osc_server.ThreadingOSCUDPServer] = None
        
        # Register OSC message handlers
        self._register_handlers()
        
        # Statistics
        self.stats = {
            "requests_received": 0,
            "requests_processed": 0,
            "errors": 0,
            "start_time": time.time(),
        }
    
    def _register_handlers(self):
        """Register OSC message handlers."""
        self.dispatcher.map("/daiw/generate", self._handle_generate)
        self.dispatcher.map("/daiw/ping", self._handle_ping)
        self.dispatcher.map("/daiw/set_intent", self._handle_set_intent)
    
    def _handle_generate(
        self,
        address: str,
        *args: Any
    ) -> None:
        """
        Handle generation request from plugin.
        
        Expected OSC message format:
        /daiw/generate <text> <motivation> <chaos> <vulnerability>
        
        Where:
        - text: str - User's emotional input
        - motivation: float (1-10) - How complete the piece should be
        - chaos: float (1-10) - Chaos tolerance level
        - vulnerability: float (1-10) - Vulnerability level
        """
        self.stats["requests_received"] += 1
        
        try:
            # Parse arguments
            if len(args) < 4:
                self._send_error("Invalid arguments. Expected: text, motivation, chaos, vulnerability")
                return
            
            text = str(args[0]) if args[0] else ""
            motivation = float(args[1]) if args[1] else 7.0
            chaos = float(args[2]) if args[2] else 5.0
            vulnerability = float(args[3]) if args[3] else 5.0
            
            # Validate inputs
            if not text.strip():
                self._send_error("Text input cannot be empty")
                return
            
            motivation = max(1.0, min(10.0, motivation))
            chaos = max(1.0, min(10.0, chaos))
            vulnerability = max(1.0, min(10.0, vulnerability))
            
            # Process with comprehensive engine
            session = TherapySession()
            affect = session.process_core_input(text)
            session.set_scales(int(motivation), chaos / 10.0)
            plan = session.generate_plan()
            
            # Convert plan to MIDI events (NoteEvent format)
            midi_events = self._plan_to_events(plan)
            
            # Get affect intensity safely
            affect_intensity = 0.7
            if session.state.affect_result:
                affect_intensity = session.state.affect_result.intensity
            
            # Create response JSON
            response = {
                "status": "success",
                "affect": {
                    "primary": affect,
                    "intensity": affect_intensity,
                },
                "plan": {
                    "tempo_bpm": plan.tempo_bpm,
                    "key": plan.root_note,
                    "mode": plan.mode,
                    "time_signature": plan.time_signature,
                    "chords": plan.chord_symbols,
                    "length_bars": plan.length_bars,
                    "complexity": plan.complexity,
                },
                "midi_events": midi_events,
                "ppq": 480,  # Include PPQ for timing calculations
            }
            
            # Send response
            self._send_result(json.dumps(response))
            self.stats["requests_processed"] += 1
            
        except Exception as e:
            self.stats["errors"] += 1
            error_msg = f"Error processing request: {str(e)}"
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            self._send_error(error_msg)
    
    def _handle_ping(self, address: str, *args: Any) -> None:
        """Handle ping request from plugin."""
        self.client.send_message("/daiw/pong", ["alive"])
        print("[INFO] Ping received, sent pong")
    
    def _handle_set_intent(self, address: str, *args: Any) -> None:
        """
        Handle intent update from plugin.
        
        Expected OSC message format:
        /daiw/set_intent <intent_json>
        """
        try:
            if len(args) < 1:
                self._send_error("Invalid arguments. Expected: intent_json")
                return
            
            intent_json = str(args[0])
            intent_data = json.loads(intent_json)
            
            # Store intent for future use (could be used to pre-load context)
            # For now, just acknowledge
            self.client.send_message("/daiw/intent_ack", ["ok"])
            print(f"[INFO] Intent received: {intent_data.get('title', 'Untitled')}")
            
        except Exception as e:
            error_msg = f"Error processing intent: {str(e)}"
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            self._send_error(error_msg)
    
    def _plan_to_events(self, plan: HarmonyPlan) -> List[Dict[str, Any]]:
        """
        Convert HarmonyPlan to list of NoteEvent dictionaries.
        
        Uses the comprehensive engine's render logic to generate proper
        NoteEvents with correct chord voicings and timing.
        """
        events = []
        
        try:
            from music_brain.structure.progression import parse_progression_string
            from music_brain.structure.chord import CHORD_QUALITIES
        except ImportError:
            # Fallback to simplified generation if modules unavailable
            return self._plan_to_events_simple(plan)
        
        # Calculate timing (matching render_plan_to_midi logic)
        ppq = 480  # Standard PPQ
        time_sig_parts = plan.time_signature.split("/")
        beats_per_bar = int(time_sig_parts[0]) if len(time_sig_parts) == 2 else 4
        ticks_per_bar = beats_per_bar * ppq
        
        # Parse chord progression
        progression_str = "-".join(plan.chord_symbols)
        parsed_chords = parse_progression_string(progression_str)
        
        # Generate NoteEvents (matching comprehensive_engine logic)
        start_tick = 0
        current_bar = 0
        total_bars = plan.length_bars
        
        while current_bar < total_bars:
            for parsed in parsed_chords:
                if current_bar >= total_bars:
                    break
                
                quality = parsed.quality
                intervals = CHORD_QUALITIES.get(quality)
                
                # Fallback for unknown qualities
                if intervals is None:
                    base_quality = "min" if "m" in quality else "maj"
                    intervals = CHORD_QUALITIES.get(base_quality, (0, 4, 7))
                
                # Calculate root MIDI note (C3 = 48)
                root_midi = 48 + parsed.root_num
                duration_ticks = ticks_per_bar
                
                # Create note_on and note_off events for each interval
                for interval in intervals:
                    pitch = root_midi + interval
                    
                    # Note on
                    events.append({
                        "type": "note_on",
                        "pitch": pitch,
                        "velocity": 80,
                        "channel": 1,
                        "tick": start_tick,
                        "duration_ticks": duration_ticks,
                        "bar": current_bar,
                        "chord": plan.chord_symbols[current_bar % len(plan.chord_symbols)],
                    })
                    
                    # Note off
                    events.append({
                        "type": "note_off",
                        "pitch": pitch,
                        "velocity": 0,
                        "channel": 1,
                        "tick": start_tick + duration_ticks,
                        "duration_ticks": 0,
                        "bar": current_bar,
                        "chord": plan.chord_symbols[current_bar % len(plan.chord_symbols)],
                    })
                
                start_tick += duration_ticks
                current_bar += 1
        
        # Sort events by tick for proper sequencing
        events.sort(key=lambda e: e["tick"])
        
        return events
    
    def _plan_to_events_simple(self, plan: HarmonyPlan) -> List[Dict[str, Any]]:
        """
        Simplified fallback event generation when comprehensive engine unavailable.
        """
        events = []
        ppq = 480
        beats_per_bar = int(plan.time_signature.split("/")[0]) if "/" in plan.time_signature else 4
        ticks_per_bar = beats_per_bar * ppq
        current_tick = 0
        
        for bar in range(plan.length_bars):
            chord_idx = bar % len(plan.chord_symbols)
            chord = plan.chord_symbols[chord_idx]
            root_note = self._chord_to_midi_note(chord, plan.root_note)
            notes = [root_note, root_note + 4, root_note + 7]  # Basic triad
            
            for note in notes:
                events.append({
                    "type": "note_on",
                    "pitch": note,
                    "velocity": 80,
                    "channel": 1,
                    "tick": current_tick,
                    "duration_ticks": ticks_per_bar,
                    "bar": bar,
                    "chord": chord,
                })
                events.append({
                    "type": "note_off",
                    "pitch": note,
                    "velocity": 0,
                    "channel": 1,
                    "tick": current_tick + ticks_per_bar,
                    "duration_ticks": 0,
                    "bar": bar,
                    "chord": chord,
                })
            
            current_tick += ticks_per_bar
        
        return events
    
    def _chord_to_midi_note(self, chord: str, root_key: str) -> int:
        """
        Convert chord symbol to MIDI note number.
        
        Simplified version - just returns the root note in MIDI format.
        """
        # Map note names to MIDI numbers (C4 = 60)
        note_map = {
            "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
            "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
            "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71,
        }
        
        # Extract root from chord (e.g., "F" from "Fmaj7")
        root = chord[0]
        if len(chord) > 1 and chord[1] in "#b":
            root = chord[:2]
        
        # Get base MIDI note
        base_note = note_map.get(root, 60)
        
        # Adjust for key context (simplified)
        key_root = root_key[0] if root_key else "C"
        if len(root_key) > 1 and root_key[1] in "#b":
            key_root = root_key[:2]
        
        key_base = note_map.get(key_root, 60)
        offset = base_note - key_base
        
        return 60 + offset  # Return in C4 octave for simplicity
    
    def _send_result(self, json_data: str) -> None:
        """Send generation result to plugin."""
        self.client.send_message("/daiw/result", [json_data])
        print(f"[INFO] Sent result to plugin (port {self.send_port})")
    
    def _send_error(self, error_msg: str) -> None:
        """Send error message to plugin."""
        error_response = {
            "status": "error",
            "message": error_msg,
        }
        self.client.send_message("/daiw/error", [json.dumps(error_response)])
        print(f"[ERROR] Sent error to plugin: {error_msg}")
    
    def start(self) -> None:
        """Start the OSC server."""
        if self.server is not None:
            print("[WARNING] Server already running")
            return
        
        try:
            self.server = osc_server.ThreadingOSCUDPServer(
                ("127.0.0.1", self.listen_port),
                self.dispatcher
            )
            
            print(f"[INFO] DAiW Brain Server starting...")
            print(f"[INFO] Listening on port {self.listen_port}")
            print(f"[INFO] Sending to port {self.send_port}")
            print(f"[INFO] Ready to receive OSC messages")
            print(f"[INFO] Press Ctrl+C to stop")
            print()
            
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down server...")
            self.stop()
        except Exception as e:
            print(f"[ERROR] Server error: {e}", file=sys.stderr)
            raise
    
    def stop(self) -> None:
        """Stop the OSC server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        
        # Print statistics
        uptime = time.time() - self.stats["start_time"]
        print(f"\n[STATS] Server Statistics:")
        print(f"  Requests received: {self.stats['requests_received']}")
        print(f"  Requests processed: {self.stats['requests_processed']}")
        print(f"  Errors: {self.stats['errors']}")
        print(f"  Uptime: {uptime:.1f} seconds")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            **self.stats,
            "uptime": time.time() - self.stats["start_time"],
        }


def main():
    """Main entry point for the brain server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DAiW Brain Server - OSC bridge for JUCE plugin integration"
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=9000,
        help="Port to listen for incoming OSC messages (default: 9000)"
    )
    parser.add_argument(
        "--send-port",
        type=int,
        default=9001,
        help="Port to send OSC responses (default: 9001)"
    )
    
    args = parser.parse_args()
    
    if not OSC_AVAILABLE:
        print("[ERROR] pythonosc is required. Install with: pip install python-osc")
        sys.exit(1)
    
    server = DAiWBrainServer(
        listen_port=args.listen_port,
        send_port=args.send_port
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        server.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()

