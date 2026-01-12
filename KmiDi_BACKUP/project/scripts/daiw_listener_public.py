#!/usr/bin/env python3
"""
DAiW Listener - Terminal Version (Public)
==========================================
Watches a folder for generation requests and outputs MIDI + audio.

Requirements:
    pip install music21 mido pydub watchdog

Usage:
    python daiw_listener.py [--output-dir ./output] [--watch-dir ./incoming]
    
Configuration:
    Set SAMPLE_DIR environment variable to your sample library path,
    or edit the SAMPLE_LIBRARY dict below.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import random

from music21 import stream, note, chord, tempo, meter, key, instrument
from mido import MidiFile, MidiTrack, Message
from pydub import AudioSegment
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============ CONFIGURATION ============
DEFAULT_OUTPUT = Path("./daiw_output")
DEFAULT_WATCH = Path("./daiw_incoming")
SAMPLE_DIR = Path(os.environ.get("DAIW_SAMPLE_DIR", "./samples"))

# ============ SAMPLE LIBRARY (customize these paths) ============
SAMPLE_LIBRARY = {
    "kick": SAMPLE_DIR / "drums" / "kick",
    "snare": SAMPLE_DIR / "drums" / "snare", 
    "hihat": SAMPLE_DIR / "drums" / "hihat",
    "percussion": SAMPLE_DIR / "drums" / "percussion",
    "bass": SAMPLE_DIR / "bass",
    "pads": SAMPLE_DIR / "synths" / "pads",
    "keys": SAMPLE_DIR / "synths" / "keys",
    "guitar": SAMPLE_DIR / "guitar",
}

# ============ CHORD/SCALE DATA ============
SCALE_PATTERNS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
}

KEY_TO_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
}


# ============ HUMANIZATION ============
def humanize_timing(base_time: int, variance: float = 0.1, pocket: str = "locked") -> int:
    """Add human timing variation."""
    pocket_offsets = {"behind": 15, "ahead": -12, "locked": 0}
    offset = pocket_offsets.get(pocket, 0)
    variation = int(random.gauss(0, variance * 30))
    return max(0, base_time + offset + variation)

def humanize_velocity(base_vel: int, variance: float = 0.15) -> int:
    """Add human velocity variation."""
    variation = int(random.gauss(0, variance * base_vel))
    return max(1, min(127, base_vel + variation))


# ============ PATTERN GENERATORS ============
class PatternGenerator:
    """Generates musical patterns from config."""
    
    def __init__(self, key_root: str, mode: str, bpm: int):
        self.root = KEY_TO_MIDI.get(key_root, 60)
        self.scale = SCALE_PATTERNS.get(mode, SCALE_PATTERNS["major"])
        self.bpm = bpm
        self.ticks_per_beat = 480
        
    def scale_note(self, degree: int, octave_offset: int = 0) -> int:
        """Get MIDI note from scale degree."""
        octave = degree // 7
        step = degree % 7
        return self.root + self.scale[step] + (octave + octave_offset) * 12
    
    def generate_drums(self, bars: int, style: str = "basic") -> list:
        """Generate drum pattern."""
        patterns = {
            "basic": self._drum_basic,
            "boom_bap": self._drum_boom_bap,
            "four_on_floor": self._drum_four_on_floor,
            "sparse": self._drum_sparse,
        }
        return patterns.get(style, self._drum_basic)(bars)
    
    def _drum_basic(self, bars: int) -> dict:
        """Basic rock beat."""
        kick, snare, hat = [], [], []
        tpb = self.ticks_per_beat
        
        for bar in range(bars):
            bar_start = bar * 4 * tpb
            # Kick on 1 and 3
            kick.append({"pitch": 36, "position": bar_start, "duration": tpb // 2})
            kick.append({"pitch": 36, "position": bar_start + 2 * tpb, "duration": tpb // 2})
            # Snare on 2 and 4
            snare.append({"pitch": 38, "position": bar_start + tpb, "duration": tpb // 2})
            snare.append({"pitch": 38, "position": bar_start + 3 * tpb, "duration": tpb // 2})
            # Hi-hat 8ths
            for i in range(8):
                hat.append({"pitch": 42, "position": bar_start + i * (tpb // 2), "duration": tpb // 4})
                
        return {"kick": kick, "snare": snare, "hihat": hat}
    
    def _drum_boom_bap(self, bars: int) -> dict:
        """Hip-hop boom bap."""
        kick, snare, hat = [], [], []
        tpb = self.ticks_per_beat
        
        for bar in range(bars):
            bar_start = bar * 4 * tpb
            # Kick on 1, and-of-2
            kick.append({"pitch": 36, "position": bar_start, "duration": tpb // 2})
            kick.append({"pitch": 36, "position": bar_start + tpb + tpb // 2, "duration": tpb // 2})
            # Snare on 2 and 4
            snare.append({"pitch": 38, "position": bar_start + tpb, "duration": tpb // 2})
            snare.append({"pitch": 38, "position": bar_start + 3 * tpb, "duration": tpb // 2})
            # Sparse hat
            for i in [0, 2, 4, 6]:
                hat.append({"pitch": 42, "position": bar_start + i * (tpb // 2), "duration": tpb // 4})
                
        return {"kick": kick, "snare": snare, "hihat": hat}
    
    def _drum_four_on_floor(self, bars: int) -> dict:
        """EDM/dance four on floor."""
        kick, snare, hat = [], [], []
        tpb = self.ticks_per_beat
        
        for bar in range(bars):
            bar_start = bar * 4 * tpb
            # Kick every beat
            for i in range(4):
                kick.append({"pitch": 36, "position": bar_start + i * tpb, "duration": tpb // 2})
            # Snare/clap on 2 and 4
            snare.append({"pitch": 38, "position": bar_start + tpb, "duration": tpb // 2})
            snare.append({"pitch": 38, "position": bar_start + 3 * tpb, "duration": tpb // 2})
            # Offbeat hat
            for i in range(4):
                hat.append({"pitch": 42, "position": bar_start + i * tpb + tpb // 2, "duration": tpb // 4})
                
        return {"kick": kick, "snare": snare, "hihat": hat}
    
    def _drum_sparse(self, bars: int) -> dict:
        """Minimal, sparse drums for intimate songs."""
        kick, snare = [], []
        tpb = self.ticks_per_beat
        
        for bar in range(bars):
            bar_start = bar * 4 * tpb
            if bar % 2 == 0:
                kick.append({"pitch": 36, "position": bar_start, "duration": tpb})
            if bar % 2 == 1:
                snare.append({"pitch": 38, "position": bar_start + 2 * tpb, "duration": tpb})
                
        return {"kick": kick, "snare": snare, "hihat": []}
    
    def generate_bass(self, chord_progression: list, bars: int) -> list:
        """Generate bass line from chords."""
        bass = []
        tpb = self.ticks_per_beat
        bars_per_chord = bars // len(chord_progression)
        
        for i, chord_root in enumerate(chord_progression):
            for bar in range(bars_per_chord):
                bar_start = (i * bars_per_chord + bar) * 4 * tpb
                root = self.root + chord_root - 12  # Bass octave
                # Root on 1
                bass.append({"pitch": root, "position": bar_start, "duration": tpb})
                # Fifth on 3 (sometimes)
                if random.random() > 0.3:
                    bass.append({"pitch": root + 7, "position": bar_start + 2 * tpb, "duration": tpb})
                    
        return bass
    
    def generate_chords(self, chord_progression: list, bars: int, voicing: str = "basic") -> list:
        """Generate chord voicings."""
        chords = []
        tpb = self.ticks_per_beat
        bars_per_chord = max(1, bars // len(chord_progression))
        
        for i, chord_def in enumerate(chord_progression):
            bar_start = i * bars_per_chord * 4 * tpb
            
            if isinstance(chord_def, int):
                # Simple root - build triad
                root = self.root + chord_def
                notes = [root, root + 4, root + 7]  # Major triad
            elif isinstance(chord_def, dict):
                # Detailed chord definition
                root = self.root + chord_def.get("root", 0)
                quality = chord_def.get("quality", "major")
                if quality == "minor":
                    notes = [root, root + 3, root + 7]
                elif quality == "dim":
                    notes = [root, root + 3, root + 6]
                elif quality == "7":
                    notes = [root, root + 4, root + 7, root + 10]
                else:
                    notes = [root, root + 4, root + 7]
            else:
                notes = [self.root, self.root + 4, self.root + 7]
            
            # Add chord (whole notes)
            duration = bars_per_chord * 4 * tpb
            chords.append({
                "notes": notes,
                "position": bar_start,
                "duration": duration
            })
            
        return chords


# ============ CORE GENERATOR ============
class DAiWGenerator:
    """Generates MIDI and audio from instruction JSON."""
    
    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.title = config.get("title", "untitled")
        self.key_sig = config.get("key", "C")
        self.mode = config.get("mode", "major")
        self.bpm = config.get("bpm", 120)
        self.time_sig = config.get("time_sig", "4/4")
        self.bars = config.get("bars", 16)
        self.groove = config.get("groove", {})
        self.style = config.get("style", "basic")
        self.chord_progression = config.get("chords", [0, 5, 3, 4])  # I-V-IV-V default
        
        self.pattern_gen = PatternGenerator(self.key_sig, self.mode, self.bpm)
        
    def generate_midi(self) -> MidiFile:
        """Generate complete MIDI from config."""
        mid = MidiFile()
        mid.ticks_per_beat = 480
        
        pocket = self.groove.get("pocket", "locked")
        humanize = self.groove.get("humanize", 0.1)
        
        # Tempo track
        tempo_track = MidiTrack()
        tempo_track.name = "Tempo"
        mid.tracks.append(tempo_track)
        
        # Generate patterns
        drums = self.pattern_gen.generate_drums(self.bars, self.style)
        bass = self.pattern_gen.generate_bass(self.chord_progression, self.bars)
        chords = self.pattern_gen.generate_chords(self.chord_progression, self.bars)
        
        # Drum tracks
        for drum_name, drum_events in drums.items():
            if drum_events:
                track = self._events_to_track(drum_events, drum_name, 9, pocket, humanize)
                mid.tracks.append(track)
        
        # Bass track
        if bass:
            track = self._events_to_track(bass, "bass", 1, pocket, humanize, velocity=90)
            mid.tracks.append(track)
        
        # Chord track
        if chords:
            track = self._chords_to_track(chords, "chords", 2, pocket, humanize)
            mid.tracks.append(track)
        
        return mid
    
    def _events_to_track(self, events: list, name: str, channel: int, 
                         pocket: str, humanize: float, velocity: int = 80) -> MidiTrack:
        """Convert events to MIDI track with humanization."""
        track = MidiTrack()
        track.name = name
        
        # Sort by position
        events = sorted(events, key=lambda x: x["position"])
        
        current_time = 0
        for event in events:
            pos = humanize_timing(event["position"], humanize, pocket)
            vel = humanize_velocity(velocity, humanize)
            duration = event["duration"]
            pitch = event["pitch"]
            
            delta = max(0, pos - current_time)
            track.append(Message('note_on', note=pitch, velocity=vel, time=delta, channel=channel))
            track.append(Message('note_off', note=pitch, velocity=0, time=duration, channel=channel))
            current_time = pos + duration
            
        return track
    
    def _chords_to_track(self, chords: list, name: str, channel: int,
                         pocket: str, humanize: float) -> MidiTrack:
        """Convert chord events to MIDI track."""
        track = MidiTrack()
        track.name = name
        
        current_time = 0
        for chord_event in chords:
            pos = humanize_timing(chord_event["position"], humanize * 0.5, pocket)
            duration = chord_event["duration"]
            notes = chord_event["notes"]
            
            delta = max(0, pos - current_time)
            
            # Note ons (strum effect - slight delay between notes)
            for i, pitch in enumerate(notes):
                vel = humanize_velocity(70, humanize)
                strum_delay = i * 10 if i > 0 else delta
                track.append(Message('note_on', note=pitch, velocity=vel, 
                                    time=strum_delay if i > 0 else delta, channel=channel))
            
            # Note offs
            track.append(Message('note_off', note=notes[0], velocity=0, time=duration, channel=channel))
            for pitch in notes[1:]:
                track.append(Message('note_off', note=pitch, velocity=0, time=0, channel=channel))
            
            current_time = pos + duration
            
        return track
    
    def generate(self) -> Path:
        """Full generation: MIDI file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in self.title if c.isalnum() or c in "_ -").strip()
        safe_title = safe_title or "song"
        base_name = f"{timestamp}_{safe_title}"
        
        midi_path = self.output_dir / f"{base_name}.mid"
        
        # Generate and save MIDI
        midi = self.generate_midi()
        midi.save(str(midi_path))
        
        print(f"✓ Generated: {midi_path}")
        return midi_path


# ============ FILE WATCHER ============
class IncomingHandler(FileSystemEventHandler):
    """Watches for incoming .json instruction files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def on_created(self, event):
        if event.src_path.endswith('.json'):
            self.process_file(Path(event.src_path))
            
    def process_file(self, json_path: Path):
        """Process an instruction file."""
        time.sleep(0.5)  # Wait for file to be fully written
        
        try:
            print(f"\n🎵 Processing: {json_path.name}")
            
            with open(json_path) as f:
                config = json.load(f)
            
            generator = DAiWGenerator(config, self.output_dir)
            midi_path = generator.generate()
            
            # Clean up instruction file
            json_path.unlink()
            
            print(f"✓ Complete! Output: {midi_path}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description="DAiW Listener - Generate music from JSON instructions")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT,
                       help="Output directory for generated files")
    parser.add_argument("--watch-dir", "-w", type=Path, default=DEFAULT_WATCH,
                       help="Directory to watch for instruction files")
    parser.add_argument("--once", "-1", type=Path, default=None,
                       help="Process a single JSON file and exit")
    
    args = parser.parse_args()
    
    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.watch_dir.mkdir(parents=True, exist_ok=True)
    
    # Single file mode
    if args.once:
        if args.once.exists():
            handler = IncomingHandler(args.output_dir)
            handler.process_file(args.once)
        else:
            print(f"File not found: {args.once}")
        return
    
    # Watch mode
    print(f"""
╔══════════════════════════════════════════╗
║         DAiW Listener Running            ║
╠══════════════════════════════════════════╣
║  Output:  {str(args.output_dir):<30} ║
║  Watch:   {str(args.watch_dir):<30} ║
╠══════════════════════════════════════════╣
║  Drop .json files into watch folder      ║
║  Press Ctrl+C to stop                    ║
╚══════════════════════════════════════════╝
""")
    
    handler = IncomingHandler(args.output_dir)
    observer = Observer()
    observer.schedule(handler, str(args.watch_dir), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        observer.stop()
    
    observer.join()
    print("Done.")


if __name__ == "__main__":
    main()
