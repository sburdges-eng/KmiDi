#!/usr/bin/env python3
"""
DAiW Listener - Menubar App (Sean's Version)
============================================
Runs in macOS menubar, listens for generation requests from GPT.
Outputs to ~/Music/DAiW_Output/

Requirements:
    pip install rumps music21 mido pydub watchdog

Usage:
    python daiw_menubar.py
    
The app watches a dropbox folder for .json instruction files from GPT.
When a file appears, it generates the song and notifies you.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import rumps
except ImportError:
    print("Install rumps: pip install rumps")
    exit(1)

from music21 import stream, note, chord, tempo, meter, key, instrument
from mido import MidiFile, MidiTrack, Message
from pydub import AudioSegment
from pydub.generators import Sine
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============ CONFIGURATION ============
OUTPUT_DIR = Path.home() / "Music" / "DAiW_Output"
DROPBOX_DIR = OUTPUT_DIR / "incoming"
SAMPLE_BASE = Path.home() / "Google Drive" / "My Drive"  # Adjust if different

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DROPBOX_DIR.mkdir(parents=True, exist_ok=True)

# ============ SAMPLE LIBRARY PATHS ============
SAMPLE_LIBRARY = {
    "drums": {
        "acoustic": SAMPLE_BASE / "audio_vault" / "Drum Tornado 2023" / "Acoustic",
        "electronic": SAMPLE_BASE / "audio_vault" / "Drum Empire 2020",
    },
    "kick": SAMPLE_BASE / "audio_vault" / "Drum Empire 2020" / "Kick",
    "snare": SAMPLE_BASE / "audio_vault" / "Drum Empire 2020" / "Snare",
    "hihat": SAMPLE_BASE / "audio_vault" / "Drum Empire 2020" / "HiHat",
    "percussion": SAMPLE_BASE / "audio_vault" / "Percussion",
    "pads": SAMPLE_BASE / "audio_vault" / "Pads & Strings",
    "keys": SAMPLE_BASE / "audio_vault" / "Plucks & Keys",
    "bass": SAMPLE_BASE / "audio_vault" / "Bass",
    "synthwave": SAMPLE_BASE / "audio_vault" / "Synthwave",
}


# ============ CORE GENERATOR ============
class DAiWGenerator:
    """Generates MIDI and renders audio from instruction JSON."""
    
    def __init__(self, config: dict):
        self.config = config
        self.title = config.get("title", "untitled")
        self.key_sig = config.get("key", "C")
        self.mode = config.get("mode", "major")
        self.bpm = config.get("bpm", 120)
        self.time_sig = config.get("time_sig", "4/4")
        self.bars = config.get("bars", 16)
        self.groove = config.get("groove", {})
        self.arrangement = config.get("arrangement", {})
        self.samples = config.get("samples", {})
        
    def generate_midi(self) -> MidiFile:
        """Generate complete MIDI from config."""
        mid = MidiFile()
        mid.ticks_per_beat = 480
        
        # Tempo track
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        microseconds = int(60_000_000 / self.bpm)
        tempo_track.append(Message('program_change', program=0, time=0))
        
        # Generate each instrument track
        for inst_name, inst_data in self.arrangement.items():
            track = self._generate_track(inst_name, inst_data)
            mid.tracks.append(track)
            
        return mid
    
    def _generate_track(self, name: str, data: dict) -> MidiTrack:
        """Generate a single instrument track."""
        track = MidiTrack()
        track.name = name
        
        pattern = data.get("pattern", [])
        velocity_base = data.get("velocity", 80)
        channel = data.get("channel", 0)
        
        # Humanization
        timing_variance = self.groove.get("humanize", 0.1)
        velocity_variance = 0.15
        pocket = self.groove.get("pocket", "locked")
        pocket_offset = {"behind": 10, "ahead": -10, "locked": 0}.get(pocket, 0)
        
        current_time = 0
        for event in pattern:
            pitch = event.get("pitch", 60)
            duration = event.get("duration", 480)
            position = event.get("position", current_time)
            
            # Apply humanization
            import random
            time_offset = int(random.gauss(0, timing_variance * 20)) + pocket_offset
            vel_offset = int(random.gauss(0, velocity_variance * velocity_base))
            velocity = max(1, min(127, velocity_base + vel_offset))
            
            delta = max(0, position - current_time + time_offset)
            track.append(Message('note_on', note=pitch, velocity=velocity, time=delta, channel=channel))
            track.append(Message('note_off', note=pitch, velocity=0, time=duration, channel=channel))
            current_time = position + duration
            
        return track
    
    def render_audio(self, midi_path: Path) -> Path:
        """Render MIDI + samples to audio."""
        audio_path = midi_path.with_suffix('.wav')
        
        # Load MIDI file
        try:
            mid = MidiFile(str(midi_path))
        except Exception as e:
            print(f"Error loading MIDI: {e}")
            # Fallback to silent audio
            duration_ms = int((self.bars * 4 * 60 / self.bpm) * 1000)
            audio = AudioSegment.silent(duration=duration_ms)
            audio.export(str(audio_path), format="wav")
            return audio_path
        
        # Calculate total duration
        duration_ms = int((self.bars * 4 * 60 / self.bpm) * 1000)
        
        # Start with silence
        output_audio = AudioSegment.silent(duration=duration_ms)
        
        # Map MIDI events to samples from library
        # Track instrument assignments from arrangement config
        for track_idx, track in enumerate(mid.tracks):
            if track_idx == 0:
                continue  # Skip tempo track
            
            # Get instrument name from track
            instrument_name = None
            for msg in track:
                if msg.type == 'track_name':
                    instrument_name = msg.name.lower()
                    break
            
            if not instrument_name:
                continue
            
            # Find samples for this instrument
            sample_dir = None
            if instrument_name in SAMPLE_LIBRARY:
                sample_dir = SAMPLE_LIBRARY[instrument_name]
                if isinstance(sample_dir, dict):
                    # Pick first available type
                    sample_dir = list(sample_dir.values())[0]
            
            if not sample_dir or not sample_dir.exists():
                continue
            
            # Load available samples
            samples = {}
            sample_files = sorted(sample_dir.glob('*.wav'))  # Sort for deterministic ordering
            for idx, sample_file in enumerate(sample_files):
                try:
                    sample = AudioSegment.from_file(str(sample_file))
                    # Map samples sequentially across MIDI range
                    # For better mapping, would use filename patterns or config
                    pitch = (idx * 127 // max(1, len(sample_files))) if sample_files else 60
                    samples[pitch] = sample
                except Exception:
                    continue
            
            if not samples:
                continue
            
            # Process MIDI events and place samples
            time_ms = 0
            for msg in track:
                time_ms += msg.time * (60000.0 / (self.bpm * mid.ticks_per_beat))
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Find closest sample
                    if samples:
                        # Simple mapping: use any available sample
                        sample = list(samples.values())[msg.note % len(samples)]
                        
                        # Adjust volume based on velocity
                        velocity_db = (msg.velocity / 127.0) * 6 - 6  # -6dB to 0dB
                        adjusted_sample = sample + velocity_db
                        
                        # Mix into output at correct position
                        output_audio = output_audio.overlay(adjusted_sample, position=int(time_ms))
        
        # Export final audio
        output_audio.export(str(audio_path), format="wav")
        return audio_path
    
    def generate(self) -> tuple[Path, Path]:
        """Full generation: MIDI + audio."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in self.title if c.isalnum() or c in "_ -").strip()
        base_name = f"{timestamp}_{safe_title}"
        
        midi_path = OUTPUT_DIR / f"{base_name}.mid"
        
        # Generate and save MIDI
        midi = self.generate_midi()
        midi.save(str(midi_path))
        
        # Render audio
        audio_path = self.render_audio(midi_path)
        
        return midi_path, audio_path


# ============ FILE WATCHER ============
class IncomingHandler(FileSystemEventHandler):
    """Watches for incoming .json instruction files."""
    
    def __init__(self, app):
        self.app = app
        
    def on_created(self, event):
        if event.src_path.endswith('.json'):
            self.app.process_instruction(Path(event.src_path))


# ============ MENUBAR APP ============
class DAiWMenubarApp(rumps.App):
    """Menubar application for DAiW."""
    
    def __init__(self):
        super().__init__("🎵", quit_button=None)
        self.menu = [
            rumps.MenuItem("DAiW Listener", callback=None),
            None,  # Separator
            rumps.MenuItem("Open Output Folder", callback=self.open_output),
            rumps.MenuItem("Open Dropbox Folder", callback=self.open_dropbox),
            None,
            rumps.MenuItem("Status: Listening...", callback=None),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        
        # Start file watcher
        self.observer = Observer()
        handler = IncomingHandler(self)
        self.observer.schedule(handler, str(DROPBOX_DIR), recursive=False)
        self.observer.start()
        
    def process_instruction(self, json_path: Path):
        """Process an incoming instruction file."""
        try:
            self.title = "🎵⏳"  # Processing indicator
            
            with open(json_path) as f:
                config = json.load(f)
            
            generator = DAiWGenerator(config)
            midi_path, audio_path = generator.generate()
            
            # Notify
            rumps.notification(
                title="DAiW Complete",
                subtitle=config.get("title", "Song"),
                message=f"Saved to {midi_path.name}"
            )
            
            # Clean up instruction file
            json_path.unlink()
            
            self.title = "🎵"
            
        except Exception as e:
            rumps.notification(
                title="DAiW Error",
                subtitle="Generation failed",
                message=str(e)
            )
            self.title = "🎵❌"
    
    def open_output(self, _):
        os.system(f'open "{OUTPUT_DIR}"')
        
    def open_dropbox(self, _):
        os.system(f'open "{DROPBOX_DIR}"')
        
    def quit_app(self, _):
        self.observer.stop()
        self.observer.join()
        rumps.quit_application()


# ============ MAIN ============
if __name__ == "__main__":
    print(f"DAiW Listener starting...")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Dropbox: {DROPBOX_DIR}")
    print(f"Drop .json instruction files into the dropbox folder.")
    
    app = DAiWMenubarApp()
    app.run()
