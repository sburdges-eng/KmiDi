"""
MIDI to Audio Renderer

Converts MIDI files to audio formats (WAV, MP3) using available synthesis engines.
Supports multiple backends: FluidSynth, pyFluidSynth, or DAW routing.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import subprocess
import shutil

logger = logging.getLogger(__name__)


class AudioRenderer:
    """
    Renders MIDI files to audio formats.
    
    Supports multiple backends:
    - FluidSynth (via command-line or pyFluidSynth)
    - SoundFont-based synthesis
    - DAW routing (future)
    """
    
    def __init__(self, soundfont_path: Optional[str] = None):
        """
        Initialize audio renderer.
        
        Args:
            soundfont_path: Path to SoundFont file (.sf2). If None, tries to find default.
        """
        self.soundfont_path = soundfont_path or self._find_default_soundfont()
        self.fluid_synth_available = self._check_fluid_synth()
        
    def _find_default_soundfont(self) -> Optional[str]:
        """Try to find a default SoundFont file."""
        # Common locations for SoundFonts
        common_paths = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default.sf2",
            "/usr/local/share/sounds/sf2/FluidR3_GM.sf2",
            "~/.local/share/sounds/sf2/default.sf2",
            "/Library/Audio/Sounds/Banks/fluid_soundfont_GM.sf2",  # macOS
        ]
        
        for path_str in common_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                logger.info(f"Found SoundFont at {path}")
                return str(path)
        
        logger.warning("No default SoundFont found. Audio rendering may not work.")
        return None
    
    def _check_fluid_synth(self) -> bool:
        """Check if FluidSynth is available."""
        # Check for command-line fluidsynth
        if shutil.which("fluidsynth"):
            return True
        
        # Check for pyFluidSynth
        try:
            import fluidsynth
            return True
        except ImportError:
            pass
        
        return False
    
    def render_midi_to_wav(
        self,
        midi_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 44100,
        gain: float = 0.2,
        **kwargs
    ) -> str:
        """
        Render MIDI file to WAV audio.
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output WAV file. If None, auto-generates.
            sample_rate: Audio sample rate (default 44100)
            gain: Audio gain/volume (0.0-1.0, default 0.2)
            **kwargs: Additional rendering options
            
        Returns:
            Path to generated WAV file
            
        Raises:
            RuntimeError: If rendering fails
        """
        midi_file = Path(midi_path)
        if not midi_file.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        if output_path is None:
            output_path = str(midi_file.with_suffix(".wav"))
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Try pyFluidSynth first (more reliable)
        if self._try_pyfluidsynth(midi_path, str(output_file), sample_rate, gain):
            logger.info(f"Rendered MIDI to WAV using pyFluidSynth: {output_file}")
            return str(output_file)
        
        # Fall back to command-line fluidsynth
        if self._try_fluidsynth_cli(midi_path, str(output_file), sample_rate, gain):
            logger.info(f"Rendered MIDI to WAV using fluidsynth CLI: {output_file}")
            return str(output_file)
        
        raise RuntimeError(
            "Failed to render MIDI to audio. Install FluidSynth:\n"
            "  macOS: brew install fluidsynth\n"
            "  Linux: apt-get install fluidsynth\n"
            "  Or: pip install pyfluidsynth"
        )
    
    def _try_pyfluidsynth(
        self,
        midi_path: str,
        output_path: str,
        sample_rate: int,
        gain: float
    ) -> bool:
        """Try rendering with pyFluidSynth library."""
        try:
            import fluidsynth
            
            if not self.soundfont_path:
                logger.error("No SoundFont available for pyFluidSynth")
                return False
            
            # Create FluidSynth instance
            fs = fluidsynth.Synth(samplerate=float(sample_rate), gain=gain)
            fs.start()
            
            # Load SoundFont
            sfid = fs.sfload(self.soundfont_path)
            if sfid == -1:
                logger.error(f"Failed to load SoundFont: {self.soundfont_path}")
                fs.delete()
                return False
            
            # Select soundfont for all channels
            for channel in range(16):
                fs.program_select(channel, sfid, 0, 0)
            
            # Read MIDI file
            import mido
            mid = mido.MidiFile(midi_path)
            
            # Render to audio
            import numpy as np
            samples = []
            current_time = 0.0
            tempo = 500000  # Default 120 BPM
            
            for msg in mid:
                current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                elif msg.type == "note_on":
                    fs.noteon(msg.channel, msg.note, msg.velocity)
                elif msg.type == "note_off":
                    fs.noteoff(msg.channel, msg.note)
                elif msg.type == "control_change":
                    fs.cc(msg.channel, msg.control, msg.value)
                elif msg.type == "program_change":
                    fs.program_change(msg.channel, msg.program)
                
                # Generate audio samples
                sample_count = int(current_time * sample_rate) - len(samples)
                if sample_count > 0:
                    audio = fs.get_samples(sample_count)
                    samples.extend(audio)
            
            # Final samples
            remaining = int(mid.length * sample_rate) - len(samples)
            if remaining > 0:
                audio = fs.get_samples(remaining)
                samples.extend(audio)
            
            # Convert to numpy array and normalize
            audio_array = np.array(samples, dtype=np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            
            # Write WAV file
            import soundfile as sf
            sf.write(output_path, audio_array, sample_rate)
            
            fs.delete()
            return True
            
        except ImportError:
            logger.debug("pyFluidSynth not available")
            return False
        except Exception as e:
            logger.warning(f"pyFluidSynth rendering failed: {e}")
            return False
    
    def _try_fluidsynth_cli(
        self,
        midi_path: str,
        output_path: str,
        sample_rate: int,
        gain: float
    ) -> bool:
        """Try rendering with command-line fluidsynth."""
        if not shutil.which("fluidsynth"):
            return False
        
        if not self.soundfont_path:
            logger.error("No SoundFont available for fluidsynth CLI")
            return False
        
        try:
            # Build fluidsynth command
            cmd = [
                "fluidsynth",
                "-F", output_path,  # Output file
                "-r", str(sample_rate),  # Sample rate
                "-g", str(gain),  # Gain
                "-q",  # Quiet mode
                self.soundfont_path,
                midi_path,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0 and Path(output_path).exists():
                return True
            else:
                logger.warning(f"fluidsynth CLI failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("fluidsynth CLI timed out")
            return False
        except Exception as e:
            logger.warning(f"fluidsynth CLI rendering failed: {e}")
            return False
    
    def render_midi_to_mp3(
        self,
        midi_path: str,
        output_path: Optional[str] = None,
        bitrate: int = 192,
        **kwargs
    ) -> str:
        """
        Render MIDI file to MP3 audio.
        
        First renders to WAV, then converts to MP3 using ffmpeg or similar.
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output MP3 file. If None, auto-generates.
            bitrate: MP3 bitrate in kbps (default 192)
            **kwargs: Additional rendering options
            
        Returns:
            Path to generated MP3 file
            
        Raises:
            RuntimeError: If rendering fails
        """
        midi_file = Path(midi_path)
        if output_path is None:
            output_path = str(midi_file.with_suffix(".mp3"))
        
        # First render to WAV
        temp_wav = str(Path(tempfile.gettempdir()) / f"temp_{midi_file.stem}.wav")
        try:
            wav_path = self.render_midi_to_wav(midi_path, temp_wav, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to render MIDI to WAV: {e}") from e
        
        # Convert WAV to MP3 using ffmpeg
        if shutil.which("ffmpeg"):
            try:
                cmd = [
                    "ffmpeg",
                    "-i", wav_path,
                    "-codec:a", "libmp3lame",
                    "-b:a", f"{bitrate}k",
                    "-y",  # Overwrite output
                    output_path,
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                # Clean up temp WAV
                Path(wav_path).unlink(missing_ok=True)
                
                if result.returncode == 0 and Path(output_path).exists():
                    logger.info(f"Rendered MIDI to MP3: {output_path}")
                    return output_path
                else:
                    raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                raise RuntimeError("ffmpeg conversion timed out")
            except Exception as e:
                raise RuntimeError(f"ffmpeg conversion failed: {e}") from e
        else:
            # No ffmpeg, return WAV instead
            logger.warning("ffmpeg not found, returning WAV instead of MP3")
            Path(output_path).unlink(missing_ok=True)
            Path(wav_path).rename(output_path)
            return output_path.replace(".mp3", ".wav")


# Global renderer instance
_default_renderer: Optional[AudioRenderer] = None


def get_renderer(soundfont_path: Optional[str] = None) -> AudioRenderer:
    """Get or create default audio renderer."""
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = AudioRenderer(soundfont_path=soundfont_path)
    return _default_renderer


def render_midi_to_audio(
    midi_path: str,
    output_path: Optional[str] = None,
    format: str = "wav",
    **kwargs
) -> str:
    """
    Convenience function to render MIDI to audio.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path for output file. If None, auto-generates.
        format: Output format ("wav" or "mp3")
        **kwargs: Additional rendering options
        
    Returns:
        Path to generated audio file
    """
    renderer = get_renderer()
    
    if format.lower() == "wav":
        return renderer.render_midi_to_wav(midi_path, output_path, **kwargs)
    elif format.lower() == "mp3":
        return renderer.render_midi_to_mp3(midi_path, output_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'wav' or 'mp3'")
