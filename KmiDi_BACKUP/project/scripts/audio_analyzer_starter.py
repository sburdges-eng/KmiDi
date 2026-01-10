"""
DAiW Phase 2 - Audio Analysis Quick Start

This is a minimal audio analysis module to get Phase 2 started.
Full implementation will expand on this foundation.

Quick Test:
  python audio_analyzer_starter.py

Requirements:
  pip install librosa aubio numpy scipy --break-system-packages
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class AudioAnalysis:
    """Results from audio analysis"""
    filepath: str
    tempo_bpm: float
    key: Optional[str]
    duration_seconds: float
    frequency_profile: Dict[str, float]  # 8-band RMS levels
    dynamic_range_db: float
    beat_times: List[float]  # Beat positions in seconds
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export"""
        return {
            'filepath': self.filepath,
            'tempo_bpm': self.tempo_bpm,
            'key': self.key,
            'duration_seconds': self.duration_seconds,
            'frequency_profile': self.frequency_profile,
            'dynamic_range_db': self.dynamic_range_db,
            'beat_count': len(self.beat_times)
        }


class AudioAnalyzer:
    """
    Basic audio analysis for Phase 2.
    
    Analyzes audio files for:
    - Tempo (BPM)
    - Key detection (basic)
    - Frequency balance (8-band)
    - Dynamic range
    - Beat positions
    """
    
    # 8-band frequency ranges (Hz)
    FREQUENCY_BANDS = {
        'low_sub': (20, 60),      # Feel more than hear
        'sub_bass': (60, 120),    # Bass foundation
        'bass': (120, 250),       # Bass clarity
        'low_mid': (250, 500),    # Body/warmth
        'mid': (500, 2000),       # Core presence
        'high_mid': (2000, 4000), # Definition/clarity
        'presence': (4000, 8000), # Articulation/air
        'air': (8000, 20000)      # Sparkle/space
    }
    
    def __init__(self):
        """Initialize audio analyzer"""
        try:
            import librosa
            import numpy as np
            self.librosa = librosa
            self.np = np
            print("✓ Audio libraries loaded")
        except ImportError as e:
            print(f"❌ Missing library: {e}")
            print("Install with: pip install librosa aubio numpy scipy --break-system-packages")
            raise
    
    def analyze_file(self, audio_path: str) -> AudioAnalysis:
        """
        Analyze audio file for musical characteristics.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            AudioAnalysis with tempo, frequency, dynamics
        """
        print(f"\nAnalyzing: {audio_path}")
        
        # Load audio (22050 Hz is good balance of quality/speed)
        print("  Loading audio...")
        y, sr = self.librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        print(f"  Duration: {duration:.1f}s")
        
        # Tempo and beat detection
        print("  Detecting tempo & beats...")
        tempo, beat_frames = self.librosa.beat.beat_track(y=y, sr=sr)
        beat_times = self.librosa.frames_to_time(beat_frames, sr=sr)
        print(f"  Tempo: {tempo:.1f} BPM")
        print(f"  Beats detected: {len(beat_times)}")
        
        # Key detection (basic via chroma)
        print("  Detecting key...")
        key = self._detect_key_basic(y, sr)
        print(f"  Key: {key if key else 'Unknown'}")
        
        # Frequency analysis (8-band)
        print("  Analyzing frequency balance...")
        freq_profile = self._analyze_frequency_bands(y, sr)
        
        # Dynamic range
        print("  Analyzing dynamics...")
        dynamic_range = self._calculate_dynamic_range(y)
        print(f"  Dynamic range: {dynamic_range:.1f} dB")
        
        return AudioAnalysis(
            filepath=audio_path,
            tempo_bpm=float(tempo),
            key=key,
            duration_seconds=duration,
            frequency_profile=freq_profile,
            dynamic_range_db=dynamic_range,
            beat_times=beat_times.tolist()
        )
    
    def _detect_key_basic(self, y, sr) -> Optional[str]:
        """
        Basic key detection using chroma features.
        Returns note name (C, C#, D, etc.) or None.
        """
        try:
            # Get chroma (12 pitch classes)
            chroma = self.librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Average over time
            chroma_mean = self.np.mean(chroma, axis=1)
            
            # Find dominant pitch class
            dominant_pitch = self.np.argmax(chroma_mean)
            
            # Map to note names
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                         'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            return note_names[dominant_pitch]
        except Exception as e:
            print(f"    Key detection failed: {e}")
            return None
    
    def _analyze_frequency_bands(self, y, sr) -> Dict[str, float]:
        """
        Analyze 8-band frequency balance.
        Returns RMS level (dB) for each band.
        """
        # Compute STFT (Short-Time Fourier Transform)
        stft = self.librosa.stft(y)
        magnitude = self.np.abs(stft)
        
        # Frequency bins
        freqs = self.librosa.fft_frequencies(sr=sr)
        
        # Analyze each band
        profile = {}
        for band_name, (low_freq, high_freq) in self.FREQUENCY_BANDS.items():
            # Find bins in this frequency range
            mask = (freqs >= low_freq) & (freqs < high_freq)
            
            if self.np.any(mask):
                # RMS of magnitudes in this band
                band_magnitude = magnitude[mask, :]
                rms = self.np.sqrt(self.np.mean(band_magnitude ** 2))
                
                # Convert to dB
                rms_db = 20 * self.np.log10(rms + 1e-10)  # Avoid log(0)
                profile[band_name] = float(rms_db)
            else:
                profile[band_name] = -100.0  # Very quiet
        
        return profile
    
    def _calculate_dynamic_range(self, y) -> float:
        """
        Calculate dynamic range in dB.
        Returns difference between peak and RMS level.
        """
        # Peak level
        peak = self.np.max(self.np.abs(y))
        peak_db = 20 * self.np.log10(peak + 1e-10)
        
        # RMS level
        rms = self.np.sqrt(self.np.mean(y ** 2))
        rms_db = 20 * self.np.log10(rms + 1e-10)
        
        # Dynamic range = peak - rms
        return float(peak_db - rms_db)
    
    def compare_to_target(
        self,
        analysis: AudioAnalysis,
        target_tempo: float = 82.0,
        target_key: str = "F"
    ) -> dict:
        """
        Compare analysis results to target (e.g., Kelly song).
        
        Returns:
            dict with comparison results and recommendations
        """
        comparison = {
            'tempo_match': abs(analysis.tempo_bpm - target_tempo) < 10,
            'tempo_diff': analysis.tempo_bpm - target_tempo,
            'key_match': analysis.key == target_key if analysis.key else False,
            'recommendations': []
        }
        
        # Tempo recommendations
        if not comparison['tempo_match']:
            if comparison['tempo_diff'] > 0:
                comparison['recommendations'].append(
                    f"Reference is {comparison['tempo_diff']:.1f} BPM faster. "
                    f"Consider time-stretching or adjusting target tempo."
                )
            else:
                comparison['recommendations'].append(
                    f"Reference is {abs(comparison['tempo_diff']):.1f} BPM slower. "
                    f"May create more intimate feel if matched."
                )
        
        # Key recommendations
        if not comparison['key_match'] and analysis.key:
            comparison['recommendations'].append(
                f"Reference is in {analysis.key}, target is {target_key}. "
                f"Consider transposing reference or adjusting harmony."
            )
        
        return comparison


def print_analysis(analysis: AudioAnalysis):
    """Pretty-print audio analysis results"""
    print("\n" + "=" * 70)
    print("AUDIO ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"\nFile: {analysis.filepath}")
    print(f"Duration: {analysis.duration_seconds:.1f} seconds")
    print(f"Tempo: {analysis.tempo_bpm:.1f} BPM")
    print(f"Key: {analysis.key if analysis.key else 'Unknown'}")
    print(f"Dynamic Range: {analysis.dynamic_range_db:.1f} dB")
    print(f"Beats Detected: {len(analysis.beat_times)}")
    
    print("\nFrequency Balance (8-band, RMS dB):")
    print("-" * 70)
    
    # Sort by frequency (low to high)
    bands = [
        ('low_sub', 'Low Sub (20-60 Hz)'),
        ('sub_bass', 'Sub Bass (60-120 Hz)'),
        ('bass', 'Bass (120-250 Hz)'),
        ('low_mid', 'Low Mid (250-500 Hz)'),
        ('mid', 'Mid (500-2k Hz)'),
        ('high_mid', 'High Mid (2k-4k Hz)'),
        ('presence', 'Presence (4k-8k Hz)'),
        ('air', 'Air (8k-20k Hz)')
    ]
    
    for band_key, band_label in bands:
        level = analysis.frequency_profile.get(band_key, -100)
        # Visual bar (normalize to 0-100 range for display)
        normalized = max(0, min(100, (level + 100) * 1.0))
        bar = '█' * int(normalized / 5)
        print(f"{band_label:25} {level:6.1f} dB  {bar}")
    
    print("=" * 70)


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DAiW PHASE 2 - Audio Analysis Quick Start")
    print("=" * 70)
    
    # Check if libraries are installed
    try:
        import librosa
        print("\n✓ All required libraries installed")
    except ImportError:
        print("\n❌ Missing libraries!")
        print("Install with:")
        print("  pip install librosa aubio numpy scipy --break-system-packages")
        exit(1)
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Test with synthetic audio (create a simple sine wave)
    print("\nCreating test audio file...")
    import numpy as np
    import scipy.io.wavfile as wavfile
    
    # Generate 5 seconds of 440 Hz tone (A4) at 82 BPM feel
    sample_rate = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of frequencies to simulate music
    signal = (
        0.4 * np.sin(2 * np.pi * 82 * t) +      # Low (bass)
        0.3 * np.sin(2 * np.pi * 220 * t) +     # Mid (guitar)
        0.2 * np.sin(2 * np.pi * 440 * t) +     # Mid-high (melody)
        0.1 * np.sin(2 * np.pi * 2000 * t)      # High (air)
    )
    
    # Add some dynamic variation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.37 * t)  # ~82 BPM pulse
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save test file
    test_file = "/home/claude/test_audio.wav"
    wavfile.write(test_file, sample_rate, (signal * 32767).astype(np.int16))
    print(f"✓ Created: {test_file}")
    
    # Analyze it
    analysis = analyzer.analyze_file(test_file)
    
    # Print results
    print_analysis(analysis)
    
    # Compare to Kelly song target
    print("\n" + "=" * 70)
    print("COMPARISON TO KELLY SONG TARGET")
    print("=" * 70)
    
    comparison = analyzer.compare_to_target(
        analysis,
        target_tempo=82.0,
        target_key="F"
    )
    
    print(f"\nTempo match: {'✓' if comparison['tempo_match'] else '✗'}")
    print(f"Tempo difference: {comparison['tempo_diff']:+.1f} BPM")
    print(f"Key match: {'✓' if comparison['key_match'] else '✗'}")
    
    if comparison['recommendations']:
        print("\nRecommendations:")
        for rec in comparison['recommendations']:
            print(f"  • {rec}")
    
    # Save analysis to JSON
    output_file = "/mnt/user-data/outputs/test_audio_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis.to_dict(), f, indent=2)
    print(f"\n✓ Analysis saved: {output_file}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 AUDIO ANALYSIS - READY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Analyze your reference tracks:")
    print("     analysis = analyzer.analyze_file('elliott_smith.wav')")
    print()
    print("  2. Compare to Kelly song target:")
    print("     comparison = analyzer.compare_to_target(analysis, 82.0, 'F')")
    print()
    print("  3. Use frequency profile for production decisions")
    print()
    print("  4. See PHASE_2_PLAN.md for full implementation roadmap")
    print("=" * 70 + "\n")
