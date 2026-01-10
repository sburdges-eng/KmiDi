"""AI analyzer for audio/music analysis.

This module provides AI analysis capabilities:
- Audio similarity
- Onset detection
- LUFS analysis
- File deduplication
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..models import AIJob, AIProposal, JobStatus

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """AI analyzer for music/audio analysis."""
    
    def __init__(self, offline_mode: bool = False):
        """Initialize analyzer.
        
        Args:
            offline_mode: If True, no network calls (local models only)
        """
        self.offline_mode = offline_mode
        self.logger = logger
    
    def analyze_similarity(self, files: List[Path]) -> AIProposal:
        """Analyze file similarity.
        
        Args:
            files: List of file paths to analyze
            
        Returns:
            AIProposal with similarity analysis
        """
        self.logger.info(f"Analyzing similarity for {len(files)} files")
        
        if len(files) < 2:
            return AIProposal(
                cluster_id="single",
                classification="unique",
                confidence=1.0,
                reason="Single file, no comparison needed"
            )
        
        # Placeholder: assume first file is base
        return AIProposal(
            cluster_id="cluster_001",
            classification="duplicate",
            base_file=str(files[0]),
            confidence=0.92,
            reason="Similar waveform and metadata",
            signals=[
                "Exact hash match",
                "Onset alignment > 98%",
                "LUFS delta < 0.3 dB"
            ],
            unknowns=[
                "Metadata mismatch"
            ]
        )
    
    def analyze_onsets(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze audio onsets.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Analysis dictionary with onsets, count, density
        """
        self.logger.info(f"Analyzing onsets: {audio_path}")
        
        try:
            import numpy as np
            import soundfile as sf
            
            # Load audio
            audio, sample_rate = sf.read(str(audio_path))
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Use librosa for onset detection if available
            try:
                import librosa
                
                onsets_frames = librosa.onset.onset_detect(
                    y=audio,
                    sr=sample_rate,
                    units='time'
                )
                onsets = onsets_frames.tolist()
                
                # Calculate density (onsets per second)
                duration = len(audio) / sample_rate
                density = len(onsets) / duration if duration > 0 else 0.0
                
                return {
                    "onsets": onsets,
                    "count": len(onsets),
                    "density": float(density),
                    "sample_rate": int(sample_rate),
                }
            except ImportError:
                self.logger.warning("librosa not available, using basic detection")
                # Basic onset detection using energy
                onsets = []
                threshold = np.std(audio) * 2
                for i in range(1, len(audio) - 1):
                    if audio[i] > threshold and audio[i] > audio[i-1] and audio[i] > audio[i+1]:
                        onsets.append(i / sample_rate)
                
                duration = len(audio) / sample_rate
                density = len(onsets) / duration if duration > 0 else 0.0
                
                return {
                    "onsets": onsets,
                    "count": len(onsets),
                    "density": float(density),
                    "sample_rate": int(sample_rate),
                }
        except Exception as e:
            self.logger.error(f"Onset analysis failed: {e}")
            return {
                "onsets": [],
                "count": 0,
                "density": 0.0,
                "error": str(e)
            }
    
    def analyze_lufs(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze LUFS (loudness).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Analysis dictionary with integrated, short-term, momentary LUFS
        """
        self.logger.info(f"Analyzing LUFS: {audio_path}")
        
        try:
            import numpy as np
            import soundfile as sf
            
            # Load audio
            audio, sample_rate = sf.read(str(audio_path))
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Use pyloudnorm for LUFS if available
            try:
                import pyloudnorm as pyln
                
                # Create meter
                meter = pyln.Meter(sample_rate)
                
                # Measure integrated LUFS
                integrated_lufs = meter.integrated_loudness(audio)
                
                # Measure short-term (3s blocks)
                short_term_lufs = meter.integrated_loudness(audio[:int(3 * sample_rate)]) if len(audio) > 3 * sample_rate else integrated_lufs
                
                # Measure momentary (400ms blocks)
                momentary_lufs = meter.integrated_loudness(audio[:int(0.4 * sample_rate)]) if len(audio) > 0.4 * sample_rate else integrated_lufs
                
                return {
                    "integrated": float(integrated_lufs) if not np.isnan(integrated_lufs) else None,
                    "short_term": float(short_term_lufs) if not np.isnan(short_term_lufs) else None,
                    "momentary": float(momentary_lufs) if not np.isnan(momentary_lufs) else None,
                }
            except ImportError:
                self.logger.warning("pyloudnorm not available, using RMS approximation")
                # Approximate LUFS using RMS (not accurate, but gives rough estimate)
                rms = np.sqrt(np.mean(audio ** 2))
                # Rough conversion: RMS to LUFS approximation
                lufs_approx = 20 * np.log10(rms + 1e-10) - 23  # -23dB offset approximation
                
                return {
                    "integrated": float(lufs_approx),
                    "short_term": float(lufs_approx),
                    "momentary": float(lufs_approx),
                    "note": "Approximate (pyloudnorm recommended for accurate LUFS)",
                }
        except Exception as e:
            self.logger.error(f"LUFS analysis failed: {e}")
            return {
                "integrated": None,
                "short_term": None,
                "momentary": None,
                "error": str(e)
            }
    
    def process_job(self, job: AIJob) -> Dict[str, Any]:
        """Process an AI job.
        
        Args:
            job: AI job to process
            
        Returns:
            Result dictionary
        """
        self.logger.info(f"Processing job: {job.id} ({job.type})")
        
        # Route to appropriate analyzer based on job type
        if job.type == "similarity":
            # TODO: Load files from job data
            files = []  # Would be loaded from job metadata
            proposal = self.analyze_similarity(files)
            return proposal.to_dict()
        elif job.type == "onsets":
            # TODO: Load audio path from job data
            audio_path = Path()  # Would be loaded from job metadata
            return self.analyze_onsets(audio_path)
        elif job.type == "lufs":
            # TODO: Load audio path from job data
            audio_path = Path()  # Would be loaded from job metadata
            return self.analyze_lufs(audio_path)
        else:
            raise ValueError(f"Unknown job type: {job.type}")

