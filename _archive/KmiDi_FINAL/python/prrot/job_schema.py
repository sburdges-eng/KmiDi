"""
Job Schema - PRROT/PARROT Job Input/Output Schemas

Defines JSON schemas for voice profile extraction jobs and
control data generation jobs.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path


class JobType(Enum):
    """Job types for PRROT/PARROT"""

    EXTRACT_VOICE_PROFILE = "extract_voice_profile"
    GENERATE_CONTROL_DATA = "generate_control_data"
    ANALYZE_ARTICULATION = "analyze_articulation"


class JobStatus(Enum):
    """Job status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VoiceProfileExtractionJob:
    """Job for extracting voice profile from reference audio"""

    job_id: str
    job_type: str = JobType.EXTRACT_VOICE_PROFILE.value

    # Input: reference audio paths
    reference_audio_paths: List[str]
    profile_id: str
    profile_name: Optional[str] = None

    # Optional: transcript text for alignment
    transcripts: Optional[List[str]] = None

    # Output paths (set by worker)
    output_profile_path: Optional[str] = None
    output_metadata_path: Optional[str] = None

    # Job metadata
    status: str = JobStatus.PENDING.value
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfileExtractionJob":
        """Create from dictionary"""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "VoiceProfileExtractionJob":
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ControlDataGenerationJob:
    """Job for generating control data from voice profile + lyrics"""

    job_id: str
    job_type: str = JobType.GENERATE_CONTROL_DATA.value

    # Input: voice profile
    voice_profile_id: str
    voice_profile_path: Optional[str] = None

    # Input: lyrics and melody
    lyrics: str
    melody_midi_notes: Optional[List[int]] = None  # MIDI note numbers
    melody_timing: Optional[List[float]] = None  # Timing in beats
    tempo_bpm: float = 120.0

    # Optional: expression parameters
    expression_params: Optional[Dict[str, Any]] = None

    # Output paths (set by worker)
    output_control_data_path: Optional[str] = None
    output_midi_path: Optional[str] = None
    output_automation_path: Optional[str] = None

    # Job metadata
    status: str = JobStatus.PENDING.value
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlDataGenerationJob":
        """Create from dictionary"""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ControlDataGenerationJob":
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ArticulationAnalysisJob:
    """Job for analyzing articulation from descriptive text or non-speech audio"""

    job_id: str
    job_type: str = JobType.ANALYZE_ARTICULATION.value

    # Input: descriptive text OR audio path
    descriptive_text: Optional[str] = None
    audio_path: Optional[str] = None

    # Output paths (set by worker)
    output_articulation_profile_path: Optional[str] = None
    output_instrument_affinity_path: Optional[str] = None

    # Job metadata
    status: str = JobStatus.PENDING.value
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticulationAnalysisJob":
        """Create from dictionary"""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ArticulationAnalysisJob":
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


def load_job_from_file(job_path: Path) -> Any:
    """Load job from JSON file"""
    with open(job_path, "r") as f:
        data = json.load(f)

    job_type = data.get("job_type")

    if job_type == JobType.EXTRACT_VOICE_PROFILE.value:
        return VoiceProfileExtractionJob.from_dict(data)
    elif job_type == JobType.GENERATE_CONTROL_DATA.value:
        return ControlDataGenerationJob.from_dict(data)
    elif job_type == JobType.ANALYZE_ARTICULATION.value:
        return ArticulationAnalysisJob.from_dict(data)
    else:
        raise ValueError(f"Unknown job type: {job_type}")


def save_job_to_file(job: Any, job_path: Path) -> None:
    """Save job to JSON file"""
    with open(job_path, "w") as f:
        f.write(job.to_json())
