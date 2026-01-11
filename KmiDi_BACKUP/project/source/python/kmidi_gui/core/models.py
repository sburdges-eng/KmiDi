"""Data models for KmiDi application.

All models are pure data structures with no GUI dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    """AI job status."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence level for AI suggestions."""
    HIGH = "high"      # >= 0.95
    MEDIUM = "medium"  # 0.80 - 0.95
    LOW = "low"        # < 0.80


@dataclass
class AIJob:
    """AI analysis job."""
    id: str
    type: str
    input_hash: str
    status: JobStatus = JobStatus.PENDING
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "input_hash": self.input_hash,
            "status": self.status.value,
            "result_path": self.result_path,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIJob":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            input_hash=data["input_hash"],
            status=JobStatus(data["status"]),
            result_path=data.get("result_path"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class AIProposal:
    """AI-generated proposal (never auto-applied)."""
    cluster_id: str
    classification: str  # "duplicate", "unique", "merge", etc.
    base_file: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    signals: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "classification": self.classification,
            "base_file": self.base_file,
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": self.signals,
            "unknowns": self.unknowns,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIProposal":
        """Deserialize from dictionary."""
        return cls(
            cluster_id=data["cluster_id"],
            classification=data["classification"],
            base_file=data.get("base_file"),
            confidence=data["confidence"],
            reason=data["reason"],
            signals=data.get("signals", []),
            unknowns=data.get("unknowns", []),
        )
    
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category."""
        if self.confidence >= 0.95:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.80:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class Decision:
    """Human-in-the-loop decision record."""
    decision_id: str
    ai_job: str
    proposal: AIProposal
    approved: bool = False
    executed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision_id": self.decision_id,
            "ai_job": self.ai_job,
            "proposal": self.proposal.to_dict(),
            "approved": self.approved,
            "executed": self.executed,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        """Deserialize from dictionary."""
        return cls(
            decision_id=data["decision_id"],
            ai_job=data["ai_job"],
            proposal=AIProposal.from_dict(data["proposal"]),
            approved=data["approved"],
            executed=data["executed"],
            created_at=datetime.fromisoformat(data["created_at"]),
            executed_at=(
                datetime.fromisoformat(data["executed_at"])
                if data.get("executed_at") else None
            ),
        )


@dataclass
class EmotionIntent:
    """Emotional intent for music generation."""
    core_event: Optional[str] = None
    core_resistance: Optional[str] = None
    core_longing: Optional[str] = None
    mood_primary: Optional[str] = None
    vulnerability_scale: float = 0.5
    narrative_arc: Optional[str] = None
    technical_genre: Optional[str] = None
    technical_key: Optional[str] = None
    technical_bpm: Optional[int] = None
    technical_rule_to_break: Optional[str] = None  # noqa: E501
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "core_event": self.core_event,
            "core_resistance": self.core_resistance,
            "core_longing": self.core_longing,
            "mood_primary": self.mood_primary,
            "vulnerability_scale": self.vulnerability_scale,
            "narrative_arc": self.narrative_arc,
            "technical_genre": self.technical_genre,
            "technical_key": self.technical_key,
            "technical_bpm": self.technical_bpm,
            "technical_rule_to_break": self.technical_rule_to_break,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionIntent":
        """Deserialize from dictionary."""
        return cls(
            core_event=data.get("core_event"),
            core_resistance=data.get("core_resistance"),
            core_longing=data.get("core_longing"),
            mood_primary=data.get("mood_primary"),
            vulnerability_scale=data.get("vulnerability_scale", 0.5),
            narrative_arc=data.get("narrative_arc"),
            technical_genre=data.get("technical_genre"),
            technical_key=data.get("technical_key"),
            technical_bpm=data.get("technical_bpm"),
            technical_rule_to_break=data.get("technical_rule_to_break"),
        )


@dataclass
class GenerationResult:
    """Result from music generation."""
    success: bool
    midi_path: Optional[str] = None
    chords: List[str] = field(default_factory=list)
    key: Optional[str] = None
    tempo: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "midi_path": self.midi_path,
            "chords": self.chords,
            "key": self.key,
            "tempo": self.tempo,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """Deserialize from dictionary."""
        return cls(
            success=data["success"],
            midi_path=data.get("midi_path"),
            chords=data.get("chords", []),
            key=data.get("key"),
            tempo=data.get("tempo"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

