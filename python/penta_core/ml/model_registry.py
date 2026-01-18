"""
Model Registry - Unified model discovery and management.

Provides a centralized registry for ML models across different backends.
Enhanced with training lifecycle management for ML training orchestration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json
import os

try:
    import jsonschema
except ImportError:
    jsonschema = None


class ModelBackend(Enum):
    """Supported ML backends."""
    ONNX = "onnx"
    TENSORFLOW_LITE = "tflite"
    COREML = "coreml"
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    RTNEURAL_JSON = "rtneural-json"


class ModelTask(Enum):
    """Supported ML tasks."""
    # Core prediction tasks
    CHORD_PREDICTION = "chord_prediction"
    CHORD_DETECTION = "chord_detection"
    KEY_DETECTION = "key_detection"
    TEMPO_ESTIMATION = "tempo_estimation"
    STYLE_TRANSFER = "style_transfer"
    EMOTION_CLASSIFICATION = "emotion_classification"
    AUDIO_GENERATION = "audio_generation"
    ONSET_DETECTION = "onset_detection"
    BEAT_TRACKING = "beat_tracking"
    # Enhanced ML tasks for penta-core
    EMOTION_EMBEDDING = "emotion_embedding"
    MELODY_GENERATION = "melody_generation"
    HARMONY_PREDICTION = "harmony_prediction"
    DYNAMICS_MAPPING = "dynamics_mapping"
    GROOVE_PREDICTION = "groove_prediction"
    INTENT_MAPPING = "intent_mapping"
    AUDIO_CLASSIFICATION = "audio_classification"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    task: ModelTask
    backend: ModelBackend
    path: str
    version: str = "1.0.0"

    # Model metadata
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    sample_rate: Optional[int] = None

    # Performance info
    latency_ms: Optional[float] = None
    memory_mb: Optional[float] = None

    # Additional metadata
    description: str = ""
    author: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task": self.task.value,
            "backend": self.backend.value,
            "path": self.path,
            "version": self.version,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "sample_rate": self.sample_rate,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            task=ModelTask(data["task"]),
            backend=ModelBackend(data["backend"]),
            path=data["path"],
            version=data.get("version", "1.0.0"),
            input_shape=data.get("input_shape"),
            output_shape=data.get("output_shape"),
            sample_rate=data.get("sample_rate"),
            latency_ms=data.get("latency_ms"),
            memory_mb=data.get("memory_mb"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
            tags=data.get("tags", []),
        )


class ModelRegistry:
    """
    Centralized registry for ML models.

    Manages model discovery, loading, and caching.
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[tuple, ModelInfo] = {}
        self._model_dirs: List[Path] = []
        self._cache: Dict[str, Any] = {}
        self._initialized = True

        # Add default model directories
        self._add_default_dirs()

    def _add_default_dirs(self) -> None:
        """Add default model search directories."""
        # Project model directory
        project_dir = Path(__file__).parent.parent.parent.parent / "Data_Files" / "models"
        if project_dir.exists():
            self._model_dirs.append(project_dir)

        # User model directory
        user_dir = Path.home() / ".idaw" / "models"
        if user_dir.exists():
            self._model_dirs.append(user_dir)

    def add_model_dir(self, path: str) -> None:
        """Add a directory to search for models."""
        model_dir = Path(path)
        if model_dir.exists() and model_dir not in self._model_dirs:
            self._model_dirs.append(model_dir)

    @staticmethod
    def _version_key(version: str) -> tuple:
        parts = []
        for part in (version or "0").split("."):
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(part)
        return tuple(parts)

    def register(self, model: ModelInfo) -> None:
        """Register a model."""
        key = (model.name, model.version)
        self._models[key] = model

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
        """Unregister a model by name or name+version."""
        if version is not None:
            key = (name, version)
            if key in self._models:
                del self._models[key]
                return True
            return False

        to_delete = [key for key in self._models if key[0] == name]
        for key in to_delete:
            del self._models[key]
        return bool(to_delete)

    def get(self, name: str, version: Optional[str] = None) -> Optional[ModelInfo]:
        """Get model info by name and optional version."""
        if version is not None:
            return self._models.get((name, version))

        candidates = [model for (model_name, _), model in self._models.items() if model_name == name]
        if not candidates:
            return None
        return max(candidates, key=lambda model: self._version_key(model.version))

    def list(self, task: Optional[ModelTask] = None) -> List[ModelInfo]:
        """List all registered models, optionally filtered by task."""
        models = list(self._models.values())
        if task:
            models = [m for m in models if m.task == task]
        return models

    def discover(self) -> int:
        """
        Discover models in registered directories.

        Returns:
            Number of models discovered
        """
        count = 0

        for model_dir in self._model_dirs:
            # Look for model manifest files
            for manifest_path in model_dir.glob("**/model_info.json"):
                try:
                    with open(manifest_path) as f:
                        data = json.load(f)

                    # Update path to be absolute
                    if not Path(data["path"]).is_absolute():
                        data["path"] = str(manifest_path.parent / data["path"])

                    model = ModelInfo.from_dict(data)
                    self.register(model)
                    count += 1
                except Exception:
                    continue

            # Auto-discover by file extension
            for ext, backend in [
                (".onnx", ModelBackend.ONNX),
                (".tflite", ModelBackend.TENSORFLOW_LITE),
                (".mlmodel", ModelBackend.COREML),
                (".pt", ModelBackend.PYTORCH),
                (".pth", ModelBackend.PYTORCH),
            ]:
                for model_path in model_dir.glob(f"**/*{ext}"):
                    name = model_path.stem
                    if name not in self._models:
                        # Infer task from directory name or file name
                        task = self._infer_task(model_path)
                        model = ModelInfo(
                            name=name,
                            task=task,
                            backend=backend,
                            path=str(model_path),
                        )
                        self.register(model)
                        count += 1

        return count

    def _infer_task(self, path: Path) -> ModelTask:
        """Infer model task from path."""
        path_str = str(path).lower()

        if "chord" in path_str:
            if "detect" in path_str:
                return ModelTask.CHORD_DETECTION
            return ModelTask.CHORD_PREDICTION
        elif "key" in path_str:
            return ModelTask.KEY_DETECTION
        elif "tempo" in path_str or "bpm" in path_str:
            return ModelTask.TEMPO_ESTIMATION
        elif "style" in path_str or "transfer" in path_str:
            return ModelTask.STYLE_TRANSFER
        elif "emotion" in path_str or "mood" in path_str:
            return ModelTask.EMOTION_CLASSIFICATION
        elif "onset" in path_str:
            return ModelTask.ONSET_DETECTION
        elif "beat" in path_str:
            return ModelTask.BEAT_TRACKING

        return ModelTask.CHORD_PREDICTION  # Default

    def save_registry(self, path: str) -> None:
        """Save registry to JSON file."""
        data = {
            "models": [m.to_dict() for m in self._models.values()],
            "model_dirs": [str(d) for d in self._model_dirs],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_registry(self, path: str) -> None:
        """Load registry from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # If this looks like the new manifest format, delegate to manifest loader
        if "registry_version" in data or "$schema" in data:
            self.load_registry_manifest(path, validate=False)
            return

        for model_data in data.get("models", []):
            model = ModelInfo.from_dict(model_data)
            self.register(model)

        for dir_path in data.get("model_dirs", []):
            self.add_model_dir(dir_path)

    # ------------------------------------------------------------------ #
    # Manifest loader (registry.json / registry.schema.json)
    # ------------------------------------------------------------------ #
    def load_registry_manifest(
        self,
        path: str,
        validate: bool = True,
        schema_path: Optional[str] = None,
    ) -> int:
        """
        Load a modern registry.json with schema validation.

        Returns:
            Number of models registered.
        """
        manifest_path = Path(path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if validate and jsonschema is not None:
            schema_file = (
                Path(schema_path)
                if schema_path
                else manifest_path.parent / "registry.schema.json"
            )
            if schema_file.exists():
                with open(schema_file, "r", encoding="utf-8") as sf:
                    schema = json.load(sf)
                jsonschema.validate(instance=manifest, schema=schema)

        base_dir = manifest_path.parent
        count = 0
        for entry in manifest.get("models", []):
            try:
                model = self._modelinfo_from_manifest_entry(entry, base_dir)
            except Exception:
                continue
            self.register(model)
            count += 1
        return count

    # Helpers ----------------------------------------------------------- #
    def _map_backend(self, fmt: str) -> ModelBackend:
        fmt = (fmt or "").lower()
        if fmt in ("onnx",):
            return ModelBackend.ONNX
        if fmt in ("coreml", "mlmodel"):
            return ModelBackend.COREML
        if fmt in ("pytorch", "torchscript"):
            return ModelBackend.PYTORCH
        if fmt in ("rtneural-json", "rtneural"):
            return ModelBackend.RTNEURAL_JSON
        if fmt in ("tflite", "tensorflow"):
            return ModelBackend.TENSORFLOW_LITE
        return ModelBackend.ONNX

    def _map_task(self, task: str) -> ModelTask:
        value = (task or "custom").lower()
        for t in ModelTask:
            if t.value == value:
                return t
        return ModelTask.CUSTOM

    def _resolve_path(self, base_dir: Path, path_str: str) -> str:
        if not path_str:
            return ""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((base_dir / p).resolve())

    def _modelinfo_from_manifest_entry(self, entry: Dict[str, Any], base_dir: Path) -> ModelInfo:
        file_path = (
            entry.get("onnx_path")
            or entry.get("coreml_path")
            or entry.get("file")
            or ""
        )
        resolved_path = self._resolve_path(base_dir, file_path) if file_path else ""

        input_size = entry.get("input_size")
        output_size = entry.get("output_size")

        return ModelInfo(
            name=entry["id"],
            task=self._map_task(entry.get("task")),
            backend=self._map_backend(entry.get("format")),
            path=resolved_path,
            version=entry.get("version", "1.0.0"),
            input_shape=[input_size] if input_size else None,
            output_shape=[output_size] if output_size else None,
            sample_rate=entry.get("sample_rate"),
            latency_ms=entry.get("inference_target_ms"),
            description=entry.get("note", ""),
            license=entry.get("license", ""),
            tags=[entry.get("status", "")] if entry.get("status") else [],
        )


# Singleton access functions
def get_registry() -> ModelRegistry:
    """Get the model registry singleton."""
    return ModelRegistry()


def register_model(model: ModelInfo) -> None:
    """Register a model in the global registry."""
    get_registry().register(model)


def get_model(name: str, version: Optional[str] = None) -> Optional[ModelInfo]:
    """Get a model from the global registry."""
    return get_registry().get(name, version=version)


def list_models(task: Optional[ModelTask] = None) -> List[ModelInfo]:
    """List models in the global registry."""
    return get_registry().list(task)


def load_registry_manifest(
    path: str,
    validate: bool = True,
    schema_path: Optional[str] = None,
) -> int:
    """
    Load registry.json (schema-based) into the global registry.

    Returns:
        Number of models registered.
    """
    return get_registry().load_registry_manifest(path, validate=validate, schema_path=schema_path)


# =============================================================================
# ENHANCED: Training Lifecycle Management
# =============================================================================

class TrainingStatus(Enum):
    """Training lifecycle status."""
    PENDING = "pending"
    QUEUED = "queued"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Tracks a training job lifecycle."""
    job_id: str
    model_name: str
    model_task: ModelTask
    status: TrainingStatus = TrainingStatus.PENDING

    # Training configuration
    target_epochs: int = 30
    current_epoch: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Metrics
    best_loss: float = float('inf')
    best_accuracy: float = 0.0
    current_loss: float = 0.0

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Paths
    checkpoint_dir: str = ""
    log_file: str = ""

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "model_task": self.model_task.value,
            "status": self.status.value,
            "target_epochs": self.target_epochs,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "current_loss": self.current_loss,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "checkpoint_dir": self.checkpoint_dir,
            "log_file": self.log_file,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        data = dict(data)  # Copy to avoid mutating input
        data["model_task"] = ModelTask(data["model_task"])
        data["status"] = TrainingStatus(data["status"])
        # Filter to only known fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


class TrainingJobManager:
    """
    Manages training job lifecycle across sessions.

    Persists job state to disk for resumable training.
    """

    _instance: Optional["TrainingJobManager"] = None

    def __new__(cls) -> "TrainingJobManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._jobs: Dict[str, TrainingJob] = {}
        self._jobs_dir = Path.home() / ".kelly" / "training_jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_file = self._jobs_dir / "jobs.json"
        self._initialized = True
        self._load_jobs()

    def _load_jobs(self):
        """Load jobs from disk."""
        if self._jobs_file.exists():
            try:
                with open(self._jobs_file, "r") as f:
                    data = json.load(f)
                for job_id, job_data in data.get("jobs", {}).items():
                    self._jobs[job_id] = TrainingJob.from_dict(job_data)
            except Exception:
                self._jobs = {}

    def _save_jobs(self):
        """Save jobs to disk."""
        data = {
            "jobs": {jid: job.to_dict() for jid, job in self._jobs.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._jobs_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_job(
        self,
        model_name: str,
        model_task: ModelTask,
        target_epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> TrainingJob:
        """Create a new training job."""
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:8]}"

        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            model_task=model_task,
            target_epochs=target_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=str(self._jobs_dir / job_id / "checkpoints"),
            log_file=str(self._jobs_dir / job_id / "training.log"),
        )

        # Create directories
        Path(job.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._jobs[job_id] = job
        self._save_jobs()
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        model_task: Optional[ModelTask] = None,
    ) -> List[TrainingJob]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        if model_task:
            jobs = [j for j in jobs if j.model_task == model_task]
        return jobs

    def update_progress(
        self,
        job_id: str,
        epoch: int,
        loss: float,
        accuracy: Optional[float] = None,
    ):
        """Update job progress."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.current_epoch = epoch
        job.current_loss = loss

        if loss < job.best_loss:
            job.best_loss = loss
        if accuracy is not None and accuracy > job.best_accuracy:
            job.best_accuracy = accuracy

        self._save_jobs()

    def start_job(self, job_id: str):
        """Mark job as started."""
        job = self._jobs.get(job_id)
        if job:
            job.status = TrainingStatus.TRAINING
            job.started_at = datetime.now().isoformat()
            self._save_jobs()

    def complete_job(self, job_id: str, success: bool = True, error: Optional[str] = None):
        """Mark job as completed or failed."""
        job = self._jobs.get(job_id)
        if job:
            job.status = TrainingStatus.COMPLETED if success else TrainingStatus.FAILED
            job.completed_at = datetime.now().isoformat()
            if error:
                job.error_message = error
            self._save_jobs()

    def cancel_job(self, job_id: str):
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if job and job.status in (TrainingStatus.PENDING, TrainingStatus.QUEUED, TrainingStatus.TRAINING):
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            self._save_jobs()

    def get_resumable_jobs(self) -> List[TrainingJob]:
        """Get jobs that can be resumed."""
        return [
            j for j in self._jobs.values()
            if j.status == TrainingStatus.TRAINING
            and j.current_epoch < j.target_epochs
        ]


def get_job_manager() -> TrainingJobManager:
    """Get the training job manager singleton."""
    return TrainingJobManager()


def create_training_job(
    model_name: str,
    model_task: ModelTask,
    **kwargs
) -> TrainingJob:
    """Create a new training job."""
    return get_job_manager().create_job(model_name, model_task, **kwargs)
