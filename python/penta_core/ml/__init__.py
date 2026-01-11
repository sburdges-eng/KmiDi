"""
ML Model Integration - Machine Learning inference for iDAW.

Provides unified interfaces for:
- ONNX Runtime (cross-platform)
- TensorFlow Lite (mobile/edge)
- CoreML (macOS/iOS)
- PyTorch (training and inference)

Supports:
- Chord prediction models
- Style transfer for groove
- Emotion classification
- Audio feature extraction
- Multi-model training orchestration
- Dynamics training for emotion-aware processing
"""

from .model_registry import (
    ModelRegistry,
    ModelInfo,
    ModelBackend,
    ModelTask,
    register_model,
    get_model,
    get_registry,
    list_models,
    load_registry_manifest,
    # Training lifecycle
    TrainingStatus,
    TrainingJob,
    TrainingJobManager,
    get_job_manager,
    create_training_job,
)

from .inference import (
    InferenceEngine,
    InferenceResult,
    create_engine,
)

from .chord_predictor import (
    ChordPredictor,
    ChordPrediction,
    predict_next_chord,
    predict_progression,
)

from .style_transfer import (
    GrooveStyleTransfer,
    StyleTransferResult,
    transfer_groove_style,
)

from .gpu_utils import (
    get_available_devices,
    select_best_device,
    GPUDevice,
    DeviceType,
)

from .dynamics_training import (
    SectionType,
    DynamicLevel,
    SectionContext,
    TempoPoint,
    TempoMap,
    GroundTruthLabel,
    LUFSMeter,
    PositionTokenizer,
    calculate_dynamics_targets_from_emotion,
)

from .training_orchestrator import (
    TrainingConfig,
    OrchestratorConfig,
    TrainingCallback,
    ProgressCallback,
    CheckpointCallback,
    BaseTrainer,
    DummyTrainer,
    PyTorchTrainer,
    GrooveTrainer,
    HarmonyTrainer,
    EmotionTrainer,
    MelodyTrainer,
    TrainingOrchestrator,
    train_all_models,
    train_model,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelInfo",
    "ModelBackend",
    "ModelTask",
    "register_model",
    "get_model",
    "get_registry",
    "list_models",
    "load_registry_manifest",
    # Training Lifecycle
    "TrainingStatus",
    "TrainingJob",
    "TrainingJobManager",
    "get_job_manager",
    "create_training_job",
    # Inference
    "InferenceEngine",
    "InferenceResult",
    "create_engine",
    # Chord Prediction
    "ChordPredictor",
    "ChordPrediction",
    "predict_next_chord",
    "predict_progression",
    # Style Transfer
    "GrooveStyleTransfer",
    "StyleTransferResult",
    "transfer_groove_style",
    # GPU
    "get_available_devices",
    "select_best_device",
    "GPUDevice",
    "DeviceType",
    # Dynamics Training
    "SectionType",
    "DynamicLevel",
    "SectionContext",
    "TempoPoint",
    "TempoMap",
    "GroundTruthLabel",
    "LUFSMeter",
    "PositionTokenizer",
    "calculate_dynamics_targets_from_emotion",
    # Training Orchestrator
    "TrainingConfig",
    "OrchestratorConfig",
    "TrainingCallback",
    "ProgressCallback",
    "CheckpointCallback",
    "BaseTrainer",
    "DummyTrainer",
    "PyTorchTrainer",
    "GrooveTrainer",
    "HarmonyTrainer",
    "EmotionTrainer",
    "MelodyTrainer",
    "TrainingOrchestrator",
    "train_all_models",
    "train_model",
]
