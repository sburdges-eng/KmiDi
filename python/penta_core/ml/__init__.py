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

# New modular AI system components
try:
    from .resource_manager import (
        ResourceManager,
        ResourceType,
        ResourceQuota,
        ResourceAllocation,
        get_resource_manager,
    )
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False

try:
    from .error_handling import (
        ErrorCategory,
        ErrorInfo,
        ErrorClassifier,
        RetryConfig,
        retry_with_backoff,
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        FallbackChain,
    )
    HAS_ERROR_HANDLING = True
except ImportError:
    HAS_ERROR_HANDLING = False

try:
    from .health import (
        HealthStatus,
        HealthCheck,
        HealthCheckResult,
        ModelHealthCheck,
        ResourceHealthCheck,
        CustomHealthCheck,
        HealthMonitor,
        get_health_monitor,
    )
    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False

try:
    from .ai_service import (
        AIService,
        ServiceStatus,
        ServiceInfo,
        get_ai_service,
    )
    HAS_AI_SERVICE = True
except ImportError:
    HAS_AI_SERVICE = False

try:
    from .model_pool import (
        ModelPool,
        PooledModel,
        EvictionPolicy,
        get_model_pool,
    )
    HAS_MODEL_POOL = True
except ImportError:
    HAS_MODEL_POOL = False

try:
    from .async_inference import (
        AsyncInferenceEngine,
        InferenceRequest,
        InferenceResponse,
        get_async_inference_engine,
    )
    HAS_ASYNC_INFERENCE = True
except ImportError:
    HAS_ASYNC_INFERENCE = False

try:
    from .inference_batching import (
        BatchProcessor,
        BatchConfig,
        BatchedInferenceEngine,
    )
    HAS_BATCHING = True
except ImportError:
    HAS_BATCHING = False

try:
    from .inference_enhanced import (
        EnhancedInferenceEngine,
        create_enhanced_engine,
        create_enhanced_engine_by_name,
    )
    HAS_ENHANCED_INFERENCE = True
except ImportError:
    HAS_ENHANCED_INFERENCE = False

try:
    from .integration_manager import (
        IntegrationManager,
        IntegrationStatus,
        IntegrationInfo,
        IntegrationComponent,
        get_integration_manager,
    )
    HAS_INTEGRATION_MANAGER = True
except ImportError:
    HAS_INTEGRATION_MANAGER = False

try:
    from .event_bus import (
        EventBus,
        Event,
        get_event_bus,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

try:
    from .training_inference_bridge import (
        TrainingInferenceBridge,
        DeploymentStatus,
        DeploymentInfo,
        get_training_bridge,
    )
    HAS_TRAINING_BRIDGE = True
except ImportError:
    HAS_TRAINING_BRIDGE = False

try:
    from .metrics import (
        MetricsCollector,
        Metric,
        MetricType,
        MetricValue,
        get_metrics_collector,
    )
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    from .structured_logging import (
        StructuredLogger,
        LogContext,
        get_logger,
        setup_logging,
    )
    HAS_STRUCTURED_LOGGING = True
except ImportError:
    HAS_STRUCTURED_LOGGING = False

try:
    from .monitoring import (
        MonitoringAPI,
        get_monitoring_api,
    )
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False

try:
    from .plugin_system import (
        PluginRegistry,
        PluginInfo,
        ModelPlugin,
        get_plugin_registry,
    )
    HAS_PLUGIN_SYSTEM = True
except ImportError:
    HAS_PLUGIN_SYSTEM = False

try:
    from .versioning import (
        ModelVersioning,
        VersionConfig,
        ABTestConfig,
        VersionPerformance,
        RolloutStrategy,
        get_versioning,
    )
    HAS_VERSIONING = True
except ImportError:
    HAS_VERSIONING = False

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
    # Resource Management
    "ResourceManager",
    "ResourceType",
    "ResourceQuota",
    "ResourceAllocation",
    "get_resource_manager",
    # Error Handling
    "ErrorCategory",
    "ErrorInfo",
    "ErrorClassifier",
    "RetryConfig",
    "retry_with_backoff",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "FallbackChain",
    # Health Checks
    "HealthStatus",
    "HealthCheck",
    "HealthCheckResult",
    "ModelHealthCheck",
    "ResourceHealthCheck",
    "CustomHealthCheck",
    "HealthMonitor",
    "get_health_monitor",
    # AI Service
    "AIService",
    "ServiceStatus",
    "ServiceInfo",
    "get_ai_service",
    # Model Pool
    "ModelPool",
    "PooledModel",
    "EvictionPolicy",
    "get_model_pool",
    # Async Inference
    "AsyncInferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "get_async_inference_engine",
    # Batching
    "BatchProcessor",
    "BatchConfig",
    "BatchedInferenceEngine",
    # Enhanced Inference
    "EnhancedInferenceEngine",
    "create_enhanced_engine",
    "create_enhanced_engine_by_name",
    # Integration Manager
    "IntegrationManager",
    "IntegrationStatus",
    "IntegrationInfo",
    "IntegrationComponent",
    "get_integration_manager",
    # Event Bus
    "EventBus",
    "Event",
    "get_event_bus",
    # Training-Inference Bridge
    "TrainingInferenceBridge",
    "DeploymentStatus",
    "DeploymentInfo",
    "get_training_bridge",
    # Metrics
    "MetricsCollector",
    "Metric",
    "MetricType",
    "MetricValue",
    "get_metrics_collector",
    # Structured Logging
    "StructuredLogger",
    "LogContext",
    "get_logger",
    "setup_logging",
    # Monitoring
    "MonitoringAPI",
    "get_monitoring_api",
    # Plugin System
    "PluginRegistry",
    "PluginInfo",
    "ModelPlugin",
    "get_plugin_registry",
    # Versioning
    "ModelVersioning",
    "VersionConfig",
    "ABTestConfig",
    "VersionPerformance",
    "RolloutStrategy",
    "get_versioning",
]
