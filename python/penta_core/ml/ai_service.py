"""
Unified AI Service - Single entry point for all AI operations.

Provides:
- Unified API for training, inference, and model management
- Service discovery and registration
- Plugin architecture for adding new models/tasks
- Dependency injection for testability
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable
from enum import Enum
import threading

try:
    from .model_registry import (
        ModelRegistry,
        ModelInfo,
        ModelTask,
        ModelBackend,
        get_registry,
        TrainingJob,
        TrainingStatus,
    )
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

try:
    from .training_orchestrator import TrainingOrchestrator, TrainingConfig
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

try:
    from .inference_enhanced import EnhancedInferenceEngine, create_enhanced_engine
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

try:
    from .health import HealthMonitor, get_health_monitor, HealthStatus
    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False

try:
    from .resource_manager import ResourceManager, get_resource_manager
    HAS_RESOURCES = True
except ImportError:
    HAS_RESOURCES = False

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ServiceInfo:
    """Information about a service."""
    name: str
    version: str
    status: ServiceStatus
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIServiceComponent(ABC):
    """Base class for AI service components."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the component."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the component."""
        pass

    @abstractmethod
    def get_status(self) -> ServiceStatus:
        """Get component status."""
        pass


class ModelService(AIServiceComponent):
    """Model management service."""

    def __init__(self):
        self._registry: Optional[ModelRegistry] = None
        self._status = ServiceStatus.INITIALIZING

    def initialize(self) -> bool:
        """Initialize model service."""
        if not HAS_REGISTRY:
            logger.error("Model registry not available")
            return False

        try:
            self._registry = get_registry()
            self._registry.discover()
            self._status = ServiceStatus.READY
            logger.info("Model service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            self._status = ServiceStatus.ERROR
            return False

    def shutdown(self) -> None:
        """Shutdown model service."""
        self._status = ServiceStatus.SHUTDOWN
        logger.info("Model service shutdown")

    def get_status(self) -> ServiceStatus:
        """Get service status."""
        return self._status

    def list_models(self, task: Optional[ModelTask] = None) -> List[ModelInfo]:
        """List available models."""
        if not self._registry:
            return []
        return self._registry.list(task)

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        if not self._registry:
            return None
        return self._registry.get(name)

    def register_model(self, model: ModelInfo) -> None:
        """Register a model."""
        if self._registry:
            self._registry.register(model)


class InferenceService(AIServiceComponent):
    """Inference service."""

    def __init__(self):
        self._engines: Dict[str, EnhancedInferenceEngine] = {}
        self._status = ServiceStatus.INITIALIZING

    def initialize(self) -> bool:
        """Initialize inference service."""
        if not HAS_INFERENCE:
            logger.error("Inference engine not available")
            return False

        self._status = ServiceStatus.READY
        logger.info("Inference service initialized")
        return True

    def shutdown(self) -> None:
        """Shutdown inference service."""
        for engine in self._engines.values():
            try:
                engine.unload()
            except Exception as e:
                logger.warning(f"Error unloading engine: {e}")

        self._engines.clear()
        self._status = ServiceStatus.SHUTDOWN
        logger.info("Inference service shutdown")

    def get_status(self) -> ServiceStatus:
        """Get service status."""
        return self._status

    def get_engine(self, model_name: str, auto_load: bool = True) -> Optional[EnhancedInferenceEngine]:
        """Get inference engine for a model."""
        if model_name in self._engines:
            return self._engines[model_name]

        if not auto_load:
            return None

        # Try to create and load engine
        try:
            from .inference_enhanced import create_enhanced_engine_by_name
            engine = create_enhanced_engine_by_name(model_name)
            if engine and engine.load():
                self._engines[model_name] = engine
                return engine
        except Exception as e:
            logger.error(f"Failed to create engine for {model_name}: {e}")

        return None

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        use_fallback: bool = True,
    ) -> Optional[Any]:
        """Run inference."""
        engine = self.get_engine(model_name)
        if not engine:
            logger.error(f"Engine not available for {model_name}")
            return None

        try:
            self._status = ServiceStatus.BUSY
            result = engine.infer(inputs, use_fallback=use_fallback)
            self._status = ServiceStatus.READY
            return result
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            self._status = ServiceStatus.ERROR
            return None
        finally:
            if self._status == ServiceStatus.ERROR:
                # Try to recover
                self._status = ServiceStatus.READY


class TrainingService(AIServiceComponent):
    """Training service."""

    def __init__(self):
        self._orchestrator: Optional[TrainingOrchestrator] = None
        self._status = ServiceStatus.INITIALIZING

    def initialize(self) -> bool:
        """Initialize training service."""
        if not HAS_TRAINING:
            logger.error("Training orchestrator not available")
            return False

        try:
            self._orchestrator = TrainingOrchestrator()
            self._status = ServiceStatus.READY
            logger.info("Training service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize training service: {e}")
            self._status = ServiceStatus.ERROR
            return False

    def shutdown(self) -> None:
        """Shutdown training service."""
        if self._orchestrator:
            self._orchestrator.clear_queue()
        self._status = ServiceStatus.SHUTDOWN
        logger.info("Training service shutdown")

    def get_status(self) -> ServiceStatus:
        """Get service status."""
        return self._status

    def queue_training(
        self,
        model_name: str,
        task: Optional[ModelTask] = None,
        **kwargs
    ) -> Optional[TrainingJob]:
        """Queue a model for training."""
        if not self._orchestrator:
            return None

        try:
            return self._orchestrator.queue_model(model_name, task=task, **kwargs)
        except Exception as e:
            logger.error(f"Failed to queue training: {e}")
            return None

    def run_training(self, parallel: bool = False) -> Dict[str, Any]:
        """Run all queued training jobs."""
        if not self._orchestrator:
            return {}

        try:
            self._status = ServiceStatus.BUSY
            results = self._orchestrator.run_all(parallel=parallel)
            self._status = ServiceStatus.READY
            return results
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._status = ServiceStatus.ERROR
            return {}


class AIService:
    """
    Unified AI Service - Single entry point for all AI operations.

    Coordinates:
    - Model management
    - Inference
    - Training
    - Health monitoring
    - Resource management
    """

    _instance: Optional[AIService] = None
    _lock = threading.Lock()

    def __new__(cls) -> AIService:
        """Singleton pattern with thread-safe double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._constructed = False
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Use _constructed flag to prevent re-running __init__ on singleton access
        # _initialized flag is for tracking whether initialize() was called
        if hasattr(self, '_constructed') and self._constructed:
            return

        self._model_service = ModelService()
        self._inference_service = InferenceService()
        self._training_service = TrainingService()
        self._health_monitor: Optional[HealthMonitor] = None
        self._resource_manager: Optional[ResourceManager] = None

        self._components: List[AIServiceComponent] = [
            self._model_service,
            self._inference_service,
            self._training_service,
        ]

        # Mark as constructed (but not yet fully initialized)
        self._constructed = True
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all services."""
        if self._initialized:
            return True

        logger.info("Initializing AI Service...")

        # Initialize resource manager
        if HAS_RESOURCES:
            self._resource_manager = get_resource_manager()

        # Initialize health monitor
        if HAS_HEALTH:
            self._health_monitor = get_health_monitor()

        # Initialize all components
        success = True
        for component in self._components:
            if not component.initialize():
                success = False
                logger.error(f"Failed to initialize component: {component.__class__.__name__}")

        if success:
            self._initialized = True
            logger.info("AI Service initialized successfully")
        else:
            logger.error("AI Service initialization failed")

        return success

    def shutdown(self) -> None:
        """Shutdown all services."""
        logger.info("Shutting down AI Service...")

        for component in reversed(self._components):
            try:
                component.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down component: {e}")

        self._initialized = False
        logger.info("AI Service shutdown complete")

    # =========================================================================
    # Model Management
    # =========================================================================

    def list_models(self, task: Optional[ModelTask] = None) -> List[ModelInfo]:
        """List available models."""
        return self._model_service.list_models(task)

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        return self._model_service.get_model(name)

    def register_model(self, model: ModelInfo) -> None:
        """Register a model."""
        self._model_service.register_model(model)

    # =========================================================================
    # Inference
    # =========================================================================

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        use_fallback: bool = True,
    ) -> Optional[Any]:
        """Run inference."""
        return self._inference_service.infer(model_name, inputs, use_fallback)

    def get_inference_engine(self, model_name: str) -> Optional[EnhancedInferenceEngine]:
        """Get inference engine for a model."""
        return self._inference_service.get_engine(model_name)

    # =========================================================================
    # Training
    # =========================================================================

    def queue_training(
        self,
        model_name: str,
        task: Optional[ModelTask] = None,
        **kwargs
    ) -> Optional[TrainingJob]:
        """Queue a model for training."""
        return self._training_service.queue_training(model_name, task=task, **kwargs)

    def run_training(self, parallel: bool = False) -> Dict[str, Any]:
        """Run all queued training jobs."""
        return self._training_service.run_training(parallel=parallel)

    # =========================================================================
    # Health and Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        status = {
            "initialized": self._initialized,
            "components": {
                "model_service": self._model_service.get_status().value,
                "inference_service": self._inference_service.get_status().value,
                "training_service": self._training_service.get_status().value,
            },
        }

        if self._health_monitor:
            status["health"] = self._health_monitor.get_status_report()

        if self._resource_manager:
            status["resources"] = self._resource_manager.get_status()

        return status

    def get_health_status(self) -> HealthStatus:
        """Get overall health status."""
        if self._health_monitor:
            return self._health_monitor.get_overall_status()
        return HealthStatus.UNKNOWN

    # =========================================================================
    # Resource Management
    # =========================================================================

    def get_resource_manager(self) -> Optional[ResourceManager]:
        """Get resource manager."""
        return self._resource_manager

    def get_health_monitor(self) -> Optional[HealthMonitor]:
        """Get health monitor."""
        return self._health_monitor


# Singleton access
def get_ai_service() -> AIService:
    """Get the AI service singleton."""
    service = AIService()
    if not service._initialized:
        service.initialize()
    return service
