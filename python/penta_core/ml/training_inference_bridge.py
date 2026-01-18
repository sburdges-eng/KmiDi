"""
Training-Inference Bridge - Seamless transition from training to inference.

Provides:
- Automatic model registration after training
- Model validation before deployment
- Hot-swapping of models
- Rollback capability
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    from .model_registry import (
        ModelRegistry,
        ModelInfo,
        ModelBackend,
        ModelTask,
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
    from .model_pool import get_model_pool
    HAS_POOL = True
except ImportError:
    HAS_POOL = False

try:
    from .health import ModelHealthCheck, HealthStatus
    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False

try:
    from .event_bus import get_event_bus, Event
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Model deployment status."""
    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentInfo:
    """Information about a model deployment."""
    model_name: str
    version: str
    status: DeploymentStatus
    deployed_at: Optional[float] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    previous_version: Optional[str] = None
    rollback_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingInferenceBridge:
    """
    Bridge between training and inference systems.

    Handles:
    - Automatic model registration after training
    - Model validation
    - Hot-swapping
    - Rollback
    """

    def __init__(self):
        self._registry: Optional[ModelRegistry] = None
        self._deployments: Dict[str, DeploymentInfo] = {}
        self._model_versions: Dict[str, List[str]] = {}  # model_name -> [versions]

        # Validation settings
        self._validate_on_deploy = True
        self._auto_rollback_on_failure = False
        self._validation_timeout_seconds = 30.0

        # Initialize dependencies
        if HAS_REGISTRY:
            try:
                self._registry = get_registry()
            except Exception:
                pass

        # Subscribe to training events
        if HAS_EVENT_BUS:
            try:
                event_bus = get_event_bus()
                event_bus.subscribe("training.completed", self._on_training_completed)
                event_bus.subscribe("training.failed", self._on_training_failed)
            except Exception:
                pass

    def register_trained_model(
        self,
        model_path: str,
        model_name: str,
        task: ModelTask,
        backend: ModelBackend,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelInfo]:
        """
        Register a trained model.

        Args:
            model_path: Path to model file
            model_name: Model name
            task: Model task
            backend: Model backend
            version: Optional version (auto-generated if None)
            metadata: Optional metadata

        Returns:
            ModelInfo if successful, None otherwise
        """
        if not self._registry:
            logger.error("Model registry not available")
            return None

        if version is None:
            import datetime
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if model file exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        # Create model info
        model_info = ModelInfo(
            name=model_name,
            task=task,
            backend=backend,
            path=model_path,
            version=version,
            description=f"Trained model registered at {time.time()}",
            tags=["trained", version],
        )

        if metadata:
            model_info.description = metadata.get("description", model_info.description)
            model_info.tags.extend(metadata.get("tags", []))

        # Register model
        self._registry.register(model_info)

        # Track version
        if model_name not in self._model_versions:
            self._model_versions[model_name] = []
        self._model_versions[model_name].append(version)

        logger.info(f"Registered trained model: {model_name} v{version}")

        # Publish event
        if HAS_EVENT_BUS:
            try:
                event_bus = get_event_bus()
                event_bus.publish(
                    "model.registered",
                    {
                        "model_name": model_name,
                        "version": version,
                        "path": model_path,
                    },
                    source="training_inference_bridge",
                )
            except Exception:
                pass

        return model_info

    def deploy_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        validate: Optional[bool] = None,
    ) -> bool:
        """
        Deploy a model for inference.

        Args:
            model_name: Model name
            version: Optional specific version (uses latest if None)
            validate: Optional validation override

        Returns:
            True if deployment successful
        """
        if not self._registry:
            logger.error("Model registry not available")
            return False

        # Get model
        if version:
            model_info = self._registry.get(model_name, version=version)
        else:
            model_info = self._registry.get(model_name)
            if not model_info and model_name in self._model_versions:
                # Get latest version from tracked versions
                versions = self._model_versions[model_name]
                if versions:
                    latest_version = versions[-1]
                    model_info = self._registry.get(model_name, version=latest_version)

        if not model_info:
            logger.error(f"Model not found: {model_name} (version: {version})")
            return False

        # Create deployment info
        deployment_key = f"{model_name}:{model_info.version}"
        deployment = DeploymentInfo(
            model_name=model_name,
            version=model_info.version,
            status=DeploymentStatus.PENDING,
            previous_version=self._get_current_version(model_name),
        )
        self._deployments[deployment_key] = deployment

        # Validate if requested
        validate = validate if validate is not None else self._validate_on_deploy
        if validate:
            deployment.status = DeploymentStatus.VALIDATING
            validation_result = self._validate_model(model_info)

            if not validation_result.get("success", False):
                deployment.status = DeploymentStatus.FAILED
                deployment.validation_results = validation_result
                logger.error(f"Model validation failed: {model_name} v{model_info.version}")

                if self._auto_rollback_on_failure and deployment.previous_version:
                    self.rollback(model_name)
                return False

            deployment.validation_results = validation_result

        # Deploy model
        try:
            # Load model into pool if available
            if HAS_POOL:
                try:
                    model_pool = get_model_pool()
                    model_pool.get(model_info, auto_load=True)
                except Exception as e:
                    logger.warning(f"Failed to load model into pool: {e}")

            deployment.status = DeploymentStatus.DEPLOYED
            deployment.deployed_at = time.time()

            logger.info(f"Model deployed: {model_name} v{model_info.version}")

            # Publish event
            if HAS_EVENT_BUS:
                try:
                    event_bus = get_event_bus()
                    event_bus.publish(
                        "model.deployed",
                        {
                            "model_name": model_name,
                            "version": model_info.version,
                            "previous_version": deployment.previous_version,
                        },
                        source="training_inference_bridge",
                    )
                except Exception:
                    pass

            return True

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"Model deployment failed: {model_name} - {e}")

            if self._auto_rollback_on_failure and deployment.previous_version:
                self.rollback(model_name)

            return False

    def _validate_model(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Validate a model before deployment."""
        validation_result = {
            "success": False,
            "checks": {},
            "errors": [],
        }

        # Check 1: Model file exists
        if not Path(model_info.path).exists():
            validation_result["errors"].append("Model file not found")
            return validation_result

        validation_result["checks"]["file_exists"] = True

        # Check 2: Model can be loaded
        if HAS_INFERENCE:
            try:
                engine = create_enhanced_engine(model_info)
                if not engine.load():
                    validation_result["errors"].append("Failed to load model")
                    return validation_result
                validation_result["checks"]["loads"] = True

                # Check 3: Model can run inference
                try:
                    import numpy as np
                    # Generate test input
                    test_input = self._generate_test_input(model_info)
                    result = engine.infer(test_input, use_fallback=False)
                    validation_result["checks"]["inference"] = True
                    validation_result["checks"]["latency_ms"] = result.latency_ms if result else 0.0
                except Exception as e:
                    validation_result["errors"].append(f"Inference test failed: {e}")
                    return validation_result
                finally:
                    # Always unload engine to prevent resource leak
                    try:
                        engine.unload()
                    except Exception as unload_error:
                        logger.warning(f"Error unloading engine during validation: {unload_error}")

            except Exception as e:
                validation_result["errors"].append(f"Model loading error: {e}")
                return validation_result

        # Check 4: Health check
        if HAS_HEALTH:
            try:
                health_check = ModelHealthCheck(
                    name=f"validation_{model_info.name}",
                    model_name=model_info.name,
                )
                health_result = health_check.check()
                validation_result["checks"]["health"] = health_result.status == HealthStatus.HEALTHY
                if health_result.status != HealthStatus.HEALTHY:
                    validation_result["errors"].append(f"Health check failed: {health_result.message}")
            except Exception as e:
                logger.warning(f"Health check error: {e}")

        validation_result["success"] = len(validation_result["errors"]) == 0
        return validation_result

    def _generate_test_input(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Generate test input for model validation."""
        import numpy as np

        # Default test input (adjust based on model task)
        if model_info.input_shape:
            shape = [1] + list(model_info.input_shape[1:])  # Add batch dimension
            return {"input": np.random.randn(*shape).astype(np.float32)}
        else:
            # Default: 128 features
            return {"input": np.random.randn(1, 128).astype(np.float32)}

    def rollback(self, model_name: str, to_version: Optional[str] = None) -> bool:
        """
        Rollback model to previous version.

        Args:
            model_name: Model name
            to_version: Optional specific version to rollback to

        Returns:
            True if rollback successful
        """
        if not self._registry:
            logger.error("Model registry not available")
            return False

        # Find deployment
        deployment = None
        for dep in self._deployments.values():
            if dep.model_name == model_name and dep.status == DeploymentStatus.DEPLOYED:
                deployment = dep
                break

        if not deployment:
            logger.error(f"No active deployment found for {model_name}")
            return False

        # Determine rollback version
        rollback_version = to_version or deployment.previous_version
        if not rollback_version:
            logger.error(f"No previous version to rollback to for {model_name}")
            return False

        # Deploy previous version
        success = self.deploy_model(model_name, version=rollback_version, validate=False)

        if success:
            deployment.rollback_version = rollback_version
            deployment.status = DeploymentStatus.ROLLED_BACK
            logger.info(f"Rolled back {model_name} to version {rollback_version}")

            # Publish event
            if HAS_EVENT_BUS:
                try:
                    event_bus = get_event_bus()
                    event_bus.publish(
                        "model.rolled_back",
                        {
                            "model_name": model_name,
                            "from_version": deployment.version,
                            "to_version": rollback_version,
                        },
                        source="training_inference_bridge",
                    )
                except Exception:
                    pass

        return success

    def _get_current_version(self, model_name: str) -> Optional[str]:
        """Get current deployed version of model."""
        for deployment in self._deployments.values():
            if deployment.model_name == model_name and deployment.status == DeploymentStatus.DEPLOYED:
                return deployment.version
        return None

    def _on_training_completed(self, event: Event):
        """Handle training completed event."""
        data = event.data
        model_name = data.get("model_name")
        model_path = data.get("model_path")
        task = data.get("task")
        backend = data.get("backend")

        if model_name and model_path:
            try:
                # Auto-register trained model
                self.register_trained_model(
                    model_path=model_path,
                    model_name=model_name,
                    task=task or ModelTask.CUSTOM,
                    backend=backend or ModelBackend.PYTORCH,
                )

                # Auto-deploy if configured
                if data.get("auto_deploy", False):
                    self.deploy_model(model_name)
            except Exception as e:
                logger.error(f"Error handling training completed: {e}")

    def _on_training_failed(self, event: Event):
        """Handle training failed event."""
        logger.warning(f"Training failed: {event.data}")

    def get_deployment_info(self, model_name: str) -> Optional[DeploymentInfo]:
        """Get deployment information for a model."""
        for deployment in self._deployments.values():
            if deployment.model_name == model_name:
                return deployment
        return None

    def list_deployments(self) -> List[DeploymentInfo]:
        """List all deployments."""
        return list(self._deployments.values())


# Singleton bridge
_training_bridge: Optional[TrainingInferenceBridge] = None
_training_bridge_lock = threading.Lock()


def get_training_bridge() -> TrainingInferenceBridge:
    """Get the global training-inference bridge singleton."""
    global _training_bridge
    if _training_bridge is None:
        with _training_bridge_lock:
            if _training_bridge is None:
                _training_bridge = TrainingInferenceBridge()
    return _training_bridge
