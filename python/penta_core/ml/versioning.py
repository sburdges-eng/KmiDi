"""
Model Versioning - A/B testing and gradual rollout support.

Provides:
- Model version tracking
- A/B testing framework
- Gradual rollout support
- Performance comparison between versions
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict

try:
    from .model_registry import ModelInfo, ModelBackend, get_registry
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

try:
    from .metrics import get_metrics_collector
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

logger = logging.getLogger(__name__)


class RolloutStrategy(Enum):
    """Rollout strategies."""
    IMMEDIATE = "immediate"  # Deploy to 100% immediately
    LINEAR = "linear"  # Linear increase over time
    EXPONENTIAL = "exponential"  # Exponential increase
    MANUAL = "manual"  # Manual control


@dataclass
class VersionConfig:
    """Configuration for a model version."""
    version: str
    weight: float = 1.0  # Traffic weight (0.0 to 1.0)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_name: str
    model_name: str
    versions: List[VersionConfig]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    strategy: RolloutStrategy = RolloutStrategy.LINEAR
    duration_hours: float = 24.0


@dataclass
class VersionPerformance:
    """Performance metrics for a version."""
    version: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    accuracy: Optional[float] = None
    last_updated: float = field(default_factory=time.time)


class ModelVersioning:
    """
    Model versioning system with A/B testing and gradual rollout.

    Supports:
    - Multiple versions of the same model
    - A/B testing
    - Gradual rollout
    - Performance tracking
    """

    def __init__(self):
        self._registry = None
        self._metrics_collector = None
        self._ab_tests: Dict[str, ABTestConfig] = {}
        self._version_performance: Dict[str, Dict[str, VersionPerformance]] = defaultdict(dict)
        self._active_versions: Dict[str, str] = {}  # model_name -> version

        # Initialize dependencies
        if HAS_REGISTRY:
            try:
                self._registry = get_registry()
            except Exception:
                pass

        if HAS_METRICS:
            try:
                self._metrics_collector = get_metrics_collector()
            except Exception:
                pass

    def register_version(
        self,
        model_name: str,
        version: str,
        model_info: ModelInfo,
    ) -> bool:
        """
        Register a model version.

        Args:
            model_name: Model name
            version: Version string
            model_info: Model information

        Returns:
            True if registration successful
        """
        if not self._registry:
            logger.error("Model registry not available")
            return False

        # Register model with version in name or metadata
        model_info.name = f"{model_name}_v{version}"
        model_info.version = version
        self._registry.register(model_info)

        # Initialize performance tracking
        if model_name not in self._version_performance:
            self._version_performance[model_name] = {}

        self._version_performance[model_name][version] = VersionPerformance(version=version)

        logger.info(f"Registered model version: {model_name} v{version}")
        return True

    def start_ab_test(
        self,
        test_name: str,
        model_name: str,
        versions: List[VersionConfig],
        strategy: RolloutStrategy = RolloutStrategy.LINEAR,
        duration_hours: float = 24.0,
    ) -> bool:
        """
        Start an A/B test.

        Args:
            test_name: Test name
            model_name: Model name
            versions: List of version configurations
            strategy: Rollout strategy
            duration_hours: Test duration in hours

        Returns:
            True if test started successfully
        """
        # Validate versions
        total_weight = sum(v.weight for v in versions)
        if abs(total_weight - 1.0) > 0.01:
            logger.error(f"Version weights must sum to 1.0, got {total_weight}")
            return False

        # Create test config
        test_config = ABTestConfig(
            test_name=test_name,
            model_name=model_name,
            versions=versions,
            strategy=strategy,
            duration_hours=duration_hours,
            end_time=time.time() + (duration_hours * 3600),
        )

        self._ab_tests[test_name] = test_config
        logger.info(f"Started A/B test: {test_name} for {model_name}")

        return True

    def stop_ab_test(self, test_name: str) -> bool:
        """Stop an A/B test."""
        if test_name not in self._ab_tests:
            logger.error(f"A/B test not found: {test_name}")
            return False

        test_config = self._ab_tests[test_name]
        test_config.end_time = time.time()

        logger.info(f"Stopped A/B test: {test_name}")
        return True

    def select_version(
        self,
        model_name: str,
        test_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Select a version for inference (based on A/B test or default).

        Args:
            model_name: Model name
            test_name: Optional specific test name

        Returns:
            Selected version or None
        """
        # Check for active A/B test
        if test_name and test_name in self._ab_tests:
            test_config = self._ab_tests[test_name]
            if test_config.model_name == model_name:
                return self._select_version_from_test(test_config)

        # Check for any active test for this model
        for test_config in self._ab_tests.values():
            if test_config.model_name == model_name:
                # Check if test is still active
                if test_config.end_time is None or time.time() < test_config.end_time:
                    return self._select_version_from_test(test_config)

        # Use default version
        return self._active_versions.get(model_name)

    def _select_version_from_test(self, test_config: ABTestConfig) -> Optional[str]:
        """Select version based on A/B test configuration."""
        import random

        # Calculate current rollout percentage based on strategy
        elapsed = time.time() - test_config.start_time
        duration = test_config.duration_hours * 3600

        if test_config.strategy == RolloutStrategy.IMMEDIATE:
            rollout_pct = 1.0
        elif test_config.strategy == RolloutStrategy.LINEAR:
            rollout_pct = min(1.0, elapsed / duration)
        elif test_config.strategy == RolloutStrategy.EXPONENTIAL:
            # Exponential: 2^(t/duration) - 1
            rollout_pct = min(1.0, (2 ** (elapsed / duration)) - 1)
        else:  # MANUAL
            rollout_pct = 1.0

        # Select version based on weights
        # Two-stage selection: first decide if included in rollout, then select among enabled versions
        if random.random() > rollout_pct:
            # Not included in rollout - return None to use default/previous version
            return None

        # Included in rollout - select among enabled versions based on weights
        rand = random.random()
        cumulative = 0.0

        for version_config in test_config.versions:
            if not version_config.enabled:
                continue

            cumulative += version_config.weight
            if rand <= cumulative:
                return version_config.version

        # Fallback to first enabled version
        for version_config in test_config.versions:
            if version_config.enabled:
                return version_config.version

        return None

    def record_inference(
        self,
        model_name: str,
        version: str,
        latency_ms: float,
        success: bool = True,
    ):
        """Record inference metrics for version."""
        if model_name not in self._version_performance:
            self._version_performance[model_name] = {}

        if version not in self._version_performance[model_name]:
            self._version_performance[model_name][version] = VersionPerformance(version=version)

        perf = self._version_performance[model_name][version]
        perf.request_count += 1
        if success:
            perf.success_count += 1
        else:
            perf.error_count += 1

        # Update latency (simplified - in practice, use proper statistics)
        if perf.request_count == 1:
            perf.average_latency_ms = latency_ms
        else:
            perf.average_latency_ms = (
                (perf.average_latency_ms * (perf.request_count - 1) + latency_ms) / perf.request_count
            )

        perf.last_updated = time.time()

    def compare_versions(
        self,
        model_name: str,
        versions: Optional[List[str]] = None,
    ) -> Dict[str, VersionPerformance]:
        """
        Compare performance of versions.

        Args:
            model_name: Model name
            versions: Optional list of versions to compare (all if None)

        Returns:
            Dictionary of version performance
        """
        if model_name not in self._version_performance:
            return {}

        all_perf = self._version_performance[model_name]

        if versions:
            return {v: all_perf[v] for v in versions if v in all_perf}
        else:
            return all_perf.copy()

    def get_best_version(
        self,
        model_name: str,
        metric: str = "average_latency_ms",
        lower_is_better: bool = True,
    ) -> Optional[str]:
        """
        Get best performing version.

        Args:
            model_name: Model name
            metric: Metric to compare
            lower_is_better: If True, lower values are better

        Returns:
            Best version or None
        """
        performances = self.compare_versions(model_name)

        if not performances:
            return None

        best_version = None
        best_value = None

        for version, perf in performances.items():
            if perf.request_count == 0:
                continue

            value = getattr(perf, metric, None)
            if value is None:
                continue

            if best_value is None:
                best_value = value
                best_version = version
            elif lower_is_better and value < best_value:
                best_value = value
                best_version = version
            elif not lower_is_better and value > best_value:
                best_value = value
                best_version = version

        return best_version

    def set_active_version(self, model_name: str, version: str):
        """Set active version for a model."""
        self._active_versions[model_name] = version
        logger.info(f"Set active version for {model_name}: {version}")

    def get_active_version(self, model_name: str) -> Optional[str]:
        """Get active version for a model."""
        return self._active_versions.get(model_name)

    def get_ab_test_status(self, test_name: Optional[str] = None) -> Dict[str, Any]:
        """Get A/B test status."""
        if test_name:
            test_config = self._ab_tests.get(test_name)
            if not test_config:
                return {"error": f"Test not found: {test_name}"}

            return {
                "test_name": test_name,
                "config": {
                    "model_name": test_config.model_name,
                    "strategy": test_config.strategy.value,
                    "start_time": test_config.start_time,
                    "end_time": test_config.end_time,
                    "duration_hours": test_config.duration_hours,
                },
                "versions": [
                    {
                        "version": v.version,
                        "weight": v.weight,
                        "enabled": v.enabled,
                    }
                    for v in test_config.versions
                ],
                "performance": self.compare_versions(test_config.model_name),
            }

        # All tests
        return {
            "tests": {
                name: {
                    "model_name": config.model_name,
                    "strategy": config.strategy.value,
                    "start_time": config.start_time,
                    "end_time": config.end_time,
                }
                for name, config in self._ab_tests.items()
            }
        }


# Singleton versioning system
_versioning: Optional[ModelVersioning] = None
_versioning_lock = threading.Lock()


def get_versioning() -> ModelVersioning:
    """Get the global versioning system singleton."""
    global _versioning
    if _versioning is None:
        with _versioning_lock:
            if _versioning is None:
                _versioning = ModelVersioning()
    return _versioning
