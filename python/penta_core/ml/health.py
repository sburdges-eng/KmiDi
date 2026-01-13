"""
Health Check System - Monitor system health and model availability.

Provides:
- Model health checks (load test, inference test)
- System resource health monitoring
- Health status reporting
- Automatic recovery from unhealthy states
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

try:
    from .resource_manager import ResourceManager, get_resource_manager, ResourceType
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False

try:
    from .inference_enhanced import EnhancedInferenceEngine
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }
        if self.error:
            result["error"] = str(self.error)
        return result


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str):
        self.name = name

    def check(self) -> HealthCheckResult:
        """
        Perform health check.

        Returns:
            Health check result
        """
        raise NotImplementedError


class ModelHealthCheck(HealthCheck):
    """Health check for ML models."""

    def __init__(
        self,
        name: str,
        model_name: str,
        test_input: Optional[Dict[str, Any]] = None,
        max_latency_ms: float = 1000.0,
    ):
        """
        Initialize model health check.

        Args:
            name: Check name
            model_name: Model name to check
            test_input: Optional test input (auto-generated if None)
            max_latency_ms: Maximum acceptable latency in milliseconds
        """
        super().__init__(name)
        self.model_name = model_name
        self.test_input = test_input
        self.max_latency_ms = max_latency_ms
        self._engine: Optional[EnhancedInferenceEngine] = None

    def _get_engine(self) -> Optional[EnhancedInferenceEngine]:
        """Get or create inference engine."""
        if self._engine is None and HAS_INFERENCE:
            try:
                from .inference_enhanced import create_enhanced_engine_by_name
                self._engine = create_enhanced_engine_by_name(self.model_name)
                if self._engine:
                    self._engine.load()
            except Exception as e:
                logger.warning(f"Failed to create engine for health check: {e}")
        return self._engine

    def check(self) -> HealthCheckResult:
        """Check model health."""
        engine = self._get_engine()

        if not engine:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model {self.model_name} not available",
                details={"model_name": self.model_name},
            )

        if not engine.is_loaded():
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model {self.model_name} not loaded",
                details={"model_name": self.model_name},
            )

        # Test inference
        try:
            import numpy as np

            # Generate test input if not provided
            if self.test_input is None:
                # Default test input (adjust based on model)
                test_input = {"input": np.random.randn(1, 128).astype(np.float32)}
            else:
                test_input = self.test_input

            start_time = time.perf_counter()
            result = engine.infer(test_input, use_fallback=False)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Check latency
            if latency_ms > self.max_latency_ms:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message=f"Model {self.model_name} latency too high: {latency_ms:.1f}ms",
                    details={
                        "model_name": self.model_name,
                        "latency_ms": latency_ms,
                        "max_latency_ms": self.max_latency_ms,
                    },
                )

            # Check circuit breaker status
            cb_status = engine.get_circuit_breaker_status()
            if cb_status["state"] == "open":
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Model {self.model_name} circuit breaker is OPEN",
                    details={
                        "model_name": self.model_name,
                        "circuit_breaker": cb_status,
                    },
                )

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message=f"Model {self.model_name} is healthy",
                details={
                    "model_name": self.model_name,
                    "latency_ms": latency_ms,
                    "circuit_breaker": cb_status,
                },
            )

        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model {self.model_name} health check failed: {e}",
                details={"model_name": self.model_name},
                error=e,
            )


class ResourceHealthCheck(HealthCheck):
    """Health check for system resources."""

    def __init__(
        self,
        name: str,
        resource_type: ResourceType,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        """
        Initialize resource health check.

        Args:
            name: Check name
            resource_type: Resource type to check
            warning_threshold: Warning threshold (0-1)
            critical_threshold: Critical threshold (0-1)
        """
        super().__init__(name)
        self.resource_type = resource_type
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def check(self) -> HealthCheckResult:
        """Check resource health."""
        if not HAS_RESOURCE_MANAGER:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Resource manager not available",
            )

        try:
            resource_manager = get_resource_manager()
            usage_ratio = resource_manager.get_usage_ratio(self.resource_type)
            available = resource_manager.get_available(self.resource_type)
            usage = resource_manager.get_usage(self.resource_type)

            if usage_ratio >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = (
                    f"Resource {self.resource_type.value} critical: "
                    f"{usage_ratio:.1%} used"
                )
            elif usage_ratio >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = (
                    f"Resource {self.resource_type.value} high: "
                    f"{usage_ratio:.1%} used"
                )
            else:
                status = HealthStatus.HEALTHY
                message = (
                    f"Resource {self.resource_type.value} healthy: "
                    f"{usage_ratio:.1%} used"
                )

            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "resource_type": self.resource_type.value,
                    "usage_ratio": usage_ratio,
                    "usage": usage,
                    "available": available,
                },
            )

        except Exception as e:
            logger.error(f"Resource health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Resource health check failed: {e}",
                error=e,
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check with user-defined function."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ):
        """
        Initialize custom health check.

        Args:
            name: Check name
            check_func: Function that returns HealthCheckResult
        """
        super().__init__(name)
        self.check_func = check_func

    def check(self) -> HealthCheckResult:
        """Run custom health check."""
        try:
            return self.check_func()
        except Exception as e:
            logger.error(f"Custom health check {self.name} failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Custom health check failed: {e}",
                error=e,
            )


class HealthMonitor:
    """
    Health monitor that runs multiple health checks and aggregates results.
    """

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._check_history: Dict[str, deque] = {}
        self._history_size = 100

    def register_check(self, check: HealthCheck):
        """Register a health check."""
        self._checks[check.name] = check
        self._check_history[check.name] = deque(maxlen=self._history_size)
        logger.info(f"Registered health check: {check.name}")

    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            del self._check_history[name]
            logger.info(f"Unregistered health check: {name}")

    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        check = self._checks.get(name)
        if not check:
            logger.warning(f"Health check not found: {name}")
            return None

        result = check.check()
        self._check_history[name].append(result)
        return result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name, check in self._checks.items():
            result = check.check()
            results[name] = result
            self._check_history[name].append(result)
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status from all checks."""
        results = self.run_all_checks()

        if not results:
            return HealthStatus.UNKNOWN

        # Determine overall status
        statuses = [r.status for r in results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        results = self.run_all_checks()
        overall_status = self.get_overall_status()

        # Calculate statistics
        check_stats = {}
        for name, result in results.items():
            history = list(self._check_history.get(name, []))
            if history:
                recent_results = history[-10:]  # Last 10 checks
                healthy_count = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
                check_stats[name] = {
                    "current_status": result.status.value,
                    "recent_health_rate": healthy_count / len(recent_results),
                    "last_check": result.timestamp,
                }

        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": {
                name: result.to_dict() for name, result in results.items()
            },
            "statistics": check_stats,
        }

    def get_check_history(self, name: str, limit: int = 10) -> List[HealthCheckResult]:
        """Get history for a specific check."""
        history = list(self._check_history.get(name, []))
        return history[-limit:]


# Singleton health monitor
_health_monitor: Optional[HealthMonitor] = None
_health_monitor_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor singleton."""
    global _health_monitor
    if _health_monitor is None:
        with _health_monitor_lock:
            if _health_monitor is None:
                _health_monitor = HealthMonitor()
    return _health_monitor
