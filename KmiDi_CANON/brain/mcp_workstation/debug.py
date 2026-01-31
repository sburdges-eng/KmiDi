"""Debug / logging helpers for orchestrator with error tracking and performance monitoring."""

import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class DebugCategory(Enum):
    """Categories for debug events."""
    AI_COMMUNICATION = "ai_communication"
    PERFORMANCE = "performance"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DebugEvent:
    """A single debug event with automatic timestamp."""
    message: str
    category: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


@dataclass
class PerformanceMetric:
    """Performance timing metric."""
    operation: str
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Optional[Dict[str, Any]] = None


class _Debug:
    """
    Debug and performance monitoring system for the orchestrator.

    Features:
    - Ring buffer for events (configurable max size)
    - Error tracking with stack traces
    - Performance metrics collection
    - Context manager for timing operations
    """

    def __init__(self, max_events: int = 1000, max_metrics: int = 500) -> None:
        """
        Initialize debug system.

        Args:
            max_events: Maximum events to keep in ring buffer
            max_metrics: Maximum performance metrics to keep
        """
        # Ring buffers - automatically discard oldest when full
        self.events: deque = deque(maxlen=max_events)
        self.metrics: deque = deque(maxlen=max_metrics)

    def info(self, category: DebugCategory, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an informational event.

        Args:
            category: Event category
            message: Event message
            details: Optional additional details
        """
        event = DebugEvent(
            message=message,
            category=category.value,
            details=details
        )
        self.events.append(event)

    def error(self, message: str, exception: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error event with optional exception and stack trace.

        Args:
            message: Error message
            exception: Optional exception object
            details: Optional additional details
        """
        stack_trace = None
        if exception:
            stack_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        event = DebugEvent(
            message=message,
            category=DebugCategory.ERROR.value,
            details=details or {},
            stack_trace=stack_trace
        )
        self.events.append(event)

    def warning(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning event.

        Args:
            message: Warning message
            details: Optional additional details
        """
        event = DebugEvent(
            message=message,
            category=DebugCategory.WARNING.value,
            details=details
        )
        self.events.append(event)

    def get_errors(self, limit: int = 25) -> List[DebugEvent]:
        """
        Get recent error events.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error events, most recent first
        """
        errors = [
            event for event in self.events
            if event.category == DebugCategory.ERROR.value
        ]
        # Return most recent first
        return list(reversed(errors))[:limit]

    def get_warnings(self, limit: int = 25) -> List[DebugEvent]:
        """
        Get recent warning events.

        Args:
            limit: Maximum number of warnings to return

        Returns:
            List of warning events, most recent first
        """
        warnings = [
            event for event in self.events
            if event.category == DebugCategory.WARNING.value
        ]
        return list(reversed(warnings))[:limit]

    def get_events_by_category(self, category: DebugCategory, limit: int = 50) -> List[DebugEvent]:
        """
        Get events by category.

        Args:
            category: Category to filter by
            limit: Maximum events to return

        Returns:
            List of events, most recent first
        """
        filtered = [
            event for event in self.events
            if event.category == category.value
        ]
        return list(reversed(filtered))[:limit]

    def record_performance(self, operation: str, duration_ms: float, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a performance metric.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            context: Optional context information
        """
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            context=context
        )
        self.metrics.append(metric)

    @contextmanager
    def measure(self, operation: str, context: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        """
        Context manager for measuring operation duration.

        Usage:
            with debug.measure("llm_parse"):
                result = llm.parse(text)

        Args:
            operation: Name of the operation to measure
            context: Optional context information
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.record_performance(operation, duration_ms, context)

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report with statistics.

        Returns:
            Dict with performance statistics per operation
        """
        if not self.metrics:
            return {
                "total_operations": 0,
                "operations": {},
                "recent_metrics": []
            }

        # Group metrics by operation
        operations: Dict[str, List[float]] = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration_ms)

        # Calculate statistics per operation
        stats = {}
        for operation, durations in operations.items():
            sorted_durations = sorted(durations)
            count = len(durations)

            stats[operation] = {
                "count": count,
                "mean_ms": sum(durations) / count,
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": sorted_durations[count // 2],
                "p95_ms": sorted_durations[int(count * 0.95)] if count > 1 else sorted_durations[0],
                "p99_ms": sorted_durations[int(count * 0.99)] if count > 1 else sorted_durations[0],
            }

        # Get recent metrics (last 10)
        recent = list(reversed(list(self.metrics)))[:10]
        recent_metrics = [
            {
                "operation": m.operation,
                "duration_ms": round(m.duration_ms, 2),
                "timestamp": m.timestamp,
                "context": m.context
            }
            for m in recent
        ]

        return {
            "total_operations": sum(len(durations) for durations in operations.values()),
            "operations": stats,
            "recent_metrics": recent_metrics
        }

    def clear(self) -> None:
        """Clear all events and metrics."""
        self.events.clear()
        self.metrics.clear()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall debug system summary.

        Returns:
            Dict with counts and recent activity
        """
        error_count = sum(1 for e in self.events if e.category == DebugCategory.ERROR.value)
        warning_count = sum(1 for e in self.events if e.category == DebugCategory.WARNING.value)

        return {
            "total_events": len(self.events),
            "total_metrics": len(self.metrics),
            "error_count": error_count,
            "warning_count": warning_count,
            "recent_errors": [
                {"message": e.message, "timestamp": e.timestamp}
                for e in self.get_errors(limit=5)
            ],
            "performance_summary": self.get_performance_report()
        }


# Singleton instance
_debug_instance: Optional[_Debug] = None


def get_debug() -> _Debug:
    """
    Get the global debug instance (singleton pattern).

    Returns:
        The debug instance
    """
    global _debug_instance
    if _debug_instance is None:
        _debug_instance = _Debug()
    return _debug_instance


# Convenience functions for common operations
def log_error(message: str, exception: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to log an error."""
    get_debug().error(message, exception, details)


def log_warning(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to log a warning."""
    get_debug().warning(message, details)


def log_info(category: DebugCategory, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to log info."""
    get_debug().info(category, message, details)


def measure_performance(operation: str, context: Optional[Dict[str, Any]] = None):
    """Convenience function to get performance measurement context manager."""
    return get_debug().measure(operation, context)
