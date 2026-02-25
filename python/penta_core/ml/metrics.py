"""
Metrics Collection - Comprehensive metrics collection system.

Provides:
- Inference latency metrics
- Model accuracy metrics
- Resource usage metrics
- Error rate tracking
- Export to Prometheus/StatsD format
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A metric value with timestamp."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Metric:
    """A metric with history and aggregation."""

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        max_history: int = 1000,
    ):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.max_history = max_history

        self._values: deque = deque(maxlen=max_history)
        self._values_by_labels: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=max_history))

    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        labels = labels or {}
        label_key = tuple(sorted(labels.items()))

        metric_value = MetricValue(value=value, labels=labels)
        self._values.append(metric_value)
        self._values_by_labels[label_key].append(metric_value)

    def get_value(self, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current metric value."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key)
            if values:
                return values[-1].value
        elif self._values:
            return self._values[-1].value
        return None

    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get sum of metric values."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key)
            if values:
                return sum(v.value for v in values)
        return sum(v.value for v in self._values)

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get count of metric values."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key)
            if values:
                return len(values)
        return len(self._values)

    def get_average(self, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get average of metric values."""
        count = self.get_count(labels)
        if count == 0:
            return None
        return self.get_sum(labels) / count

    def get_min(self, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get minimum value."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key)
            if values:
                return min(v.value for v in values)
        elif self._values:
            return min(v.value for v in self._values)
        return None

    def get_max(self, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get maximum value."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key)
            if values:
                return max(v.value for v in values)
        elif self._values:
            return max(v.value for v in self._values)
        return None

    def get_percentile(
        self,
        percentile: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get percentile value."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key)
        else:
            values = self._values

        if not values:
            return None

        sorted_values = sorted(v.value for v in values)
        # Proper percentile calculation using interpolation
        # For percentile p, use (p/100) * (n-1) to get the position
        # This handles edge cases correctly (0th percentile = min, 100th percentile = max)
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (percentile / 100.0) * (len(sorted_values) - 1)
        index = int(position)
        # Handle interpolation for non-integer positions
        if index < len(sorted_values) - 1:
            fraction = position - index
            return sorted_values[index] + fraction * (
                sorted_values[index + 1] - sorted_values[index]
            )
        return sorted_values[index]

    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        count = self.get_count(labels)
        if count == 0:
            return {
                "count": 0,
                "sum": 0.0,
                "average": None,
                "min": None,
                "max": None,
            }

        return {
            "count": count,
            "sum": self.get_sum(labels),
            "average": self.get_average(labels),
            "min": self.get_min(labels),
            "max": self.get_max(labels),
            "p50": self.get_percentile(50, labels),
            "p95": self.get_percentile(95, labels),
            "p99": self.get_percentile(99, labels),
        }

    def get_stats_since(
        self,
        cutoff_time: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Get statistics for values recorded since cutoff_time."""
        if labels:
            label_key = tuple(sorted(labels.items()))
            values = self._values_by_labels.get(label_key, deque())
        else:
            values = self._values

        filtered = [v.value for v in values if v.timestamp >= cutoff_time]
        if not filtered:
            return {
                "count": 0,
                "sum": 0.0,
                "average": None,
                "min": None,
                "max": None,
            }

        filtered.sort()
        count = len(filtered)
        total = sum(filtered)
        avg = total / count

        def percentile(p: float) -> float:
            if count == 1:
                return filtered[0]
            position = (p / 100.0) * (count - 1)
            index = int(position)
            if index < count - 1:
                fraction = position - index
                return filtered[index] + fraction * (filtered[index + 1] - filtered[index])
            return filtered[index]

        return {
            "count": count,
            "sum": total,
            "average": avg,
            "min": filtered[0],
            "max": filtered[-1],
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
        }


class MetricsCollector:
    """
    Metrics collector for ML system.

    Tracks:
    - Inference latency
    - Model accuracy
    - Resource usage
    - Error rates
    """

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}

        # Initialize standard metrics
        self._initialize_standard_metrics()

    def _initialize_standard_metrics(self):
        """Initialize standard metrics."""
        self.register_metric(
            "inference_latency_ms",
            MetricType.HISTOGRAM,
            "Inference latency in milliseconds",
        )
        self.register_metric(
            "inference_requests_total",
            MetricType.COUNTER,
            "Total inference requests",
        )
        self.register_metric(
            "inference_errors_total",
            MetricType.COUNTER,
            "Total inference errors",
        )
        self.register_metric(
            "model_accuracy",
            MetricType.GAUGE,
            "Model accuracy",
        )
        self.register_metric(
            "gpu_memory_usage_mb",
            MetricType.GAUGE,
            "GPU memory usage in MB",
        )
        self.register_metric(
            "cpu_memory_usage_mb",
            MetricType.GAUGE,
            "CPU memory usage in MB",
        )

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
    ) -> Metric:
        """Register a new metric."""
        metric = Metric(name, metric_type, description)
        self._metrics[name] = metric
        logger.debug(f"Registered metric: {name}")
        return metric

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name."""
        return self._metrics.get(name)

    def record_inference(
        self,
        model_name: str,
        latency_ms: float,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record inference metric."""
        labels = labels or {}
        labels["model"] = model_name

        # Record latency
        latency_metric = self.get_metric("inference_latency_ms")
        if latency_metric:
            latency_metric.record(latency_ms, labels)

        # Record request count
        requests_metric = self.get_metric("inference_requests_total")
        if requests_metric:
            requests_metric.record(1, labels)

        # Record error if failed
        if not success:
            errors_metric = self.get_metric("inference_errors_total")
            if errors_metric:
                errors_metric.record(1, labels)

    def record_accuracy(
        self,
        model_name: str,
        accuracy: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record model accuracy."""
        labels = labels or {}
        labels["model"] = model_name

        accuracy_metric = self.get_metric("model_accuracy")
        if accuracy_metric:
            accuracy_metric.record(accuracy, labels)

    def record_resource_usage(
        self,
        resource_type: str,
        usage_mb: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record resource usage."""
        labels = labels or {}
        labels["resource"] = resource_type

        metric_name = f"{resource_type}_memory_usage_mb"
        metric = self.get_metric(metric_name)
        if not metric:
            metric = self.register_metric(
                metric_name,
                MetricType.GAUGE,
                f"{resource_type} memory usage in MB",
            )
        metric.record(usage_mb, labels)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        return self._metrics.copy()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# HELP penta_core_ml_metrics ML system metrics")
        lines.append("# TYPE penta_core_ml_metrics counter")

        for name, metric in self._metrics.items():
            # Export metric definition
            lines.append(f"# TYPE {name} {metric.metric_type.value}")
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")

            # Export values
            stats = metric.get_stats()
            if stats["count"] > 0:
                # Export summary
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
                if stats["average"] is not None:
                    lines.append(f"{name}_avg {stats['average']}")
                if stats["min"] is not None:
                    lines.append(f"{name}_min {stats['min']}")
                if stats["max"] is not None:
                    lines.append(f"{name}_max {stats['max']}")

        return "\n".join(lines)

    def export_statsd(self) -> List[str]:
        """Export metrics in StatsD format."""
        lines = []
        for name, metric in self._metrics.items():
            stats = metric.get_stats()
            if stats["count"] > 0:
                # Export as StatsD format
                if stats["average"] is not None:
                    lines.append(f"{name}:{stats['average']}|g")
                if stats["count"] > 0:
                    lines.append(f"{name}_count:{stats['count']}|c")

        return lines

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}
        for name, metric in self._metrics.items():
            summary[name] = metric.get_stats()
        return summary

    def clear(self, metric_name: Optional[str] = None):
        """Clear metrics."""
        if metric_name:
            metric = self._metrics.get(metric_name)
            if metric:
                metric._values.clear()
                metric._values_by_labels.clear()
        else:
            for metric in self._metrics.values():
                metric._values.clear()
                metric._values_by_labels.clear()


# Singleton metrics collector
_metrics_collector: Optional[MetricsCollector] = None
_metrics_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector
