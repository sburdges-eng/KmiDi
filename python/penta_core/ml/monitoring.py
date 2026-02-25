"""
Monitoring API - REST endpoints for metrics, status, and historical data.

Provides:
- REST API for metrics
- Real-time status endpoints
- Historical data aggregation
- Alert generation
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

try:
    from .metrics import get_metrics_collector, MetricsCollector
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    from .health import get_health_monitor, HealthMonitor, HealthStatus
    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False

try:
    from .ai_service import get_ai_service, AIService
    HAS_AI_SERVICE = True
except ImportError:
    HAS_AI_SERVICE = False

try:
    from .resource_manager import get_resource_manager, ResourceManager
    HAS_RESOURCES = True
except ImportError:
    HAS_RESOURCES = False

logger = logging.getLogger(__name__)


class MonitoringAPI:
    """
    Monitoring API for exposing system metrics and status.

    Provides REST-like interface for monitoring data.
    """

    def __init__(self):
        self._metrics_collector: Optional[MetricsCollector] = None
        self._health_monitor: Optional[HealthMonitor] = None
        self._ai_service: Optional[AIService] = None
        self._resource_manager: Optional[ResourceManager] = None

        # Initialize dependencies
        if HAS_METRICS:
            try:
                self._metrics_collector = get_metrics_collector()
            except Exception:
                pass

        if HAS_HEALTH:
            try:
                self._health_monitor = get_health_monitor()
            except Exception:
                pass

        if HAS_AI_SERVICE:
            try:
                self._ai_service = get_ai_service()
            except Exception:
                pass

        if HAS_RESOURCES:
            try:
                self._resource_manager = get_resource_manager()
            except Exception:
                pass

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Get metrics data.

        Args:
            metric_name: Optional specific metric name
            format: Output format (json, prometheus, statsd)

        Returns:
            Metrics data
        """
        if not self._metrics_collector:
            return {"error": "Metrics collector not available"}

        if metric_name:
            metric = self._metrics_collector.get_metric(metric_name)
            if not metric:
                return {"error": f"Metric not found: {metric_name}"}

            if format == "prometheus":
                return {"data": self._metrics_collector.export_prometheus()}
            elif format == "statsd":
                return {"data": self._metrics_collector.export_statsd()}
            else:
                return {
                    "metric": metric_name,
                    "stats": metric.get_stats(),
                }

        # All metrics
        if format == "prometheus":
            return {"data": self._metrics_collector.export_prometheus()}
        elif format == "statsd":
            return {"data": self._metrics_collector.export_statsd()}
        else:
            return {
                "metrics": self._metrics_collector.get_summary(),
                "timestamp": time.time(),
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        if not self._health_monitor:
            return {"error": "Health monitor not available"}

        status_report = self._health_monitor.get_status_report()
        return {
            "status": status_report,
            "timestamp": time.time(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": time.time(),
            "components": {},
        }

        # AI Service status
        if self._ai_service:
            try:
                status["components"]["ai_service"] = self._ai_service.get_status()
            except Exception as e:
                status["components"]["ai_service"] = {"error": str(e)}

        # Resource status
        if self._resource_manager:
            try:
                status["components"]["resources"] = self._resource_manager.get_status()
            except Exception as e:
                status["components"]["resources"] = {"error": str(e)}

        # Health status
        if self._health_monitor:
            try:
                overall_status = self._health_monitor.get_overall_status()
                status["components"]["health"] = {
                    "overall_status": overall_status.value,
                }
            except Exception as e:
                status["components"]["health"] = {"error": str(e)}

        return status

    def get_metrics_summary(
        self,
        time_range_minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Get metrics summary for time range.

        Args:
            time_range_minutes: Time range in minutes

        Returns:
            Metrics summary
        """
        if not self._metrics_collector:
            return {"error": "Metrics collector not available"}

        # Get all metrics
        all_metrics = self._metrics_collector.get_all_metrics()
        summary = {}

        cutoff_time = time.time() - (time_range_minutes * 60)

        for name, metric in all_metrics.items():
            stats = metric.get_stats_since(cutoff_time)
            summary[name] = stats

        return {
            "summary": summary,
            "time_range_minutes": time_range_minutes,
            "timestamp": time.time(),
        }

    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get current alerts.

        Returns:
            List of alerts
        """
        alerts = []

        # Check health status
        if self._health_monitor:
            try:
                status_report = self._health_monitor.get_status_report()
                overall_status = status_report.get("overall_status")

                if overall_status == "unhealthy":
                    alerts.append({
                        "severity": "critical",
                        "message": "System health is unhealthy",
                        "component": "health_monitor",
                        "timestamp": time.time(),
                    })

                # Check individual checks
                for check_name, check_data in status_report.get("checks", {}).items():
                    if check_data.get("current_status") == "unhealthy":
                        alerts.append({
                            "severity": "warning",
                            "message": f"Health check failed: {check_name}",
                            "component": "health_monitor",
                            "check": check_name,
                            "timestamp": time.time(),
                        })
            except Exception as e:
                logger.error(f"Error checking health alerts: {e}")

        # Check resource usage
        if self._resource_manager:
            try:
                resource_status = self._resource_manager.get_status()
                for resource_type, usage_data in resource_status.get("usage", {}).items():
                    ratio = usage_data.get("ratio", 0.0)
                    if ratio > 0.95:
                        alerts.append({
                            "severity": "critical",
                            "message": f"Resource {resource_type} usage critical: {ratio:.1%}",
                            "component": "resource_manager",
                            "resource": resource_type,
                            "usage_ratio": ratio,
                            "timestamp": time.time(),
                        })
                    elif ratio > 0.80:
                        alerts.append({
                            "severity": "warning",
                            "message": f"Resource {resource_type} usage high: {ratio:.1%}",
                            "component": "resource_manager",
                            "resource": resource_type,
                            "usage_ratio": ratio,
                            "timestamp": time.time(),
                        })
            except Exception as e:
                logger.error(f"Error checking resource alerts: {e}")

        # Check error rates
        if self._metrics_collector:
            try:
                errors_metric = self._metrics_collector.get_metric("inference_errors_total")
                requests_metric = self._metrics_collector.get_metric("inference_requests_total")

                if errors_metric and requests_metric:
                    error_count = errors_metric.get_count()
                    request_count = requests_metric.get_count()

                    if request_count > 0:
                        error_rate = error_count / request_count
                        if error_rate > 0.1:  # 10% error rate
                            alerts.append({
                                "severity": "critical",
                                "message": f"High error rate: {error_rate:.1%}",
                                "component": "metrics",
                                "error_rate": error_rate,
                                "timestamp": time.time(),
                            })
                        elif error_rate > 0.05:  # 5% error rate
                            alerts.append({
                                "severity": "warning",
                                "message": f"Elevated error rate: {error_rate:.1%}",
                                "component": "metrics",
                                "error_rate": error_rate,
                                "timestamp": time.time(),
                            })
            except Exception as e:
                logger.error(f"Error checking error rate alerts: {e}")

        return alerts

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "timestamp": time.time(),
            "metrics": self.get_metrics(),
            "health": self.get_health_status(),
            "system_status": self.get_system_status(),
            "alerts": self.get_alerts(),
        }


# Singleton monitoring API
_monitoring_api: Optional[MonitoringAPI] = None
_monitoring_api_lock = threading.Lock()


def get_monitoring_api() -> MonitoringAPI:
    """Get the global monitoring API singleton."""
    global _monitoring_api
    if _monitoring_api is None:
        with _monitoring_api_lock:
            if _monitoring_api is None:
                _monitoring_api = MonitoringAPI()
    return _monitoring_api
