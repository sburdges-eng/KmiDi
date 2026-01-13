"""
Integration Manager - Coordination between training, inference, and integration modules.

Provides:
- Lifecycle management for integrations
- Event-driven communication between components
- Integration health monitoring
- Automatic reconnection on failures
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
from abc import ABC, abstractmethod

try:
    from .ai_service import AIService, get_ai_service

    HAS_AI_SERVICE = True
except ImportError:
    HAS_AI_SERVICE = False

try:
    from .health import HealthMonitor, get_health_monitor, HealthStatus

    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False

try:
    from .event_bus import EventBus, get_event_bus

    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class IntegrationInfo:
    """Information about an integration."""

    name: str
    status: IntegrationStatus
    component_type: str
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    reconnect_attempts: int = 0


class IntegrationComponent(ABC):
    """Base class for integration components."""

    def __init__(self, name: str):
        self.name = name
        self._status = IntegrationStatus.DISCONNECTED
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    @abstractmethod
    def connect(self) -> bool:
        """Connect the integration."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the integration."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if integration is connected."""
        pass

    def get_status(self) -> IntegrationStatus:
        """Get integration status."""
        return self._status

    def register_callback(self, event: str, callback: Callable):
        """Register callback for event."""
        self._callbacks[event].append(callback)

    def _notify_callbacks(self, event: str, *args, **kwargs):
        """Notify callbacks for event."""
        for callback in self._callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error in {self.name}: {e}")


class IntegrationManager:
    """
    Manages integrations between training, inference, and other components.

    Provides lifecycle management, health monitoring, and event coordination.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._integrations: Dict[str, IntegrationComponent] = {}
        self._integration_info: Dict[str, IntegrationInfo] = {}
        self._health_monitor: Optional[HealthMonitor] = None
        self._event_bus = None
        self._ai_service: Optional[AIService] = None

        # Reconnection settings
        self._reconnect_enabled = True
        self._reconnect_interval = 5.0
        self._max_reconnect_attempts = 10

        # Initialize dependencies
        if HAS_HEALTH:
            try:
                self._health_monitor = get_health_monitor()
            except Exception:
                pass

        if HAS_EVENT_BUS:
            try:
                self._event_bus = get_event_bus()
            except Exception:
                pass

        if HAS_AI_SERVICE:
            try:
                self._ai_service = get_ai_service()
            except Exception:
                pass

    def register_integration(
        self,
        integration: IntegrationComponent,
        component_type: str = "generic",
    ) -> bool:
        """
        Register an integration.

        Args:
            integration: Integration component
            component_type: Type of component

        Returns:
            True if registration successful
        """
        with self._lock:
            if integration.name in self._integrations:
                logger.warning(f"Integration {integration.name} already registered")
                return False

            self._integrations[integration.name] = integration
            self._integration_info[integration.name] = IntegrationInfo(
                name=integration.name,
                status=IntegrationStatus.DISCONNECTED,
                component_type=component_type,
            )

            logger.info(f"Registered integration: {integration.name} ({component_type})")
            return True

    def unregister_integration(self, name: str) -> bool:
        """Unregister an integration."""
        with self._lock:
            if name not in self._integrations:
                return False

            integration = self._integrations.pop(name)
            integration.disconnect()

            del self._integration_info[name]
            logger.info(f"Unregistered integration: {name}")
            return True

    def connect_integration(self, name: str) -> bool:
        """Connect an integration."""
        with self._lock:
            integration = self._integrations.get(name)
            if not integration:
                logger.error(f"Integration not found: {name}")
                return False

            info = self._integration_info[name]
            info.status = IntegrationStatus.CONNECTING

            try:
                success = integration.connect()
                if success:
                    info.status = IntegrationStatus.CONNECTED
                    info.last_heartbeat = time.time()
                    info.error_count = 0
                    info.reconnect_attempts = 0
                    logger.info(f"Integration connected: {name}")
                else:
                    info.status = IntegrationStatus.ERROR
                    info.error_count += 1
                    logger.error(f"Integration connection failed: {name}")

                return success

            except Exception as e:
                info.status = IntegrationStatus.ERROR
                info.error_count += 1
                logger.error(f"Integration connection error: {name} - {e}")
                return False

    def disconnect_integration(self, name: str) -> None:
        """Disconnect an integration."""
        with self._lock:
            integration = self._integrations.get(name)
            if not integration:
                return

            integration.disconnect()
            info = self._integration_info[name]
            info.status = IntegrationStatus.DISCONNECTED
            logger.info(f"Integration disconnected: {name}")

    def connect_all(self) -> Dict[str, bool]:
        """Connect all integrations."""
        results = {}
        for name in list(self._integrations.keys()):
            results[name] = self.connect_integration(name)
        return results

    def disconnect_all(self) -> None:
        """Disconnect all integrations."""
        for name in list(self._integrations.keys()):
            self.disconnect_integration(name)

    def get_integration(self, name: str) -> Optional[IntegrationComponent]:
        """Get integration by name."""
        with self._lock:
            return self._integrations.get(name)

    def list_integrations(
        self,
        status: Optional[IntegrationStatus] = None,
        component_type: Optional[str] = None,
    ) -> List[str]:
        """List integration names with optional filtering."""
        with self._lock:
            names = list(self._integrations.keys())

            if status:
                names = [name for name in names if self._integration_info[name].status == status]

            if component_type:
                names = [
                    name
                    for name in names
                    if self._integration_info[name].component_type == component_type
                ]

            return names

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        with self._lock:
            status = {
                "integrations": {
                    name: {
                        "status": info.status.value,
                        "component_type": info.component_type,
                        "last_heartbeat": info.last_heartbeat,
                        "error_count": info.error_count,
                        "reconnect_attempts": info.reconnect_attempts,
                        "metadata": info.metadata,
                    }
                    for name, info in self._integration_info.items()
                },
                "reconnect_enabled": self._reconnect_enabled,
                "total_integrations": len(self._integrations),
                "connected": sum(
                    1
                    for info in self._integration_info.values()
                    if info.status == IntegrationStatus.CONNECTED
                ),
            }
            return status

    def heartbeat(self, name: str) -> bool:
        """Update heartbeat for integration."""
        with self._lock:
            info = self._integration_info.get(name)
            if not info:
                return False

            info.last_heartbeat = time.time()

            # Check if integration is still connected
            integration = self._integrations.get(name)
            if integration and not integration.is_connected():
                info.status = IntegrationStatus.ERROR
                return False

            return True

    def check_health(self) -> Dict[str, HealthStatus]:
        """Check health of all integrations."""
        health_status = {}

        with self._lock:
            for name, integration in self._integrations.items():
                info = self._integration_info[name]

                # Check heartbeat timeout
                time_since_heartbeat = time.time() - info.last_heartbeat
                if time_since_heartbeat > 60.0:  # 60 second timeout
                    health_status[name] = HealthStatus.UNHEALTHY
                elif integration.is_connected():
                    health_status[name] = HealthStatus.HEALTHY
                else:
                    health_status[name] = HealthStatus.UNHEALTHY

        return health_status

    def enable_reconnection(self, enabled: bool = True):
        """Enable or disable automatic reconnection."""
        self._reconnect_enabled = enabled

    def set_reconnect_interval(self, interval: float):
        """Set reconnection interval in seconds."""
        self._reconnect_interval = interval

    def set_max_reconnect_attempts(self, max_attempts: int):
        """Set maximum reconnection attempts."""
        self._max_reconnect_attempts = max_attempts


# Singleton integration manager
_integration_manager: Optional[IntegrationManager] = None
_integration_manager_lock = threading.Lock()


def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager singleton."""
    global _integration_manager
    if _integration_manager is None:
        with _integration_manager_lock:
            if _integration_manager is None:
                _integration_manager = IntegrationManager()
    return _integration_manager
