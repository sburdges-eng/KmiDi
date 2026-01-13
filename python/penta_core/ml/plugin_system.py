"""
Plugin System - Dynamic model registration and discovery.

Provides:
- Plugin registration and discovery
- Interface contracts for model plugins
- Dynamic loading of model plugins
- Version management for plugins
"""

from __future__ import annotations

import logging
import threading
import importlib
import importlib.util
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    plugin_type: str
    module_path: str
    class_name: str
    description: str = ""
    author: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelPlugin(ABC):
    """Base class for model plugins."""

    @abstractmethod
    def get_model_info(self) -> Any:
        """Get model information."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load the model."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model."""
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, Any]) -> Any:
        """Run inference."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {}


class PluginRegistry:
    """
    Registry for model plugins.

    Supports dynamic loading and discovery of plugins.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_instances: Dict[str, ModelPlugin] = {}
        self._plugin_classes: Dict[str, Type[ModelPlugin]] = {}

    def register_plugin(
        self,
        plugin_info: PluginInfo,
        plugin_class: Optional[Type[ModelPlugin]] = None,
    ) -> bool:
        """
        Register a plugin.

        Args:
            plugin_info: Plugin information
            plugin_class: Optional plugin class (auto-loaded if None)

        Returns:
            True if registration successful
        """
        if plugin_info.name in self._plugins:
            logger.warning(f"Plugin already registered: {plugin_info.name}")
            return False

        # Load plugin class if not provided
        if plugin_class is None:
            try:
                plugin_class = self._load_plugin_class(plugin_info)
            except Exception as e:
                logger.error(f"Failed to load plugin class: {e}")
                return False

        # Validate plugin class
        if not issubclass(plugin_class, ModelPlugin):
            logger.error(f"Plugin class must inherit from ModelPlugin: {plugin_info.name}")
            return False

        self._plugins[plugin_info.name] = plugin_info
        self._plugin_classes[plugin_info.name] = plugin_class

        logger.info(f"Registered plugin: {plugin_info.name} v{plugin_info.version}")
        return True

    def _load_plugin_class(self, plugin_info: PluginInfo) -> Type[ModelPlugin]:
        """Load plugin class from module."""
        try:
            module = importlib.import_module(plugin_info.module_path)
            plugin_class = getattr(module, plugin_info.class_name)

            if not inspect.isclass(plugin_class):
                raise ValueError(f"{plugin_info.class_name} is not a class")

            return plugin_class
        except Exception as e:
            raise RuntimeError(f"Failed to load plugin class: {e}") from e

    def create_plugin_instance(self, plugin_name: str) -> Optional[ModelPlugin]:
        """
        Create an instance of a plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin instance or None
        """
        if plugin_name in self._plugin_instances:
            return self._plugin_instances[plugin_name]

        plugin_info = self._plugins.get(plugin_name)
        if not plugin_info:
            logger.error(f"Plugin not found: {plugin_name}")
            return None

        plugin_class = self._plugin_classes.get(plugin_name)
        if not plugin_class:
            logger.error(f"Plugin class not found: {plugin_name}")
            return None

        try:
            instance = plugin_class()
            self._plugin_instances[plugin_name] = instance
            logger.debug(f"Created plugin instance: {plugin_name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create plugin instance: {e}")
            return None

    def discover_plugins(self, plugin_dir: str) -> int:
        """
        Discover plugins in a directory.

        Args:
            plugin_dir: Directory to search for plugins

        Returns:
            Number of plugins discovered
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return 0

        count = 0
        for plugin_file in plugin_path.glob("*.py"):
            try:
                # Try to load plugin from file
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for plugin classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, ModelPlugin) and obj != ModelPlugin:
                            # Found a plugin
                            plugin_info = PluginInfo(
                                name=name.lower(),
                                version=getattr(obj, "__version__", "1.0.0"),
                                plugin_type="model",
                                module_path=module_name,
                                class_name=name,
                                description=getattr(obj, "__doc__", ""),
                            )
                            if self.register_plugin(plugin_info, obj):
                                count += 1

            except Exception as e:
                logger.warning(f"Failed to load plugin from {plugin_file}: {e}")

        logger.info(f"Discovered {count} plugins from {plugin_dir}")
        return count

    def list_plugins(
        self,
        plugin_type: Optional[str] = None,
    ) -> List[PluginInfo]:
        """List registered plugins."""
        plugins = list(self._plugins.values())

        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]

        return plugins

    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin information."""
        return self._plugins.get(name)

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin."""
        if name not in self._plugins:
            return False

        # Unload instance if loaded
        if name in self._plugin_instances:
            instance = self._plugin_instances.pop(name)
            try:
                instance.unload()
            except Exception:
                pass

        del self._plugins[name]
        del self._plugin_classes[name]

        logger.info(f"Unregistered plugin: {name}")
        return True


# Singleton plugin registry
_plugin_registry: Optional[PluginRegistry] = None
_plugin_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry singleton."""
    global _plugin_registry
    if _plugin_registry is None:
        with _plugin_registry_lock:
            if _plugin_registry is None:
                _plugin_registry = PluginRegistry()
    return _plugin_registry


