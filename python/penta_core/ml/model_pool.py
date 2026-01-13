"""
Model Pool - Model caching and pooling system.

Provides:
- LRU cache for loaded models
- Lazy loading with pre-warming
- Memory-aware eviction policies
- Per-backend pooling (separate pools for ONNX, CoreML, etc.)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

try:
    from .inference_enhanced import EnhancedInferenceEngine, create_enhanced_engine

    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

try:
    from .resource_manager import ResourceManager, get_resource_manager, ResourceType

    HAS_RESOURCES = True
except ImportError:
    HAS_RESOURCES = False

from penta_core.ml.model_registry import ModelInfo, ModelBackend

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Model eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    MEMORY = "memory"  # Evict largest models first
    HYBRID = "hybrid"  # Combination of LRU and memory


@dataclass
class PooledModel:
    """Represents a pooled model."""

    model_info: ModelInfo
    engine: EnhancedInferenceEngine
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    memory_estimate_mb: float = 0.0
    loaded_at: float = field(default_factory=time.time)

    def touch(self):
        """Update last used time and use count."""
        self.last_used = time.time()
        self.use_count += 1


class ModelPool:
    """
    Model pool with caching and eviction policies.

    Manages a pool of loaded models to avoid repeated loading/unloading.
    """

    def __init__(
        self,
        max_size: int = 10,
        max_memory_mb: float = 4096.0,
        eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
        pre_warm: bool = True,
    ):
        """
        Initialize model pool.

        Args:
            max_size: Maximum number of models in pool
            max_memory_mb: Maximum memory usage in MB
            eviction_policy: Eviction policy to use
            pre_warm: Pre-warm models on first access
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.eviction_policy = eviction_policy
        self.pre_warm = pre_warm

        self._lock = threading.RLock()
        self._pools: Dict[ModelBackend, OrderedDict[str, PooledModel]] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}

        # Resource manager for memory tracking
        self._resource_manager: Optional[ResourceManager] = None
        if HAS_RESOURCES:
            try:
                self._resource_manager = get_resource_manager()
            except Exception:
                pass

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "loads": 0,
        }

    def get(
        self,
        model_info: ModelInfo,
        auto_load: bool = True,
    ) -> Optional[EnhancedInferenceEngine]:
        """
        Get model from pool.

        Args:
            model_info: Model information
            auto_load: Automatically load if not in pool

        Returns:
            Inference engine or None
        """
        with self._lock:
            pool = self._get_pool(model_info.backend)
            model_key = self._get_model_key(model_info)

            # Check if model is in pool
            if model_key in pool:
                pooled_model = pool[model_key]
                pooled_model.touch()
                self._access_times[model_key] = time.time()
                self._access_counts[model_key] = self._access_counts.get(model_key, 0) + 1

                # Move to end (most recently used)
                pool.move_to_end(model_key)

                self._stats["hits"] += 1
                logger.debug(f"Model pool hit: {model_key}")
                return pooled_model.engine

            # Model not in pool
            self._stats["misses"] += 1
            logger.debug(f"Model pool miss: {model_key}")

            if not auto_load:
                return None

            # Load model
            return self._load_model(model_info)

    def _load_model(self, model_info: ModelInfo) -> Optional[EnhancedInferenceEngine]:
        """Load a model into the pool."""
        if not HAS_INFERENCE:
            logger.error("Inference engine not available")
            return None

        model_key = self._get_model_key(model_info)
        pool = self._get_pool(model_info.backend)

        # Check if we need to evict
        if len(pool) >= self.max_size:
            self._evict_model(model_info.backend)

        # Estimate memory
        memory_estimate = self._estimate_model_memory(model_info)

        # Check memory limit
        current_memory = sum(m.memory_estimate_mb for m in pool.values())
        if current_memory + memory_estimate > self.max_memory_mb:
            # Evict based on memory
            self._evict_by_memory(memory_estimate)

        # Create engine
        try:
            engine = create_enhanced_engine(model_info)
            if not engine.load():
                logger.error(f"Failed to load model: {model_key}")
                return None

            pooled_model = PooledModel(
                model_info=model_info,
                engine=engine,
                memory_estimate_mb=memory_estimate,
            )

            pool[model_key] = pooled_model
            self._access_times[model_key] = time.time()
            self._access_counts[model_key] = 1

            self._stats["loads"] += 1
            logger.info(f"Loaded model into pool: {model_key}")

            return engine

        except Exception as e:
            logger.error(f"Error loading model {model_key}: {e}")
            return None

    def _evict_model(self, backend: ModelBackend):
        """Evict a model based on eviction policy."""
        pool = self._get_pool(backend)

        if not pool:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used (first in OrderedDict)
            model_key = next(iter(pool))
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            model_key = min(pool.keys(), key=lambda k: self._access_counts.get(k, 0))
        elif self.eviction_policy == EvictionPolicy.MEMORY:
            # Evict largest model
            model_key = max(pool.keys(), key=lambda k: pool[k].memory_estimate_mb)
        else:  # HYBRID
            # Combine LRU and memory
            # Score = (age * 0.5) + (memory_ratio * 0.5)
            current_time = time.time()
            scores = {}
            total_memory = sum(m.memory_estimate_mb for m in pool.values())

            for key, pooled_model in pool.items():
                age = current_time - pooled_model.last_used
                memory_ratio = (
                    pooled_model.memory_estimate_mb / total_memory if total_memory > 0 else 0
                )
                score = (age / 3600.0) * 0.5 + memory_ratio * 0.5
                scores[key] = score

            model_key = max(scores.keys(), key=lambda k: scores[k])

        # Evict the selected model
        pooled_model = pool.pop(model_key)
        try:
            pooled_model.engine.unload()
        except Exception as e:
            logger.warning(f"Error unloading evicted model: {e}")

        del self._access_times[model_key]
        del self._access_counts[model_key]

        self._stats["evictions"] += 1
        logger.info(f"Evicted model from pool: {model_key}")

    def _evict_by_memory(self, required_mb: float):
        """Evict models until enough memory is available."""
        total_memory = sum(
            m.memory_estimate_mb for pool in self._pools.values() for m in pool.values()
        )

        while total_memory + required_mb > self.max_memory_mb:
            # Find backend with most memory usage
            backend_memory = {
                backend: sum(m.memory_estimate_mb for m in pool.values())
                for backend, pool in self._pools.items()
            }

            if not backend_memory:
                break

            # Evict from backend with most memory
            max_backend = max(backend_memory.keys(), key=lambda b: backend_memory[b])
            self._evict_model(max_backend)

            # Recalculate total memory
            total_memory = sum(
                m.memory_estimate_mb for pool in self._pools.values() for m in pool.values()
            )

    def _get_pool(self, backend: ModelBackend) -> OrderedDict[str, PooledModel]:
        """Get or create pool for backend."""
        if backend not in self._pools:
            self._pools[backend] = OrderedDict()
        return self._pools[backend]

    def _get_model_key(self, model_info: ModelInfo) -> str:
        """Get unique key for model."""
        return f"{model_info.name}:{model_info.version}:{model_info.backend.value}"

    def _estimate_model_memory(self, model_info: ModelInfo) -> float:
        """Estimate memory required for model."""
        # Try to get from resource manager
        if self._resource_manager:
            try:
                # Estimate based on model file size if available
                import os

                if os.path.exists(model_info.path):
                    file_size_mb = os.path.getsize(model_info.path) / (1024 * 1024)
                    return self._resource_manager.estimate_model_memory(file_size_mb)
            except Exception:
                pass

        # Default estimate: 100MB per model
        return 100.0

    def pre_warm(self, model_infos: List[ModelInfo]):
        """Pre-warm models by loading them into the pool."""
        logger.info(f"Pre-warming {len(model_infos)} models...")

        for model_info in model_infos:
            try:
                self.get(model_info, auto_load=True)
            except Exception as e:
                logger.warning(f"Failed to pre-warm {model_info.name}: {e}")

    def clear(self, backend: Optional[ModelBackend] = None):
        """Clear model pool."""
        with self._lock:
            if backend:
                pool = self._get_pool(backend)
                for pooled_model in pool.values():
                    try:
                        pooled_model.engine.unload()
                    except Exception as e:
                        logger.warning(f"Error unloading model: {e}")
                pool.clear()
            else:
                for pool in self._pools.values():
                    for pooled_model in pool.values():
                        try:
                            pooled_model.engine.unload()
                        except Exception as e:
                            logger.warning(f"Error unloading model: {e}")
                self._pools.clear()
                self._access_times.clear()
                self._access_counts.clear()

            logger.info("Model pool cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_models = sum(len(pool) for pool in self._pools.values())
            total_memory = sum(
                m.memory_estimate_mb for pool in self._pools.values() for m in pool.values()
            )

            hit_rate = (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            )

            return {
                "total_models": total_models,
                "total_memory_mb": total_memory,
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_mb,
                "eviction_policy": self.eviction_policy.value,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "loads": self._stats["loads"],
                "by_backend": {backend.value: len(pool) for backend, pool in self._pools.items()},
            }

    def list_loaded_models(self) -> List[str]:
        """List currently loaded model keys."""
        with self._lock:
            models = []
            for pool in self._pools.values():
                models.extend(pool.keys())
            return models


# Singleton model pool
_model_pool: Optional[ModelPool] = None
_model_pool_lock = threading.Lock()


def get_model_pool() -> ModelPool:
    """Get the global model pool singleton."""
    global _model_pool
    if _model_pool is None:
        with _model_pool_lock:
            if _model_pool is None:
                _model_pool = ModelPool()
    return _model_pool
