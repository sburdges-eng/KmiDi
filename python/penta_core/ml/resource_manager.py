"""
Resource Manager - Centralized resource management for GPU, memory, and CPU.

Provides:
- GPU memory allocation tracking
- CPU thread pool management
- Memory usage monitoring
- Resource quota enforcement
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from contextlib import contextmanager

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .gpu_utils import get_available_devices, GPUDevice, DeviceType

    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources managed."""

    GPU_MEMORY = "gpu_memory"
    CPU_MEMORY = "cpu_memory"
    CPU_THREADS = "cpu_threads"
    GPU_COMPUTE = "gpu_compute"


@dataclass
class ResourceQuota:
    """Resource quota configuration."""

    resource_type: ResourceType
    max_usage: float  # Maximum allowed usage (MB for memory, count for threads)
    reserved: float = 0.0  # Reserved amount that cannot be allocated
    warning_threshold: float = 0.8  # Warn when usage exceeds this fraction

    def get_available(self) -> float:
        """Get available resource amount."""
        return max(0.0, self.max_usage - self.reserved)


@dataclass
class ResourceAllocation:
    """Tracks a resource allocation."""

    allocation_id: str
    resource_type: ResourceType
    amount: float
    owner: str  # Component/process that owns this allocation
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class ResourceManager:
    """
    Centralized resource manager for GPU, memory, and CPU resources.

    Tracks allocations, enforces quotas, and provides resource monitoring.
    """

    _instance: Optional[ResourceManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> ResourceManager:
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    # Create instance lock after singleton creation
                    cls._instance._lock = threading.RLock()
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Instance lock is created in __new__ for thread safety
        # All instance methods use self._lock for synchronization
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._quotas: Dict[ResourceType, ResourceQuota] = {}
        self._usage: Dict[ResourceType, float] = defaultdict(float)
        self._device_info: Dict[str, GPUDevice] = {}

        # Initialize default quotas
        self._initialize_default_quotas()

        # Detect available devices
        self._detect_devices()

        self._initialized = True
        logger.info("ResourceManager initialized")

    def _initialize_default_quotas(self):
        """Initialize default resource quotas."""
        # Default quotas will be set based on detected hardware
        # These are placeholders that will be updated by _detect_devices
        self._quotas[ResourceType.CPU_MEMORY] = ResourceQuota(
            resource_type=ResourceType.CPU_MEMORY,
            max_usage=8192.0,  # 8GB default
            reserved=1024.0,  # Reserve 1GB
        )
        self._quotas[ResourceType.CPU_THREADS] = ResourceQuota(
            resource_type=ResourceType.CPU_THREADS,
            max_usage=8.0,  # 8 threads default
            reserved=2.0,  # Reserve 2 threads
        )

    def _detect_devices(self):
        """Detect and configure available devices."""
        if HAS_GPU_UTILS:
            devices = get_available_devices()
            for device in devices:
                device_key = f"{device.device_type.value}:{device.index}"
                self._device_info[device_key] = device

                # Set GPU memory quota if available
                if device.device_type != DeviceType.CPU and device.memory_total_mb > 0:
                    # Use 90% of total memory, reserve 10%
                    max_memory = device.memory_total_mb * 0.9
                    reserved = device.memory_total_mb * 0.1

                    quota_key = f"{ResourceType.GPU_MEMORY.value}:{device_key}"
                    # Store per-device quotas in metadata
                    if ResourceType.GPU_MEMORY not in self._quotas:
                        # Initialize with first device
                        self._quotas[ResourceType.GPU_MEMORY] = ResourceQuota(
                            resource_type=ResourceType.GPU_MEMORY,
                            max_usage=max_memory,
                            reserved=reserved,
                        )

        logger.info(f"Detected {len(self._device_info)} devices")

    def set_quota(self, resource_type: ResourceType, quota: ResourceQuota):
        """Set resource quota."""
        with self._lock:
            self._quotas[resource_type] = quota
            logger.info(
                f"Set quota for {resource_type.value}: max={quota.max_usage}, reserved={quota.reserved}"
            )

    def get_quota(self, resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get resource quota."""
        return self._quotas.get(resource_type)

    def allocate(
        self,
        resource_type: ResourceType,
        amount: float,
        owner: str,
        allocation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Allocate resources.

        Args:
            resource_type: Type of resource to allocate
            amount: Amount to allocate
            owner: Owner identifier
            allocation_id: Optional allocation ID (auto-generated if None)
            metadata: Optional metadata

        Returns:
            Allocation ID if successful, None if allocation failed
        """
        with self._lock:
            quota = self._quotas.get(resource_type)
            if not quota:
                logger.warning(f"No quota set for {resource_type.value}")
                return None

            # Check if allocation would exceed quota
            current_usage = self._usage[resource_type]
            available = quota.get_available()

            if current_usage + amount > available:
                logger.warning(
                    f"Insufficient {resource_type.value}: "
                    f"requested={amount}, available={available - current_usage}, "
                    f"current_usage={current_usage}"
                )
                return None

            # Generate allocation ID if not provided
            if allocation_id is None:
                import uuid

                allocation_id = f"{resource_type.value}_{uuid.uuid4().hex[:8]}"

            # Create allocation
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=resource_type,
                amount=amount,
                owner=owner,
                metadata=metadata or {},
            )

            self._allocations[allocation_id] = allocation
            self._usage[resource_type] += amount

            # Check warning threshold
            usage_ratio = self._usage[resource_type] / quota.max_usage
            if usage_ratio >= quota.warning_threshold:
                logger.warning(
                    f"Resource usage high for {resource_type.value}: "
                    f"{usage_ratio:.1%} of quota used"
                )

            logger.debug(
                f"Allocated {amount} {resource_type.value} to {owner} "
                f"(allocation_id={allocation_id})"
            )

            return allocation_id

    def deallocate(self, allocation_id: str) -> bool:
        """
        Deallocate resources.

        Args:
            allocation_id: Allocation ID to deallocate

        Returns:
            True if deallocation successful, False if allocation not found
        """
        with self._lock:
            allocation = self._allocations.pop(allocation_id, None)
            if not allocation:
                logger.warning(f"Allocation not found: {allocation_id}")
                return False

            self._usage[allocation.resource_type] -= allocation.amount
            self._usage[allocation.resource_type] = max(0.0, self._usage[allocation.resource_type])

            logger.debug(
                f"Deallocated {allocation.amount} {allocation.resource_type.value} "
                f"from {allocation.owner} (allocation_id={allocation_id})"
            )

            return True

    def get_usage(self, resource_type: ResourceType) -> float:
        """Get current resource usage."""
        with self._lock:
            return self._usage[resource_type]

    def get_available(self, resource_type: ResourceType) -> float:
        """Get available resource amount."""
        with self._lock:
            quota = self._quotas.get(resource_type)
            if not quota:
                return 0.0
            return max(0.0, quota.get_available() - self._usage[resource_type])

    def get_usage_ratio(self, resource_type: ResourceType) -> float:
        """Get resource usage as ratio of quota."""
        with self._lock:
            quota = self._quotas.get(resource_type)
            if not quota or quota.max_usage == 0:
                return 0.0
            return self._usage[resource_type] / quota.max_usage

    def list_allocations(
        self,
        resource_type: Optional[ResourceType] = None,
        owner: Optional[str] = None,
    ) -> List[ResourceAllocation]:
        """List current allocations with optional filtering."""
        with self._lock:
            allocations = list(self._allocations.values())

            if resource_type:
                allocations = [a for a in allocations if a.resource_type == resource_type]
            if owner:
                allocations = [a for a in allocations if a.owner == owner]

            return allocations

    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """Get allocation by ID."""
        with self._lock:
            return self._allocations.get(allocation_id)

    def get_status(self) -> Dict:
        """Get comprehensive resource status."""
        with self._lock:
            status = {
                "quotas": {
                    rt.value: {
                        "max_usage": q.max_usage,
                        "reserved": q.reserved,
                        "available": q.get_available(),
                    }
                    for rt, q in self._quotas.items()
                },
                "usage": {
                    rt.value: {
                        "current": self._usage[rt],
                        "available": self.get_available(rt),
                        "ratio": self.get_usage_ratio(rt),
                    }
                    for rt in self._usage.keys()
                },
                "allocations": {
                    aid: {
                        "resource_type": alloc.resource_type.value,
                        "amount": alloc.amount,
                        "owner": alloc.owner,
                        "timestamp": alloc.timestamp,
                    }
                    for aid, alloc in self._allocations.items()
                },
                "devices": {
                    key: {
                        "name": dev.name,
                        "type": dev.device_type.value,
                        "memory_total_mb": dev.memory_total_mb,
                        "memory_free_mb": dev.memory_free_mb,
                    }
                    for key, dev in self._device_info.items()
                },
            }
            return status

    @contextmanager
    def temporary_allocation(
        self,
        resource_type: ResourceType,
        amount: float,
        owner: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Context manager for temporary resource allocation.

        Usage:
            with resource_manager.temporary_allocation(ResourceType.GPU_MEMORY, 512, "my_component"):
                # Use resources here
                pass
        """
        allocation_id = self.allocate(resource_type, amount, owner, metadata=metadata)
        if allocation_id is None:
            raise RuntimeError(f"Failed to allocate {amount} {resource_type.value}")

        try:
            yield allocation_id
        finally:
            self.deallocate(allocation_id)

    def cleanup_orphaned_allocations(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clean up allocations that are older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of allocations cleaned up
        """
        current_time = time.time()
        orphaned = []

        with self._lock:
            for allocation_id, allocation in list(self._allocations.items()):
                age = current_time - allocation.timestamp
                if age > max_age_seconds:
                    orphaned.append(allocation_id)

            for allocation_id in orphaned:
                self.deallocate(allocation_id)

        if orphaned:
            logger.warning(f"Cleaned up {len(orphaned)} orphaned allocations")

        return len(orphaned)

    def estimate_model_memory(
        self,
        model_size_mb: float,
        batch_size: int = 1,
        input_size_mb: float = 0.1,
    ) -> float:
        """
        Estimate memory required for model inference.

        Args:
            model_size_mb: Model file size in MB
            batch_size: Inference batch size
            input_size_mb: Input tensor size per sample in MB

        Returns:
            Estimated memory in MB
        """
        # Model weights
        memory = model_size_mb

        # Activations (rough estimate: 2x model size)
        memory += model_size_mb * 2

        # Input/output buffers
        memory += input_size_mb * batch_size * 2

        # Workspace (25% overhead)
        memory *= 1.25

        return memory


# Singleton access
def get_resource_manager() -> ResourceManager:
    """Get the resource manager singleton."""
    return ResourceManager()
