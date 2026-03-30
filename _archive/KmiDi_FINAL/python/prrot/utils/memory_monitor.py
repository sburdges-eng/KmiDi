"""
Memory Monitor - Monitor memory usage for 16GB constraint compliance

Ensures worker processes stay within memory limits on 16GB systems.
"""

import psutil
import os
from typing import Optional, Tuple


class MemoryMonitor:
    """Monitor memory usage for 16GB constraint compliance"""

    # Memory limits for 16GB system
    SYSTEM_TOTAL_GB = 16
    WORKER_MAX_GB = 8  # Worker should use max 8GB (leave room for system)
    WORKER_WARN_GB = 6  # Warn if approaching limit

    def __init__(self, process: Optional[psutil.Process] = None):
        """Initialize memory monitor"""
        if process is None:
            process = psutil.Process(os.getpid())
        self.process = process

    def get_memory_usage(self) -> Tuple[float, float, float]:
        """
        Get memory usage statistics

        Returns:
            (process_memory_gb, system_available_gb, system_total_gb)
        """
        process_info = self.process.memory_info()
        process_memory_gb = process_info.rss / (1024**3)

        system_memory = psutil.virtual_memory()
        system_available_gb = system_memory.available / (1024**3)
        system_total_gb = system_memory.total / (1024**3)

        return process_memory_gb, system_available_gb, system_total_gb

    def check_memory_limit(self) -> Tuple[bool, str]:
        """
        Check if memory usage is within limits

        Returns:
            (is_within_limit, warning_message)
        """
        process_mem, system_available, system_total = self.get_memory_usage()

        # Check process memory
        if process_mem > self.WORKER_MAX_GB:
            return (
                False,
                f"Process memory ({process_mem:.2f}GB) exceeds limit ({self.WORKER_MAX_GB}GB)",
            )

        if process_mem > self.WORKER_WARN_GB:
            return True, f"Warning: Process memory ({process_mem:.2f}GB) approaching limit"

        # Check system available memory
        if system_available < 2.0:  # Less than 2GB available
            return False, f"System available memory ({system_available:.2f}GB) is too low"

        return True, ""

    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics"""
        process_mem, system_available, system_total = self.get_memory_usage()

        return {
            "process_memory_gb": process_mem,
            "system_available_gb": system_available,
            "system_total_gb": system_total,
            "process_memory_percent": (process_mem / system_total) * 100,
            "system_available_percent": (system_available / system_total) * 100,
            "within_limit": process_mem <= self.WORKER_MAX_GB,
            "warning_threshold_exceeded": process_mem > self.WORKER_WARN_GB,
        }
