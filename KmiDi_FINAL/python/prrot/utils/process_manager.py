"""
Process Manager - Ensure single worker process and proper cleanup

For 16GB Mac systems, ensures only one ML worker runs at a time
and processes exit fully to reclaim memory.
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from typing import Optional
import signal

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages worker process lifecycle for 16GB constraint compliance"""

    LOCK_FILE = Path("/tmp/prrot_worker.lock")
    MAX_PROCESS_AGE_SECONDS = 3600  # 1 hour - kill stale processes

    def __init__(self):
        """Initialize process manager"""
        self.lock_file = self.LOCK_FILE
        self.pid_file = self.lock_file.with_suffix(".pid")

    def acquire_lock(self) -> bool:
        """
        Acquire lock for single worker process

        Returns:
            True if lock acquired, False if another worker is running
        """
        # Check if lock file exists
        if self.lock_file.exists():
            # Check if process is still running
            try:
                pid = int(self.pid_file.read_text().strip())
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    # Check process age
                    age = time.time() - process.create_time()
                    if age < self.MAX_PROCESS_AGE_SECONDS:
                        logger.warning(f"Another PRROT worker is running (PID: {pid})")
                        return False
                    else:
                        # Stale process - kill it
                        logger.warning(
                            f"Stale PRROT worker process found (PID: {pid}, age: {age:.0f}s)"
                        )
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                            pass
                # Process doesn't exist - remove stale lock
                self.lock_file.unlink(missing_ok=True)
                self.pid_file.unlink(missing_ok=True)
            except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
                # Lock file exists but PID is invalid - remove stale lock
                self.lock_file.unlink(missing_ok=True)
                self.pid_file.unlink(missing_ok=True)

        # Acquire lock
        try:
            self.pid_file.write_text(str(os.getpid()))
            self.lock_file.touch()
            logger.info(f"Process lock acquired (PID: {os.getpid()})")
            return True
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self) -> None:
        """Release process lock"""
        try:
            self.lock_file.unlink(missing_ok=True)
            self.pid_file.unlink(missing_ok=True)
            logger.info("Process lock released")
        except Exception as e:
            logger.warning(f"Failed to release lock: {e}")

    def cleanup_on_exit(self) -> None:
        """Cleanup handler for process exit"""
        self.release_lock()

    def register_exit_handlers(self) -> None:
        """Register signal handlers for cleanup"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, cleaning up...")
            self.cleanup_on_exit()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def check_system_memory(self) -> tuple[bool, str]:
        """
        Check if system has sufficient memory for worker

        Returns:
            (has_sufficient_memory, message)
        """
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        # Need at least 10GB available (8GB for worker + 2GB system reserve)
        if available_gb < 10.0:
            return False, f"Insufficient memory: {available_gb:.2f}GB available, need 10GB"

        return True, f"Memory OK: {available_gb:.2f}GB available"


def ensure_single_worker(func):
    """Decorator to ensure only one worker runs at a time"""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        manager = ProcessManager()

        # Check system memory before acquiring lock
        has_memory, memory_msg = manager.check_system_memory()
        if not has_memory:
            logger.error(f"Cannot start worker: {memory_msg}")
            sys.exit(1)

        if not manager.acquire_lock():
            logger.error("Another PRROT worker is already running. Exiting.")
            sys.exit(1)

        manager.register_exit_handlers()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            manager.cleanup_on_exit()

    return wrapper
