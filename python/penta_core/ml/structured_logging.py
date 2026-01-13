"""
Structured Logging - Enhanced logging with context and tracing.

Provides:
- Structured logging with context
- Log levels per component
- Request tracing
- Performance profiling hooks
"""

from __future__ import annotations

import logging
import time
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LogContext:
    """Logging context for request tracing."""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    component: str = ""
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


class StructuredLogger:
    """
    Structured logger with context and tracing support.
    """

    def __init__(self, name: str, component: str = ""):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            component: Component name
        """
        self.logger = logging.getLogger(name)
        self.component = component
        self._context: Optional[LogContext] = None
        self._context_stack: list = []
        self._local = threading.local()

    @property
    def context(self) -> Optional[LogContext]:
        """Get current logging context."""
        if hasattr(self._local, 'context'):
            return self._local.context
        return self._context

    @context.setter
    def context(self, value: Optional[LogContext]):
        """Set logging context."""
        self._local.context = value

    @contextmanager
    def trace(self, operation: str, **metadata):
        """
        Context manager for request tracing.

        Usage:
            with logger.trace("inference", model="my_model"):
                # operation code
                pass
        """
        old_context = self.context
        new_context = LogContext(
            component=self.component,
            operation=operation,
            metadata=metadata,
        )
        self.context = new_context
        self._context_stack.append(new_context)

        start_time = time.perf_counter()

        try:
            self.info(f"Starting {operation}", **metadata)
            yield new_context
        except Exception as e:
            self.error(f"Error in {operation}: {e}", exc_info=True)
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            self.info(f"Completed {operation} in {elapsed*1000:.1f}ms", **metadata)
            self._context_stack.pop()
            self.context = old_context

    def _format_message(self, message: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """Format message with context."""
        context = self.context
        extra = {
            "component": self.component,
            **(context.metadata if context else {}),
            **kwargs,
        }

        if context:
            extra["request_id"] = context.request_id
            extra["operation"] = context.operation

        # Format message with context
        if context:
            formatted = f"[{context.request_id}] {message}"
        else:
            formatted = message

        return formatted, extra

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        formatted, extra = self._format_message(message, **kwargs)
        self.logger.debug(formatted, extra=extra)

    def info(self, message: str, **kwargs):
        """Log info message."""
        formatted, extra = self._format_message(message, **kwargs)
        self.logger.info(formatted, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        formatted, extra = self._format_message(message, **kwargs)
        self.logger.warning(formatted, extra=extra)

    def error(self, message: str, **kwargs):
        """Log error message."""
        formatted, extra = self._format_message(message, **kwargs)
        self.logger.error(formatted, extra=extra)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        formatted, extra = self._format_message(message, **kwargs)
        self.logger.critical(formatted, extra=extra)

    @contextmanager
    def profile(self, operation: str, **metadata):
        """
        Context manager for performance profiling.

        Usage:
            with logger.profile("inference"):
                # operation code
                pass
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory

            self.info(
                f"Profile: {operation}",
                elapsed_ms=elapsed * 1000,
                memory_delta_mb=memory_delta,
                **metadata,
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


def get_logger(name: str, component: str = "") -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger(name, component)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    component: str = "penta_core_ml",
):
    """
    Setup structured logging.

    Args:
        level: Log level
        format_string: Optional format string
        component: Component name
    """
    if format_string is None:
        format_string = (
            "%(asctime)s [%(levelname)s] [%(component)s] "
            "[%(request_id)s] [%(operation)s] %(message)s"
        )

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
    )

    # Set default values for extra fields
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, 'component'):
            record.component = component
        if not hasattr(record, 'request_id'):
            record.request_id = ""
        if not hasattr(record, 'operation'):
            record.operation = ""
        return record

    logging.setLogRecordFactory(record_factory)
