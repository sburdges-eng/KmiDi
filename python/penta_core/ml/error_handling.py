"""
Error Handling - Retry logic, fallback chains, and circuit breaker pattern.

Provides robust error handling for ML inference with:
- Retry logic with exponential backoff
- Fallback chain (ML model → rule-based → default)
- Error classification and recovery strategies
- Circuit breaker pattern for failing models
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    TRANSIENT = "transient"  # Temporary, may succeed on retry
    PERMANENT = "permanent"  # Won't succeed on retry
    RESOURCE = "resource"  # Resource-related (memory, GPU)
    NETWORK = "network"  # Network-related
    MODEL = "model"  # Model-related (corrupt, incompatible)
    INPUT = "input"  # Input-related (invalid shape, type)


@dataclass
class ErrorInfo:
    """Information about an error."""
    error: Exception
    category: ErrorCategory
    retryable: bool
    message: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies errors into categories."""

    @staticmethod
    def classify(error: Exception) -> ErrorInfo:
        """
        Classify an error.

        Args:
            error: Exception to classify

        Returns:
            ErrorInfo with classification
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Transient errors (network, timeouts, temporary resource issues)
        if any(keyword in error_str for keyword in [
            "timeout", "connection", "temporary", "busy", "locked",
            "rate limit", "throttle", "overload"
        ]):
            return ErrorInfo(
                error=error,
                category=ErrorCategory.TRANSIENT,
                retryable=True,
                message=str(error),
            )

        # Resource errors
        if any(keyword in error_str for keyword in [
            "memory", "out of memory", "cuda", "gpu", "allocation",
            "insufficient", "quota", "limit"
        ]) or error_type in ["MemoryError", "RuntimeError"]:
            return ErrorInfo(
                error=error,
                category=ErrorCategory.RESOURCE,
                retryable=False,  # Usually need to free resources first
                message=str(error),
            )

        # Model errors
        if any(keyword in error_str for keyword in [
            "model", "checkpoint", "weight", "shape", "dimension",
            "incompatible", "corrupt", "invalid model"
        ]):
            return ErrorInfo(
                error=error,
                category=ErrorCategory.MODEL,
                retryable=False,
                message=str(error),
            )

        # Input errors
        if any(keyword in error_str for keyword in [
            "input", "shape", "dtype", "type", "invalid input",
            "dimension mismatch"
        ]):
            return ErrorInfo(
                error=error,
                category=ErrorCategory.INPUT,
                retryable=False,
                message=str(error),
            )

        # Network errors
        if any(keyword in error_str for keyword in [
            "network", "connection", "http", "socket", "dns"
        ]):
            return ErrorInfo(
                error=error,
                category=ErrorCategory.NETWORK,
                retryable=True,
                message=str(error),
            )

        # Default: permanent error
        return ErrorInfo(
            error=error,
            category=ErrorCategory.PERMANENT,
            retryable=False,
            message=str(error),
        )


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 5.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    error_classifier: Optional[ErrorClassifier] = None,
) -> T:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        config: Retry configuration
        error_classifier: Optional error classifier

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()
    if error_classifier is None:
        error_classifier = ErrorClassifier()

    last_error = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_info = error_classifier.classify(e)

            # Don't retry if error is not retryable
            if not error_info.retryable:
                logger.debug(f"Non-retryable error: {error_info.message}")
                raise

            # Don't retry if we've exhausted attempts
            if attempt >= config.max_retries:
                logger.warning(
                    f"Retry exhausted after {config.max_retries} attempts: "
                    f"{error_info.message}"
                )
                raise

            # Calculate delay
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt),
                config.max_delay
            )

            # Add jitter
            if config.jitter:
                import random
                delay = delay * (0.5 + random.random() * 0.5)

            logger.debug(
                f"Retry attempt {attempt + 1}/{config.max_retries} after "
                f"{delay:.2f}s: {error_info.message}"
            )

            time.sleep(delay)

    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry logic failed unexpectedly")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout_seconds: float = 60.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern for failing operations.

    Prevents cascading failures by stopping requests to failing services.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    def call(self, func: Callable[[], T]) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute

        Returns:
            Function result

        Raises:
            Exception if circuit is open or function fails
        """
        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.name}: Attempting recovery")
            else:
                raise RuntimeError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Last failure: {self.last_failure_time}"
                )

        # Execute function
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery."""
        if self.last_failure_time is None:
            return False

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout_seconds

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls += 1

            # Check max calls first - if exceeded, go back to open regardless of success
            if self.half_open_calls >= self.config.half_open_max_calls:
                # Too many calls in half-open, go back to open
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                self.half_open_calls = 0  # Reset counter when transitioning to OPEN
                logger.warning(
                    f"Circuit breaker {self.name}: OPEN "
                    f"(too many calls in half-open state)"
                )
            elif self.success_count >= self.config.success_threshold:
                # Enough successes to close the circuit
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name}: CLOSED (recovered)")
        else:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure in half-open, go back to open
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.half_open_calls = 0  # Reset counter when transitioning to OPEN
            logger.warning(
                f"Circuit breaker {self.name}: OPEN (failure in half-open state)"
            )
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(
                f"Circuit breaker {self.name}: OPEN "
                f"(failure threshold reached: {self.failure_count})"
            )

    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        logger.info(f"Circuit breaker {self.name}: Reset")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


class FallbackChain:
    """
    Fallback chain for graceful degradation.

    Tries multiple strategies in order until one succeeds.
    """

    def __init__(
        self,
        strategies: List[Callable[[], T]],
        strategy_names: Optional[List[str]] = None,
    ):
        self.strategies = strategies
        self.strategy_names = strategy_names or [
            f"strategy_{i}" for i in range(len(strategies))
        ]

        if len(self.strategies) != len(self.strategy_names):
            raise ValueError("strategies and strategy_names must have same length")

    def execute(self) -> T:
        """
        Execute fallback chain.

        Returns:
            Result from first successful strategy

        Raises:
            Last exception if all strategies fail
        """
        last_error = None

        for strategy, name in zip(self.strategies, self.strategy_names):
            try:
                logger.debug(f"Trying fallback strategy: {name}")
                result = strategy()
                logger.debug(f"Fallback strategy succeeded: {name}")
                return result
            except Exception as e:
                last_error = e
                logger.debug(f"Fallback strategy failed: {name} - {e}")
                continue

        # All strategies failed
        if last_error:
            logger.error("All fallback strategies failed")
            raise last_error
        raise RuntimeError("No fallback strategies provided")
