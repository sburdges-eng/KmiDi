"""
Enhanced Inference Engine - With error handling, retry logic, and circuit breakers.

Wraps the base inference engine with:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Fallback chains
- Enhanced error handling
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Callable
import numpy as np

from .error_handling import (
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    FallbackChain,
    retry_with_backoff,
    ErrorClassifier,
)

# Try to import base inference engine
try:
    from penta_core.ml.inference import (
        InferenceEngine,
        InferenceResult,
        create_engine,
        create_engine_by_name,
    )
    HAS_BASE_INFERENCE = True
except ImportError:
    # Fallback if base inference not available
    HAS_BASE_INFERENCE = False
    InferenceEngine = None
    InferenceResult = None

from penta_core.ml.model_registry import ModelInfo, ModelBackend, get_model

logger = logging.getLogger(__name__)


class EnhancedInferenceEngine:
    """
    Enhanced inference engine with error handling, retry logic, and circuit breakers.

    Wraps a base InferenceEngine with:
    - Automatic retry on transient errors
    - Circuit breaker to prevent cascading failures
    - Fallback chains for graceful degradation
    """

    def __init__(
        self,
        model_info: ModelInfo,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_strategies: Optional[List[Callable]] = None,
    ):
        """
        Initialize enhanced inference engine.

        Args:
            model_info: Model information
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            fallback_strategies: Optional fallback strategies
        """
        self.model_info = model_info
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        # Create base engine
        if HAS_BASE_INFERENCE:
            self._base_engine = create_engine(model_info)
        else:
            self._base_engine = None
            logger.warning("Base inference engine not available")

        # Create circuit breaker
        self._circuit_breaker = CircuitBreaker(
            name=f"inference_{model_info.name}",
            config=self.circuit_breaker_config,
        )

        # Error classifier
        self._error_classifier = ErrorClassifier()

        # Fallback strategies
        self._fallback_strategies = fallback_strategies or []

    def load(self) -> bool:
        """Load the model with error handling."""
        if not self._base_engine:
            return False

        def _load():
            return self._base_engine.load()

        try:
            return retry_with_backoff(
                _load,
                config=self.retry_config,
                error_classifier=self._error_classifier,
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_info.name}: {e}")
            return False

    def unload(self) -> None:
        """Unload the model."""
        if self._base_engine:
            try:
                self._base_engine.unload()
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        use_fallback: bool = True,
    ) -> InferenceResult:
        """
        Run inference with error handling and fallbacks.

        Args:
            inputs: Input tensors
            use_fallback: Use fallback strategies if inference fails

        Returns:
            Inference result

        Raises:
            Exception if inference fails and no fallback succeeds
        """
        if not self._base_engine or not self._base_engine.is_loaded():
            raise RuntimeError(f"Model {self.model_info.name} not loaded")

        def _infer():
            return self._base_engine.infer(inputs)

        # Build fallback chain if enabled
        if use_fallback and self._fallback_strategies:
            strategies = [_infer] + self._fallback_strategies
            strategy_names = ["primary"] + [
                f"fallback_{i}" for i in range(len(self._fallback_strategies))
            ]
            fallback_chain = FallbackChain(strategies, strategy_names)

            def _infer_with_fallback():
                return fallback_chain.execute()

            # Execute through circuit breaker with retry
            def _execute():
                return self._circuit_breaker.call(_infer_with_fallback)

            return retry_with_backoff(
                _execute,
                config=self.retry_config,
                error_classifier=self._error_classifier,
            )
        else:
            # Execute through circuit breaker with retry
            def _execute():
                return self._circuit_breaker.call(_infer)

            return retry_with_backoff(
                _execute,
                config=self.retry_config,
                error_classifier=self._error_classifier,
            )

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._base_engine is not None and self._base_engine.is_loaded()

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return self._circuit_breaker.get_status()

    def reset_circuit_breaker(self):
        """Reset circuit breaker."""
        self._circuit_breaker.reset()


def create_enhanced_engine(
    model_info: ModelInfo,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    fallback_strategies: Optional[List[Callable]] = None,
) -> EnhancedInferenceEngine:
    """
    Create an enhanced inference engine.

    Args:
        model_info: Model information
        retry_config: Optional retry configuration
        circuit_breaker_config: Optional circuit breaker configuration
        fallback_strategies: Optional fallback strategies

    Returns:
        Enhanced inference engine
    """
    return EnhancedInferenceEngine(
        model_info=model_info,
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        fallback_strategies=fallback_strategies,
    )


def create_enhanced_engine_by_name(
    name: str,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    fallback_strategies: Optional[List[Callable]] = None,
) -> Optional[EnhancedInferenceEngine]:
    """
    Create an enhanced inference engine by model name.

    Args:
        name: Registered model name
        retry_config: Optional retry configuration
        circuit_breaker_config: Optional circuit breaker configuration
        fallback_strategies: Optional fallback strategies

    Returns:
        Enhanced inference engine or None if model not found
    """
    model_info = get_model(name)
    if model_info:
        return create_enhanced_engine(
            model_info=model_info,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            fallback_strategies=fallback_strategies,
        )
    return None
