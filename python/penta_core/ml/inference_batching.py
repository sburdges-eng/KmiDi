"""
Inference Batching - Automatic batching support for inference.

Provides:
- Automatic batching of requests
- Dynamic batch sizing based on latency targets
- Batch timeout handling
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np

try:
    from .inference_enhanced import EnhancedInferenceEngine

    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

try:
    from penta_core.ml.inference import InferenceResult

    HAS_INFERENCE_RESULT = True
except ImportError:
    # Fallback if inference module not available
    HAS_INFERENCE_RESULT = False
    InferenceResult = None

logger = logging.getLogger(__name__)


@dataclass
class BatchedRequest:
    """Request in a batch."""

    inputs: Dict[str, np.ndarray]
    callback: Optional[Callable[[InferenceResult], None]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for batching."""

    max_batch_size: int = 32
    min_batch_size: int = 1
    timeout_ms: float = 10.0
    max_wait_ms: float = 50.0
    target_latency_ms: float = 100.0
    adaptive: bool = True


class BatchProcessor:
    """
    Processes inference requests in batches.

    Automatically batches requests to improve throughput.
    """

    def __init__(
        self,
        engine: EnhancedInferenceEngine,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize batch processor.

        Args:
            engine: Inference engine to use
            config: Batch configuration
        """
        self.engine = engine
        self.config = config or BatchConfig()

        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._current_batch_size = self.config.min_batch_size
        self._latency_history: deque = deque(maxlen=100)

        # Statistics
        self._stats = {
            "batches_processed": 0,
            "requests_processed": 0,
            "average_batch_size": 0.0,
            "average_latency_ms": 0.0,
        }

    def add_request(
        self,
        inputs: Dict[str, np.ndarray],
        callback: Optional[Callable[[InferenceResult], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add request to batch queue.

        Args:
            inputs: Input tensors
            callback: Optional callback for result
            metadata: Optional metadata
        """
        request = BatchedRequest(
            inputs=inputs,
            callback=callback,
            metadata=metadata or {},
        )

        with self._lock:
            self._queue.append(request)

    def process_batch(
        self,
        timeout_ms: Optional[float] = None,
    ) -> List[InferenceResult]:
        """
        Process a batch of requests.

        Args:
            timeout_ms: Optional timeout override

        Returns:
            List of inference results
        """
        timeout = timeout_ms or self.config.timeout_ms
        deadline = time.time() + (timeout / 1000.0)

        # Collect requests
        batch: List[BatchedRequest] = []
        target_batch_size = (
            self._current_batch_size if self.config.adaptive else self.config.max_batch_size
        )
        target_batch_size = max(self.config.min_batch_size, target_batch_size)

        with self._lock:
            # Get requests up to target batch size or timeout
            while len(batch) < target_batch_size:
                if not self._queue:
                    break

                request = self._queue.popleft()
                batch.append(request)

                # Check timeout
                if time.time() >= deadline:
                    break

        if not batch:
            return []

        # Ensure minimum batch size
        if len(batch) < self.config.min_batch_size and self._queue:
            # Wait a bit more for more requests
            wait_until = time.time() + (self.config.max_wait_ms / 1000.0)
            while len(batch) < self.config.min_batch_size and time.time() < wait_until:
                with self._lock:
                    if self._queue:
                        batch.append(self._queue.popleft())
                    else:
                        time.sleep(0.001)  # Small sleep to avoid busy waiting

        # Process batch
        return self._execute_batch(batch)

    def _execute_batch(self, batch: List[BatchedRequest]) -> List[InferenceResult]:
        """Execute a batch of requests."""
        if not batch:
            return []

        start_time = time.perf_counter()

        try:
            # Combine inputs into batch
            batched_inputs = self._combine_inputs([r.inputs for r in batch])

            # Run inference
            result = self.engine.infer(batched_inputs, use_fallback=False)

            # Validate result before splitting
            if result is None:
                # Create error results for all requests
                results = self._create_error_results(len(batch))
            else:
                # Split results
                results = self._split_results(result, len(batch))

            # Update statistics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(len(batch), latency_ms)

            # Call callbacks
            for request, result_item in zip(batch, results):
                if request.callback:
                    try:
                        request.callback(result_item)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return error results
            error_results = []

            # Safely get model info with null checks
            model_name = "unknown"
            backend = None
            if self.engine and hasattr(self.engine, "model_info") and self.engine.model_info:
                model_name = self.engine.model_info.name
                backend = self.engine.model_info.backend

            for request in batch:
                error_result = InferenceResult(
                    outputs={},
                    latency_ms=0.0,
                    model_name=model_name,
                    backend=backend,
                )
                error_results.append(error_result)

                if request.callback:
                    try:
                        request.callback(error_result)
                    except Exception as callback_error:
                        logger.error(f"Callback error: {callback_error}")

            return error_results

    def _combine_inputs(self, inputs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine multiple input dictionaries into batched inputs."""
        if not inputs_list:
            return {}

        # Get all input keys
        all_keys = set()
        for inputs in inputs_list:
            all_keys.update(inputs.keys())

        batched = {}
        for key in all_keys:
            # Stack arrays along batch dimension
            arrays = [inputs.get(key) for inputs in inputs_list]
            arrays = [arr for arr in arrays if arr is not None]

            if arrays:
                batched[key] = np.stack(arrays, axis=0)

        return batched

    def _create_error_results(self, batch_size: int) -> List[InferenceResult]:
        """Create error InferenceResult objects for failed batch."""
        if not HAS_INFERENCE_RESULT or InferenceResult is None:
            # Fallback: return empty list if InferenceResult not available
            return []

        import numpy as np

        error_results = []
        model_name = "unknown"
        backend = None

        if hasattr(self.engine, "model_info") and self.engine.model_info:
            model_name = self.engine.model_info.name
            backend = self.engine.model_info.backend

        for _ in range(batch_size):
            error_result = InferenceResult(
                outputs={},
                latency_ms=0.0,
                model_name=model_name,
                backend=backend,
            )
            error_results.append(error_result)
        return error_results

    def _split_results(self, result: InferenceResult, batch_size: int) -> List[InferenceResult]:
        """Split batched result into individual results."""
        if not result:
            return self._create_error_results(batch_size)

        results = []
        for i in range(batch_size):
            # Extract individual result from batch
            individual_outputs = {}
            for key, output in result.outputs.items():
                if output.ndim > 0:
                    individual_outputs[key] = output[i]
                else:
                    individual_outputs[key] = output

            individual_result = InferenceResult(
                outputs=individual_outputs,
                latency_ms=result.latency_ms / batch_size,  # Approximate per-item latency
                model_name=result.model_name,
                backend=result.backend,
                confidence=result.confidence,
                labels=result.labels,
            )
            results.append(individual_result)

        return results

    def _update_stats(self, batch_size: int, latency_ms: float):
        """Update statistics and adapt batch size."""
        self._stats["batches_processed"] += 1
        self._stats["requests_processed"] += batch_size

        # Update running averages
        total_batches = self._stats["batches_processed"]
        self._stats["average_batch_size"] = (
            self._stats["average_batch_size"] * (total_batches - 1) + batch_size
        ) / total_batches
        self._stats["average_latency_ms"] = (
            self._stats["average_latency_ms"] * (total_batches - 1) + latency_ms
        ) / total_batches

        # Store latency for adaptive sizing
        self._latency_history.append(latency_ms)

        # Adaptive batch sizing
        if self.config.adaptive and len(self._latency_history) >= 10:
            avg_latency = sum(self._latency_history) / len(self._latency_history)

            if avg_latency < self.config.target_latency_ms * 0.8:
                # Latency is low, can increase batch size
                self._current_batch_size = min(
                    self._current_batch_size + 1, self.config.max_batch_size
                )
            elif avg_latency > self.config.target_latency_ms * 1.2:
                # Latency is high, decrease batch size
                self._current_batch_size = max(
                    self._current_batch_size - 1, self.config.min_batch_size
                )

    def get_queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            "queue_size": self.get_queue_size(),
            "current_batch_size": self._current_batch_size,
            "batches_processed": self._stats["batches_processed"],
            "requests_processed": self._stats["requests_processed"],
            "average_batch_size": self._stats["average_batch_size"],
            "average_latency_ms": self._stats["average_latency_ms"],
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "min_batch_size": self.config.min_batch_size,
                "timeout_ms": self.config.timeout_ms,
                "target_latency_ms": self.config.target_latency_ms,
                "adaptive": self.config.adaptive,
            },
        }

    def clear_queue(self):
        """Clear the request queue."""
        with self._lock:
            self._queue.clear()


class BatchedInferenceEngine:
    """
    Inference engine wrapper with automatic batching.

    Automatically batches requests to improve throughput.
    """

    def __init__(
        self,
        engine: EnhancedInferenceEngine,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize batched inference engine.

        Args:
            engine: Base inference engine
            config: Batch configuration
        """
        self.engine = engine
        self.processor = BatchProcessor(engine, config)

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        wait: bool = True,
        callback: Optional[Callable[[InferenceResult], None]] = None,
    ) -> Optional[InferenceResult]:
        """
        Run inference (may be batched).

        Args:
            inputs: Input tensors
            wait: Wait for result (if False, uses callback)
            callback: Optional callback for async result

        Returns:
            Inference result if wait=True, None otherwise
        """
        if wait:
            # Synchronous: process immediately
            return self.engine.infer(inputs, use_fallback=True)
        else:
            # Asynchronous: add to batch queue
            self.processor.add_request(inputs, callback=callback)
            return None

    def process_pending(self) -> List[InferenceResult]:
        """Process pending batched requests."""
        return self.processor.process_batch()

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return self.processor.get_stats()
