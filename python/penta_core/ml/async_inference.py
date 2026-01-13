"""
Async Inference Engine - Async/await support for non-blocking inference.

Provides:
- Async inference methods
- Batch processing with configurable batch sizes
- Queue-based request handling
- Parallel inference across multiple models
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import numpy as np

try:
    from .model_pool import get_model_pool, ModelPool

    HAS_POOL = True
except ImportError:
    HAS_POOL = False

try:
    from .inference_enhanced import EnhancedInferenceEngine

    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

from penta_core.ml.model_registry import ModelInfo
from penta_core.ml.inference import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for inference."""

    request_id: str
    model_name: str
    inputs: Dict[str, np.ndarray]
    callback: Optional[Callable[[InferenceResult], None]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Response from inference."""

    request_id: str
    result: Optional[InferenceResult] = None
    error: Optional[Exception] = None
    latency_ms: float = 0.0


class AsyncInferenceEngine:
    """
    Async inference engine with queue-based request handling.

    Supports:
    - Non-blocking async inference
    - Batch processing
    - Parallel inference across models
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        max_workers: int = 4,
        batch_size: int = 1,
        batch_timeout_ms: float = 10.0,
    ):
        """
        Initialize async inference engine.

        Args:
            max_queue_size: Maximum queue size
            max_workers: Maximum number of worker tasks
            batch_size: Batch size for processing
            batch_timeout_ms: Maximum time to wait for batch in milliseconds
        """
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._model_pool: Optional[ModelPool] = None
        self._pending_futures: Dict[str, asyncio.Future] = {}  # Track futures by request_id

        if HAS_POOL:
            try:
                self._model_pool = get_model_pool()
            except Exception:
                pass

        # Statistics
        self._stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "batches_processed": 0,
            "average_latency_ms": 0.0,
        }

    async def start(self):
        """Start the async inference engine."""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(f"worker-{i}")) for i in range(self.max_workers)
        ]

        logger.info(f"Async inference engine started with {self.max_workers} workers")

    async def stop(self):
        """Stop the async inference engine."""
        if not self._running:
            return

        self._running = False

        # Wait for queue to drain
        await self._queue.join()

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Async inference engine stopped")

    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        """
        Submit inference request (async).

        Args:
            model_name: Model name
            inputs: Input tensors
            request_id: Optional request ID
            metadata: Optional metadata

        Returns:
            Inference response
        """
        if not self._running:
            raise RuntimeError("Async inference engine not running")

        if request_id is None:
            import uuid

            request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Create future to track result
        future = asyncio.Future()
        self._pending_futures[request_id] = future

        request = InferenceRequest(
            request_id=request_id,
            model_name=model_name,
            inputs=inputs,
            metadata=metadata or {},
        )

        try:
            await self._queue.put(request)
        except asyncio.QueueFull:
            # Clean up future
            self._pending_futures.pop(request_id, None)
            return InferenceResponse(
                request_id=request_id,
                error=RuntimeError("Inference queue is full"),
            )

        # Wait for result from worker
        try:
            response = await future
            return response
        except Exception as e:
            # Clean up future on error
            self._pending_futures.pop(request_id, None)
            return InferenceResponse(
                request_id=request_id,
                error=e,
            )

    async def infer_batch(
        self,
        model_name: str,
        inputs_list: List[Dict[str, np.ndarray]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[InferenceResponse]:
        """
        Submit batch inference requests.

        Args:
            model_name: Model name
            inputs_list: List of input tensors
            metadata: Optional metadata

        Returns:
            List of inference responses
        """
        if not HAS_INFERENCE:
            return [
                InferenceResponse(
                    request_id=f"batch_{i}",
                    error=RuntimeError("Inference engine not available"),
                )
                for i in range(len(inputs_list))
            ]

        # Get engine
        engine = None
        if self._model_pool:
            from penta_core.ml.model_registry import get_model

            model_info = get_model(model_name)
            if model_info:
                engine = self._model_pool.get(model_info, auto_load=True)

        if not engine:
            return [
                InferenceResponse(
                    request_id=f"batch_{i}",
                    error=RuntimeError(f"Model {model_name} not available"),
                )
                for i in range(len(inputs_list))
            ]

        # Process batch
        results = []
        start_time = time.perf_counter()

        try:
            # Run batch inference
            for i, inputs in enumerate(inputs_list):
                try:
                    result = engine.infer(inputs, use_fallback=False)
                    results.append(
                        InferenceResponse(
                            request_id=f"batch_{i}",
                            result=result,
                            latency_ms=result.latency_ms if result else 0.0,
                        )
                    )
                except Exception as e:
                    results.append(
                        InferenceResponse(
                            request_id=f"batch_{i}",
                            error=e,
                        )
                    )

            batch_latency = (time.perf_counter() - start_time) * 1000
            self._stats["batches_processed"] += 1
            self._stats["requests_processed"] += len(inputs_list)

            logger.debug(
                f"Processed batch of {len(inputs_list)} requests " f"in {batch_latency:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Return error responses for all
            results = [
                InferenceResponse(
                    request_id=f"batch_{i}",
                    error=e,
                )
                for i in range(len(inputs_list))
            ]

        return results

    async def _worker(self, worker_name: str):
        """Worker task for processing inference requests."""
        logger.debug(f"Worker {worker_name} started")

        batch: List[InferenceRequest] = []
        batch_deadline: Optional[float] = None

        while self._running:
            try:
                # Wait for request with timeout
                timeout = None
                if batch_deadline:
                    timeout = max(0.0, batch_deadline - time.time())

                try:
                    if timeout is not None and timeout > 0:
                        request = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    else:
                        request = await self._queue.get()
                except asyncio.TimeoutError:
                    # Timeout - process current batch
                    if batch:
                        await self._process_batch(batch)
                        batch.clear()
                        batch_deadline = None
                    continue

                batch.append(request)

                # Set deadline for first item in batch
                if batch_deadline is None:
                    batch_deadline = time.time() + (self.batch_timeout_ms / 1000.0)

                # Process batch if full or timeout
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch)
                    batch.clear()
                    batch_deadline = None
                elif batch_deadline and time.time() >= batch_deadline:
                    await self._process_batch(batch)
                    batch.clear()
                    batch_deadline = None

            except asyncio.CancelledError:
                # Process remaining batch before exit
                if batch:
                    await self._process_batch(batch)
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")

        logger.debug(f"Worker {worker_name} stopped")

    async def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests."""
        if not batch:
            return

        # Group by model
        by_model: Dict[str, List[InferenceRequest]] = {}
        for request in batch:
            if request.model_name not in by_model:
                by_model[request.model_name] = []
            by_model[request.model_name].append(request)

        # Process each model's requests
        for model_name, requests in by_model.items():
            await self._process_model_batch(model_name, requests)

    async def _process_model_batch(self, model_name: str, requests: List[InferenceRequest]):
        """Process batch of requests for a single model."""
        if not HAS_INFERENCE:
            error = RuntimeError("Inference engine not available")
            for request in requests:
                response = InferenceResponse(
                    request_id=request.request_id,
                    error=error,
                )
                future = self._pending_futures.pop(request.request_id, None)
                if future and not future.done():
                    future.set_result(response)

                if request.callback:
                    try:
                        request.callback(None)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                self._queue.task_done()
            return

        # Get engine
        engine = None
        if self._model_pool:
            from penta_core.ml.model_registry import get_model

            model_info = get_model(model_name)
            if model_info:
                engine = self._model_pool.get(model_info, auto_load=True)

        if not engine:
            error = RuntimeError(f"Model {model_name} not available")
            for request in requests:
                response = InferenceResponse(
                    request_id=request.request_id,
                    error=error,
                )
                future = self._pending_futures.pop(request.request_id, None)
                if future and not future.done():
                    future.set_result(response)

                if request.callback:
                    try:
                        request.callback(None)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                self._queue.task_done()
            return

        # Process requests
        for request in requests:
            start_time = time.perf_counter()
            response: Optional[InferenceResponse] = None

            try:
                # Run inference in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: engine.infer(request.inputs, use_fallback=False)
                )

                latency_ms = (time.perf_counter() - start_time) * 1000
                response = InferenceResponse(
                    request_id=request.request_id,
                    result=result,
                    latency_ms=latency_ms,
                )

                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                self._stats["requests_processed"] += 1

            except Exception as e:
                logger.error(f"Inference failed for {request.request_id}: {e}")
                self._stats["requests_failed"] += 1
                latency_ms = (time.perf_counter() - start_time) * 1000
                response = InferenceResponse(
                    request_id=request.request_id,
                    error=e,
                    latency_ms=latency_ms,
                )

                if request.callback:
                    try:
                        request.callback(None)
                    except Exception as callback_error:
                        logger.error(f"Callback error: {callback_error}")

            finally:
                # Set future result if pending
                future = self._pending_futures.pop(request.request_id, None)
                if future and not future.done():
                    if response:
                        future.set_result(response)
                    else:
                        # Fallback error response
                        future.set_result(
                            InferenceResponse(
                                request_id=request.request_id,
                                error=RuntimeError("Inference processing failed"),
                            )
                        )

                self._queue.task_done()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        queue_size = self._queue.qsize()
        return {
            "queue_size": queue_size,
            "max_queue_size": self.max_queue_size,
            "workers": len(self._workers),
            "running": self._running,
            "requests_processed": self._stats["requests_processed"],
            "requests_failed": self._stats["requests_failed"],
            "batches_processed": self._stats["batches_processed"],
            "average_latency_ms": self._stats["average_latency_ms"],
        }


# Singleton async engine
_async_engine: Optional[AsyncInferenceEngine] = None
_async_engine_lock = threading.Lock()


def get_async_inference_engine() -> AsyncInferenceEngine:
    """Get the global async inference engine singleton."""
    global _async_engine
    if _async_engine is None:
        with _async_engine_lock:
            if _async_engine is None:
                _async_engine = AsyncInferenceEngine()
    return _async_engine
