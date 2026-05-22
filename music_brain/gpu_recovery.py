"""GPU error-recovery decorator for ML inference paths.

Closes the **GPU reset recovery flow** spec item documented as a gap in
``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``.

Wraps a function so that recognised GPU failures (CUDA OOM, MPS device
lost, generic ``RuntimeError`` carrying a GPU-related substring) trigger
a cleanup callback before either retrying or surfacing the error.

Design constraints:
- Stdlib + typing only. The decorator references torch/CUDA names by
  *string match*, not by importing torch, so it stays usable in
  environments where torch isn't installed (e.g. the headless API
  process). Callers that want type-precise matching can pass their own
  ``exception_predicate``.
- The decorator is synchronous. The cleanup callback runs inside the
  except-handler; it must be fast (typically: ``torch.cuda.empty_cache()``
  + ``gc.collect()``).
- ``max_retries`` defaults to 1 — the first attempt plus one recovery
  attempt is usually enough to recover from a transient OOM. Setting it
  to 0 turns the decorator into a "log+cleanup then re-raise" wrapper.

Public API:
    @gpu_recoverable(cleanup=lambda: torch.cuda.empty_cache())
    def infer(batch): ...

    # or with bespoke matching:
    @gpu_recoverable(
        cleanup=cleanup,
        exception_predicate=lambda exc: isinstance(exc, MyGPUError),
        max_retries=2,
    )
    def infer(batch): ...
"""

from __future__ import annotations

import functools
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Substrings that indicate a GPU/accelerator failure recoverable by cleanup.
# Kept as strings (not isinstance checks against torch.cuda.OutOfMemoryError)
# so this module loads without torch installed.
_GPU_FAILURE_HINTS: tuple[str, ...] = (
    "CUDA out of memory",
    "out of memory",  # generic OOM message body
    "device-side assert",
    "MPS backend out of memory",
    "Metal command buffer error",
    "device lost",
    "an illegal memory access",
    "CUBLAS_STATUS",
)


def looks_like_gpu_failure(exc: BaseException) -> bool:
    """Heuristic: does ``exc`` look like a recoverable GPU failure?

    Matches against the exception message; case-insensitive. Used as the
    default ``exception_predicate`` for ``gpu_recoverable``.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint.lower() in text for hint in _GPU_FAILURE_HINTS)


def gpu_recoverable(
    *,
    cleanup: Optional[Callable[[], None]] = None,
    exception_predicate: Optional[Callable[[BaseException], bool]] = None,
    max_retries: int = 1,
    on_recover: Optional[Callable[[BaseException, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory: wrap ``fn`` with GPU-error recovery.

    Args:
        cleanup: Called from inside the except handler before retrying.
            Typical: ``lambda: (torch.cuda.empty_cache(), gc.collect())``.
            Pass None to skip cleanup (the retry alone may suffice for
            transient errors).
        exception_predicate: Decide whether an exception is recoverable.
            Defaults to ``looks_like_gpu_failure``.
        max_retries: Number of additional attempts after a recoverable
            failure. 0 means "log+cleanup then re-raise on first
            failure" — useful when the caller wants observability
            without auto-retry.
        on_recover: Diagnostic hook ``(exc, attempt_index)`` called
            before each retry. Use for telemetry / structured logging.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    predicate = exception_predicate or looks_like_gpu_failure

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            attempts = 0
            last_exc: Optional[BaseException] = None
            while True:
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 — we re-raise non-matching
                    if not predicate(exc):
                        raise
                    last_exc = exc
                    if cleanup is not None:
                        try:
                            cleanup()
                        except Exception:  # noqa: BLE001 — cleanup must not mask original
                            pass
                    if attempts >= max_retries:
                        raise
                    attempts += 1
                    if on_recover is not None:
                        try:
                            on_recover(exc, attempts)
                        except Exception:  # noqa: BLE001 — telemetry is best-effort
                            pass
            # Unreachable; raise to satisfy type-checkers if loop is bypassed.
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


__all__ = ["gpu_recoverable", "looks_like_gpu_failure"]
