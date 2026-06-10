import logging
from functools import wraps
from typing import Any, Callable, Optional

try:
    from fastapi import HTTPException
except ImportError:
    # Fallback if fastapi is missing
    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)


def api_error_handler(
    status_code: int = 500,
    detail: str = "Internal server error.",
    log_message: Optional[str] = None,
) -> Callable:
    """
    Decorator to abstract away repetitive exception handling boilerplate in API routes.
    Catches all non-HTTPExceptions, logs them, and raises a 500 HTTPException.
    """

    def decorator(func: Callable) -> Callable:
        # Check if the wrapped function is async
        import inspect

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    raise
                except Exception as e:
                    msg = log_message or f"{func.__name__} failed"
                    logging.exception(msg)
                    raise HTTPException(status_code=status_code, detail=detail) from e

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except HTTPException:
                    raise
                except Exception as e:
                    msg = log_message or f"{func.__name__} failed"
                    logging.exception(msg)
                    raise HTTPException(status_code=status_code, detail=detail) from e

            return sync_wrapper

    return decorator
