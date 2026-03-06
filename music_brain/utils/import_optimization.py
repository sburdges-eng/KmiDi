"""Import optimization utilities."""

from functools import lru_cache
from typing import Any, Callable, Optional

# Cache for expensive imports
_import_cache: dict = {}


@lru_cache(maxsize=32)
def cached_import(module_name: str, attribute: Optional[str] = None) -> Any:
    """
    Cached import function to avoid repeated expensive imports.

    Args:
        module_name: Name of the module to import
        attribute: Optional attribute to get from the module

    Returns:
        The imported module or attribute
    """
    if module_name in _import_cache:
        module = _import_cache[module_name]
    else:
        module = __import__(module_name, fromlist=[''])
        _import_cache[module_name] = module

    if attribute:
        return getattr(module, attribute)
    return module


def lazy_import(module_name: str, attribute: str) -> Callable:
    """
    Create a lazy import function for a specific attribute.

    Usage:
        BassEngine = lazy_import('music_brain.kelly_companion.engines', 'BassEngine')
        engine = BassEngine()  # Import happens here
    """
    def _lazy_getter():
        module = __import__(module_name, fromlist=[attribute])
        return getattr(module, attribute)

    return _lazy_getter
