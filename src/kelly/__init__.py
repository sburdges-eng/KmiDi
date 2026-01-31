"""
Kelly Project - Therapeutic Music Generation Framework

Core package for the Kelly MIDI Companion system.
"""

__version__ = "0.1.0"

# Import integrations if available
try:
    from . import integrations
    __all__ = ['integrations']
except ImportError:
    __all__ = []
