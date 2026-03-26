"""
penta_core - Re-export facade for music_brain/penta_core/

This package is a thin shim that makes ``from penta_core.harmony import X``
(and similar) work when the repo root is on sys.path.  The single source of
truth is ``music_brain/penta_core/``; nothing is duplicated here.
"""

import importlib.util
import sys
import os as _os

# ---------------------------------------------------------------------------
# Ensure the canonical package location is on sys.path so that the real
# penta_core package can be loaded under a temporary alias.
# ---------------------------------------------------------------------------
_MUSIC_BRAIN = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "music_brain",
)
_CANONICAL_DIR = _os.path.join(_MUSIC_BRAIN, "penta_core")

# Note: this mutates sys.path for the lifetime of the process. Intentional:
# the facade needs music_brain on the path so the canonical package's
# relative imports resolve correctly.
if _MUSIC_BRAIN not in sys.path:
    sys.path.insert(0, _MUSIC_BRAIN)

# ---------------------------------------------------------------------------
# Load the real package under a private alias so that re-importing this
# facade does not recurse into itself.  If it is already present (e.g. the
# canonical path was already on sys.path before this facade was imported),
# just grab the cached module.
# ---------------------------------------------------------------------------
_REAL_PKG_NAME = "_penta_core_impl"

if _REAL_PKG_NAME not in sys.modules:
    # Load penta_core from music_brain/ and register it under the alias.
    _spec = importlib.util.spec_from_file_location(
        _REAL_PKG_NAME,
        _os.path.join(_CANONICAL_DIR, "__init__.py"),
        submodule_search_locations=[_CANONICAL_DIR],
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_REAL_PKG_NAME] = _mod
    _spec.loader.exec_module(_mod)

_impl = sys.modules[_REAL_PKG_NAME]

# ---------------------------------------------------------------------------
# Helper: load a subpackage from the canonical tree under the private alias
# and register it in sys.modules as penta_core.<name>.
# ---------------------------------------------------------------------------


def _load_subpackage(name):
    """Load music_brain/penta_core/<name>/ under _penta_core_impl.<name> and
    register it as penta_core.<name>.  No-op if already loaded.  Returns the
    module or None on failure."""
    real_name = f"{_REAL_PKG_NAME}.{name}"
    alias = f"penta_core.{name}"

    if alias in sys.modules:
        return sys.modules[alias]

    subpkg_path = _os.path.join(_CANONICAL_DIR, name)
    init_path = _os.path.join(subpkg_path, "__init__.py")

    if not _os.path.isfile(init_path):
        return None

    sub_spec = importlib.util.spec_from_file_location(
        real_name,
        init_path,
        submodule_search_locations=[subpkg_path],
    )
    sub_mod = importlib.util.module_from_spec(sub_spec)
    sub_mod.__package__ = real_name
    sys.modules[real_name] = sub_mod
    try:
        sub_spec.loader.exec_module(sub_mod)
    except Exception:
        del sys.modules[real_name]
        return None

    sys.modules[alias] = sub_mod
    # Also set as an attribute on this facade package so that
    # ``import penta_core.harmony`` (attribute-access style) works.
    globals()[name] = sub_mod
    return sub_mod


# ---------------------------------------------------------------------------
# Register submodules under the penta_core.* namespace so that
# ``from penta_core.harmony import counterpoint`` (and similar) works.
#
# Only subpackages that are known to load without optional third-party
# dependencies are included here.  Excluded:
#   - groove_penta: imports from penta_core.groove.* which does not exist in
#     the canonical tree (pre-existing bug; the directory is groove_penta/).
#   - server: requires 'dotenv' and 'mcp'/'fastmcp' at import time and has
#     top-level side effects (FastMCP initialisation).
# ---------------------------------------------------------------------------
_SUBMODULES = [
    "harmony",
    "collaboration",
    "teachers",
]

for _submod in _SUBMODULES:
    _load_subpackage(_submod)

# harmony is the only symbol in the canonical __init__.__all__.
__all__ = ["harmony"]
