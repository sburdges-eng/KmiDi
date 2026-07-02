"""
Tests for the root-level penta_core re-export facade.

Verifies that ``from penta_core.harmony import X`` and
``from penta_core.<submodule> import X`` both work when the repo
root is on sys.path, and that the facade delegates to the canonical
implementation in music_brain/penta_core/.
"""

import sys
import os

import pytest

# conftest.py already ensures repo root is on sys.path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _fresh_root_facade():
    """Import the facade from a clean slate, immune to suite ordering.

    Other tests (e.g. test_apsc_wrapper) put music_brain/ at sys.path[0],
    which makes a bare ``import penta_core`` resolve to the canonical
    package directly and bypass the repo-root facade. That is fine at
    runtime, but these tests assert facade mechanics — so purge cached
    modules, force repo root to the front of sys.path, and restore
    everything afterwards.
    """

    def _facade_keys():
        return [
            k
            for k in sys.modules
            if k == "penta_core"
            or k.startswith("penta_core.")
            or k == "_penta_core_impl"
            or k.startswith("_penta_core_impl.")
        ]

    saved_modules = {k: sys.modules[k] for k in _facade_keys()}
    saved_path = list(sys.path)
    for k in _facade_keys():
        del sys.modules[k]
    if REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    try:
        yield
    finally:
        for k in _facade_keys():
            del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


# ---------------------------------------------------------------------------
# Top-level facade import
# ---------------------------------------------------------------------------


class TestFacadeImport:
    """The root-level penta_core package must be importable and point to the facade."""

    def test_package_importable(self):
        import penta_core

        assert penta_core is not None

    def test_facade_file_is_at_repo_root(self):
        import penta_core

        # penta_core/__init__.py lives at repo_root/penta_core/__init__.py
        # so dirname(dirname(__file__)) should equal repo root.
        facade_pkg_dir = os.path.dirname(os.path.abspath(penta_core.__file__))
        facade_parent = os.path.dirname(facade_pkg_dir)
        assert facade_parent == REPO_ROOT, (
            f"penta_core package must be a direct child of repo root, "
            f"got: {penta_core.__file__}"
        )

    def test_all_contains_harmony(self):
        import penta_core

        assert "harmony" in penta_core.__all__

    def test_double_import_is_idempotent(self):
        import penta_core
        import penta_core as pc2  # noqa: F401

        assert penta_core is pc2, "Importing penta_core twice must return the same object"


# ---------------------------------------------------------------------------
# Harmony subpackage imports
# ---------------------------------------------------------------------------


class TestHarmonySubpackage:
    """``from penta_core.harmony import X`` must work for all harmony submodules."""

    def test_harmony_subpackage_importable(self):
        import penta_core.harmony as h

        assert h is not None

    def test_counterpoint_submodule(self):
        from penta_core.harmony import counterpoint

        assert counterpoint is not None

    def test_jazz_voicings_submodule(self):
        from penta_core.harmony import jazz_voicings

        assert jazz_voicings is not None

    def test_neo_riemannian_submodule(self):
        from penta_core.harmony import neo_riemannian

        assert neo_riemannian is not None

    def test_microtonal_submodule(self):
        from penta_core.harmony import microtonal

        assert microtonal is not None

    def test_tension_submodule(self):
        from penta_core.harmony import tension

        assert tension is not None

    def test_harmony_classes_from_counterpoint(self):
        from penta_core.harmony.counterpoint import CounterpointVoice, Species, Motion

        assert issubclass(CounterpointVoice, object)
        assert issubclass(Species, object)
        assert issubclass(Motion, object)

    def test_harmony_classes_from_neo_riemannian(self):
        from penta_core.harmony.neo_riemannian import NeoRiemannianTransform

        assert NeoRiemannianTransform is not None

    def test_harmony_classes_from_microtonal(self):
        from penta_core.harmony.microtonal import TuningSystem, MicrotonalPitch

        assert TuningSystem is not None
        assert MicrotonalPitch is not None


# ---------------------------------------------------------------------------
# Collaboration subpackage imports
# ---------------------------------------------------------------------------


class TestCollaborationSubpackage:
    """``from penta_core.collaboration import X`` must work."""

    def test_collaboration_importable(self):
        import penta_core.collaboration as col

        assert col is not None

    def test_websocket_server_classes(self):
        from penta_core.collaboration import (
            CollaborationServer,
            Session,
            Participant,
            MessageType,
        )

        assert CollaborationServer is not None
        assert Session is not None
        assert Participant is not None
        assert MessageType is not None

    def test_intent_versioning_classes(self):
        from penta_core.collaboration import (
            IntentVersionControl,
            ChangeType,
        )

        assert IntentVersionControl is not None
        assert ChangeType is not None

    def test_collab_ui_classes(self):
        from penta_core.collaboration import (
            render_collaboration_ui,
        )

        assert callable(render_collaboration_ui)


# ---------------------------------------------------------------------------
# Teachers subpackage imports
# ---------------------------------------------------------------------------


class TestTeachersSubpackage:
    """penta_core.teachers must be importable (package marker)."""

    def test_teachers_importable(self):
        import penta_core.teachers

        assert penta_core.teachers is not None

    def test_teachers_via_sys_modules(self):
        import penta_core.teachers  # noqa: F401

        assert "penta_core.teachers" in sys.modules


# ---------------------------------------------------------------------------
# Single-source-of-truth and identity checks
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    """The facade must delegate to the real implementation, not shadow it."""

    def test_impl_alias_in_sys_modules(self):
        """Facade must have loaded the real package as _penta_core_impl."""
        import penta_core  # noqa: F401 — ensure facade is loaded

        assert (
            "_penta_core_impl" in sys.modules
        ), "Facade must register _penta_core_impl in sys.modules"

    def test_impl_is_canonical_source(self):
        """_penta_core_impl must point to music_brain/penta_core/__init__.py."""
        import penta_core  # noqa: F401

        impl = sys.modules["_penta_core_impl"]
        expected = os.path.join(REPO_ROOT, "music_brain", "penta_core", "__init__.py")
        assert os.path.abspath(impl.__file__) == os.path.abspath(
            expected
        ), f"_penta_core_impl.__file__ should be canonical source; got {impl.__file__}"

    def test_canonical_file_untouched_on_disk(self):
        """music_brain/penta_core/__init__.py must still exist on disk."""
        canonical = os.path.join(REPO_ROOT, "music_brain", "penta_core", "__init__.py")
        assert os.path.isfile(canonical), "Canonical __init__.py must still exist on disk"

    def test_harmony_object_identity(self):
        """Importing penta_core.harmony twice must return the identical object."""
        import penta_core.harmony as h1
        import penta_core.harmony as h2

        assert h1 is h2, "penta_core.harmony must be the same object on re-import"

    def test_harmony_module_identity_vs_impl(self):
        """penta_core.harmony and _penta_core_impl.harmony must be the same object."""
        import penta_core  # noqa: F401
        import penta_core.harmony as facade_harmony  # noqa: F401

        impl_harmony = sys.modules.get("_penta_core_impl.harmony")
        assert impl_harmony is not None, "_penta_core_impl.harmony must be in sys.modules"
        assert (
            sys.modules["penta_core.harmony"] is impl_harmony
        ), "penta_core.harmony must be the same object as _penta_core_impl.harmony"

    def test_no_circular_import(self):
        """Importing penta_core multiple times must not raise."""
        import penta_core  # noqa: F401
        import penta_core  # noqa: F401,F811

        # If we reach here without exception, circular import protection works.

    def test_music_brain_on_sys_path(self):
        """Facade must have added music_brain/ to sys.path."""
        import penta_core  # noqa: F401

        music_brain = os.path.join(REPO_ROOT, "music_brain")
        assert (
            music_brain in sys.path
        ), "Facade must add music_brain/ to sys.path for canonical relative imports"
