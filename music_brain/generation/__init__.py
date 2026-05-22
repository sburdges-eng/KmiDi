"""Generation orchestration: scopes, isolation, rollback."""

from music_brain.generation.scope import GenerationScope, RollbackError

__all__ = ["GenerationScope", "RollbackError"]
