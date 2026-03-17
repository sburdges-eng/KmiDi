"""
Schema boundary: Request only at API boundary; CompleteSongIntent required internally.

No function past conversion should accept GenerateRequest. process_song_intent
and process_intent accept CompleteSongIntent only.
"""

import inspect

import pytest

from music_brain.session.intent_schema import CompleteSongIntent


def test_process_song_intent_accepts_complete_intent_only():
    """DAiWAPI.process_song_intent must take CompleteSongIntent, not Request."""
    from music_brain.api import DAiWAPI

    sig = inspect.signature(DAiWAPI.process_song_intent)
    params = list(sig.parameters.values())
    assert len(params) >= 2  # self, intent
    intent_param = next(p for p in params if p.name == "intent")
    ann = intent_param.annotation
    if getattr(ann, "__origin__", None) is not None:
        args = getattr(ann, "__args__", ())
        assert CompleteSongIntent in args or ann == CompleteSongIntent
    else:
        assert ann == CompleteSongIntent


def test_process_intent_accepts_complete_intent_only():
    """session.intent_processor.process_intent must take CompleteSongIntent."""
    from music_brain.session.intent_processor import process_intent

    sig = inspect.signature(process_intent)
    params = list(sig.parameters.values())
    assert len(params) >= 1
    intent_param = params[0]
    assert intent_param.name == "intent"
    assert intent_param.annotation == CompleteSongIntent
