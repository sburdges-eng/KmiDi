"""
Shared helpers for MCP tools.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Tuple, TypeVar

from music_brain.api import DAiWAPI
from music_brain.session.intent_schema import CompleteSongIntent

JsonDict = Dict[str, Any]
T = TypeVar("T")

api = DAiWAPI()


async def run_sync(func, *args, **kwargs) -> Any:
    """
    Execute a blocking function in a worker thread and await the result.
    """

    return await asyncio.to_thread(func, *args, **kwargs)


def file_to_base64(path: str) -> str:
    """Return a base64-encoded string for the specified file."""

    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


@contextmanager
def midi_file_context(
    midi_path: Optional[str] = None,
    midi_base64: Optional[str] = None,
    suffix: str = ".mid",
) -> Generator[str, None, None]:
    """
    Provide a temporary MIDI file path regardless of whether the caller supplied
    a path or base64-encoded content.
    """

    if midi_path:
        yield midi_path
        return

    if not midi_base64:
        raise ValueError("Provide either 'midi_path' or 'midi_base64'.")

    data = base64.b64decode(midi_base64)
    fd, temp_path = tempfile.mkstemp(prefix="daiw_mcp_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:  # pragma: no cover - best-effort cleanup
            pass


def make_midi_payload(path: str, filename: Optional[str] = None) -> JsonDict:
    """
    Convert a MIDI file into a payload suitable for MCP responses.
    """

    return {
        "filename": filename or os.path.basename(path),
        "midi_base64": file_to_base64(path),
    }


def parse_intent_json(intent_json: str) -> CompleteSongIntent:
    """
    Deserialize a CompleteSongIntent from a JSON string.
    """

    data = json.loads(intent_json)
    return CompleteSongIntent.from_dict(data)


__all__ = [
    "api",
    "run_sync",
    "midi_file_context",
    "make_midi_payload",
    "parse_intent_json",
]

