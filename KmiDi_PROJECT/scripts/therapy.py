"""
Therapy session MCP tool.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

from .common import api, make_midi_payload, run_sync

JsonDict = Dict[str, Any]


async def therapy_session_tool(
    text: str,
    motivation: int = 7,
    chaos_tolerance: float = 0.5,
    include_midi: bool = True,
) -> JsonDict:
    """
    Run the therapy session engine and optionally return a MIDI plan.
    """

    if include_midi:
        with tempfile.TemporaryDirectory(prefix="daiw_mcp_") as tmpdir:
            midi_path = os.path.join(tmpdir, "therapy.mid")
            result = await run_sync(
                api.therapy_session,
                text,
                motivation,
                chaos_tolerance,
                midi_path,
            )
            if "midi_path" in result:
                result["midi"] = make_midi_payload(result["midi_path"], filename="therapy_session.mid")
            return result

    return await run_sync(
        api.therapy_session,
        text,
        motivation,
        chaos_tolerance,
        None,
    )


def register_tools(server) -> None:
    """Register therapy session tool."""

    server.tool(
        name="daiw.therapy.session",
        description="Capture emotional context and return a harmonic/melodic plan.",
    )(therapy_session_tool)


__all__ = ["register_tools"]

