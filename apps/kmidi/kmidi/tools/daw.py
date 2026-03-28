"""DAW transport and state tools (play, stop, get state)."""
from typing import Any, Dict

from ai_core.interfaces import AgentTool


class DawPlayTool(AgentTool):
    name = "daw_play"
    description = "Start DAW playback."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "action": "play"}


class DawStopTool(AgentTool):
    name = "daw_stop"
    description = "Stop DAW playback."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "action": "stop"}


class DawGetStateTool(AgentTool):
    name = "daw_get_state"
    description = "Get current DAW state (transport, tempo, etc.)."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "state": {"playing": False, "tempo": 120}}
