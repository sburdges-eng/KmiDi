"""Intent tools for parsing user music production intent."""
from typing import Any, Dict

from ai_core.interfaces import AgentTool


class IntentTool(AgentTool):
    name = "intent"
    description = "Parse or resolve user intent (e.g. 'make it darker', 'add a fill')."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text") or payload.get("query", "")
        return {"status": "success", "intent": {}, "text": text[:200]}
