"""Session and intent parsing tools."""
from typing import Any, Dict

from ai_core.interfaces import AgentTool


class IntentParseTool(AgentTool):
    name = "intent_parse"
    description = "Parse user utterance into structured intent (goal, parameters)."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        utterance = payload.get("utterance") or payload.get("text", "")
        return {"status": "success", "intent": {}, "utterance": utterance[:200]}


class SessionInterrogateTool(AgentTool):
    name = "session_interrogate"
    description = "Query current session state (tracks, selection, context)."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "state")
        return {"status": "success", "query": query, "session": {}}
