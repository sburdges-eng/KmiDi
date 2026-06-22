from abc import ABC
from typing import Any, Dict


class AgentTool(ABC):
    name: str
    description: str

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
