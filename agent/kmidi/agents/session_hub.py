"""Session hub agent: central DAW + voice + events (UnifiedHub-style)."""
from kmidi.tools.daw import DawGetStateTool, DawPlayTool, DawStopTool
from kmidi.tools.harmony import HarmonyTool
from kmidi.tools.intent import IntentTool
from kmidi.tools.session import IntentParseTool, SessionInterrogateTool

from ai_core.orchestrator import BaseAgentOrchestrator


class SessionHubAgent(BaseAgentOrchestrator):
    """Main KmiDi session agent: DAW control, intent, harmony, session state."""

    def __init__(self):
        tools = [
            DawPlayTool(),
            DawStopTool(),
            DawGetStateTool(),
            IntentParseTool(),
            SessionInterrogateTool(),
            IntentTool(),
            HarmonyTool(),
        ]
        system_prompt = """
        You are the KmiDi session agent. Control the DAW (play, stop, get state),
        parse user intent, interrogate session state, and use harmony when needed.
        """
        super().__init__(tools=tools, system_prompt=system_prompt, max_retries=2)
