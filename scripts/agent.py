"""
Local chatbot agent bridging an offline LLM with DAiW APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from music_brain.api import DAiWAPI
from music_brain.chatbot import tools


@dataclass
class AgentConfig:
    model_path: Path
    system_prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    history_limit: int = 20
    persona: str = "DAiW offline companion"


class LLMRunner:
    """
    Placeholder LLM runner for offline chatbot functionality.
    
    This is a stub implementation that returns echo responses.
    To enable full offline LLM support, integrate with llama.cpp or GPT4All.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response from message history.
        
        Note: This is a placeholder implementation. For production use,
        integrate with an offline LLM library (llama.cpp, GPT4All, etc.).
        """
        last_user = messages[-1]["content"]
        return f"[offline placeholder] You said: {last_user[:100]}"


class ChatAgent:
    def __init__(self, config: AgentConfig, api: Optional[DAiWAPI] = None):
        self.config = config
        self.api = api or DAiWAPI()
        self.llm = LLMRunner(config)
        self.history: List[Dict[str, str]] = [{"role": "system", "content": config.system_prompt}]

    def chat(self, user_message: str) -> str:
        tool_response = self._maybe_run_tool(user_message)
        if tool_response:
            assistant_reply = tool_response
        else:
            self.history.append({"role": "user", "content": user_message})
            assistant_reply = self.llm.generate(self.history[-self.config.history_limit :])
        self.history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    def _maybe_run_tool(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        if "auto-tune" in text_lower or "autotune" in text_lower:
            return tools.describe_auto_tune_usage()
        if "backing track" in text_lower:
            return tools.describe_backing_workflow()
        return None

