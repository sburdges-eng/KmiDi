import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from ai_core.interfaces import AgentTool


class AgentState(TypedDict):
    original_intent: str
    system_prompt: str
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    pending_tool_calls: List[Dict[str, Any]]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    error_count: int


class BaseAgentOrchestrator:
    def __init__(self, tools: List[AgentTool], system_prompt: str, max_retries: int = 3):
        self.tool_registry = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("reasoning_node", self._reasoning_node)
        workflow.add_node("tool_executor", self._tool_node)
        workflow.add_edge(START, "reasoning_node")
        workflow.add_conditional_edges(
            "reasoning_node",
            self._route_execution,
            {"execute_tools": "tool_executor", "human_fallback": END, "finish": END},
        )
        workflow.add_edge("tool_executor", "reasoning_node")
        return workflow.compile()

    async def _reasoning_node(self, state: AgentState) -> dict:
        print("[AI CORE] Reasoning Engine Active...")
        if state.get("error_count", 0) >= self.max_retries:
            print("[AI CORE] Max retries hit. Halting.")
            return {"pending_tool_calls": []}
        if not state.get("tool_results"):
            return {
                "pending_tool_calls": [
                    {
                        "name": list(self.tool_registry.keys())[0],
                        "payload": {},
                    }
                ]
            }
        return {"pending_tool_calls": []}

    async def _tool_node(self, state: AgentState) -> dict:
        results = []
        error_increment = 0
        for call in state["pending_tool_calls"]:
            tool_name = call.get("name")
            payload = call.get("payload", {})
            if tool_name in self.tool_registry:
                tool = self.tool_registry[tool_name]
                print(f"[AI CORE] Executing Tool: {tool_name}")
                result = await tool.execute(payload)
                results.append({"tool": tool_name, "result": result})
                if result.get("status") == "error":
                    error_increment += 1
            else:
                results.append(
                    {
                        "tool": tool_name,
                        "result": {"status": "error", "message": "Tool not found."},
                    }
                )
                error_increment += 1
        return {
            "pending_tool_calls": [],
            "tool_results": results,
            "error_count": state.get("error_count", 0) + error_increment,
        }

    def _route_execution(self, state: AgentState) -> str:
        if state.get("error_count", 0) >= self.max_retries:
            return "human_fallback"
        if len(state.get("pending_tool_calls", [])) > 0:
            return "execute_tools"
        return "finish"

    async def process_intent(self, user_intent: str) -> dict:
        initial_state = {
            "original_intent": user_intent,
            "system_prompt": self.system_prompt,
            "chat_history": [],
            "pending_tool_calls": [],
            "tool_results": [],
            "error_count": 0,
        }
        return await self.graph.ainvoke(initial_state)
