"""Training ops agent: preflight, launch AWS, monitor, fetch artifacts."""
from typing import Any, Dict

from ai_core.interfaces import AgentTool
from ai_core.orchestrator import BaseAgentOrchestrator


class LaunchAwsTrainTool(AgentTool):
    name = "launch_aws_train"
    description = "Launch AWS GPU training run (after dry-run and preflight)."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "dry_run": payload.get("dry_run", False), "run_id": None}


class FetchAwsArtifactsTool(AgentTool):
    name = "fetch_aws_artifacts"
    description = "Fetch artifacts from completed run (student or break-glass teacher)."

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "run_id": payload.get("run_id"), "output_dir": payload.get("output_dir")}


class TrainingOpsAgent(BaseAgentOrchestrator):
    """Training ops: preflight, launch, monitor, fetch artifacts."""

    def __init__(self):
        tools = [LaunchAwsTrainTool(), FetchAwsArtifactsTool()]
        system_prompt = """
        You are the Training Ops agent. Run preflight, then launch_aws_train (dry-run first),
        then fetch_aws_artifacts. Preserve runId and output paths.
        """
        super().__init__(tools=tools, system_prompt=system_prompt, max_retries=1)
