"""Training runbook agent: transcript → build/split/package/validate/sync → launch → fetch."""
from kmidi.tools.dataset_packaging import (
    BuildTrainingRecordsTool,
    PackageDatasetTool,
    SplitDatasetTool,
    SyncPackageToS3Tool,
    ValidatePackageTool,
)

from ai_core.orchestrator import BaseAgentOrchestrator


class TrainingRunbookAgent(BaseAgentOrchestrator):
    """End-to-end training runbook: dataset packaging phases then launch/fetch (gate checks)."""

    def __init__(self):
        tools = [
            BuildTrainingRecordsTool(),
            SplitDatasetTool(),
            PackageDatasetTool(),
            ValidatePackageTool(),
            SyncPackageToS3Tool(),
        ]
        system_prompt = """
        You are the Training Runbook agent. Run phases in order: build records, split,
        package, validate, sync. Do not skip gates; stop on failure.
        """
        super().__init__(tools=tools, system_prompt=system_prompt, max_retries=1)
