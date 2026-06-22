"""Dataset packager agent: build → split → package → validate → sync."""
from kmidi.tools.dataset_packaging import (
    BuildTrainingRecordsTool,
    PackageDatasetTool,
    SplitDatasetTool,
    SyncPackageToS3Tool,
    ValidatePackageTool,
)

from ai_core.orchestrator import BaseAgentOrchestrator


class DatasetPackagerAgent(BaseAgentOrchestrator):
    """Deterministic dataset packaging from transcript JSONL through S3 sync."""

    def __init__(self):
        tools = [
            BuildTrainingRecordsTool(),
            SplitDatasetTool(),
            PackageDatasetTool(),
            ValidatePackageTool(),
            SyncPackageToS3Tool(),
        ]
        system_prompt = """
        You are the Dataset Packager agent. Build records from transcript JSONL,
        split deterministically, package, validate, then sync to S3. Use dry-run before write/sync.
        """
        super().__init__(tools=tools, system_prompt=system_prompt, max_retries=2)
