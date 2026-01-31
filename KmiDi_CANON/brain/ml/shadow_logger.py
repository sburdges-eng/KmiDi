"""
ShadowLogger — JSONL logging for shadow-mode side-by-side metrics.

Writes to logs/shadow/ by default. Override via KMI_DI_SHADOW_LOG_DIR.
Per DATA_AND_TRAINING: for production, prefer ~/Models/logs/shadow.
"""

import json
import os
import time
from typing import Any, Dict, Protocol


class _InferResult(Protocol):
    """Protocol for inference results (payload, status)."""
    payload: Dict[str, Any]
    status: str


class ShadowLogger:
    """
    Logs primary vs shadow inference for offline analysis.

    One JSONL file per capability. Rows: ts, inputs, primary, shadow, statuses.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or os.environ.get("KMI_DI_SHADOW_LOG_DIR", "logs/shadow")
        os.makedirs(self.base_dir, exist_ok=True)

    def log(
        self,
        capability: str,
        inputs: Dict[str, Any],
        primary: _InferResult,
        shadow: _InferResult,
    ) -> None:
        """Append one row to capability-specific JSONL."""
        ts = int(time.time())
        path = os.path.join(self.base_dir, f"{capability}.jsonl")

        # Sanitize inputs for JSON (e.g. skip large tensors)
        inputs_safe = self._sanitize(inputs)
        primary_safe = self._sanitize(primary.payload) if primary.payload else {}
        shadow_safe = self._sanitize(shadow.payload) if shadow.payload else {}

        row = {
            "ts": ts,
            "inputs": inputs_safe,
            "primary": primary_safe,
            "shadow": shadow_safe,
            "primary_status": primary.status,
            "shadow_status": shadow.status,
        }

        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _sanitize(self, obj: Any) -> Any:
        """Make object JSON-serializable; skip non-serializable or large values."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return f"<{type(obj).__name__}>"
