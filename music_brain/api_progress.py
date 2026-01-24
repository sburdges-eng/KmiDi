from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket


@dataclass
class TaskState:
    task_id: str
    status: str = "pending"
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class ProgressTracker:
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskState] = {}
        self._sockets: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def create_task(self) -> str:
        task_id = uuid.uuid4().hex
        async with self._lock:
            self._tasks[task_id] = TaskState(task_id=task_id)
            self._sockets.setdefault(task_id, set())
        await self.publish(task_id, {"type": "task_started", "task_id": task_id, "progress": 0.0})
        return task_id

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._sockets.setdefault(task_id, set()).add(websocket)
        task = self._tasks.get(task_id)
        if task:
            await websocket.send_json({"type": "task_status", **task.to_dict()})

    async def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            sockets = self._sockets.get(task_id)
            if sockets and websocket in sockets:
                sockets.remove(websocket)

    async def update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = progress
            if stage is not None:
                task.stage = stage
            if message is not None:
                task.message = message
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            task.updated_at = time.time()

        await self.publish(task_id, {"type": "task_progress", **self._tasks[task_id].to_dict()})

    async def publish(self, task_id: str, event: Dict[str, Any]) -> None:
        async with self._lock:
            sockets = list(self._sockets.get(task_id, set()))

        for socket in sockets:
            try:
                await socket.send_json(event)
            except Exception:
                await self.disconnect(task_id, socket)

    def get_task(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def delete_task(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._sockets.pop(task_id, None)
