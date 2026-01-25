"""
iDAW Collaboration Module.

Real-time multi-user session sharing for collaborative music production.
"""

from .collab_ui import (
    CollaborationClient,
    CollaboratorInfo,
    render_collaboration_ui,
)
from .intent_versioning import (
    Branch,
    Change,
    ChangeType,
    Commit,
    IntentVersionControl,
    Tag,
)
from .websocket_server import (
    CollaborationServer,
    MessageType,
    Participant,
    Session,
)

__all__ = [
    # WebSocket server
    "CollaborationServer",
    "Session",
    "Participant",
    "MessageType",
    # Intent versioning
    "IntentVersionControl",
    "Commit",
    "Branch",
    "Tag",
    "Change",
    "ChangeType",
    # Collaboration UI
    "CollaborationClient",
    "CollaboratorInfo",
    "render_collaboration_ui",
]
