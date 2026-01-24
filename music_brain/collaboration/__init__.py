"""
Collaboration Features - Real-time collaborative music creation.

Provides:
- WebSocket-based session sharing
- Version control for song intents
- Collaborative editing with conflict resolution
- Comment and annotation system
- User presence and awareness
"""

from music_brain.collaboration.comments import (
    Annotation,
    Comment,
    CommentThread,
    add_comment,
    resolve_thread,
)
from music_brain.collaboration.editing import (
    CollaborativeDocument,
    Operation,
    OperationType,
    apply_operation,
    transform_operations,
)
from music_brain.collaboration.session import (
    CollaborativeSession,
    SessionParticipant,
    SessionRole,
    create_session,
    join_session,
    leave_session,
)
from music_brain.collaboration.version_control import (
    IntentDiff,
    IntentVersion,
    IntentVersionControl,
    create_version,
    get_history,
    restore_version,
)
from music_brain.collaboration.websocket import (
    CollaborationClient,
    CollaborationServer,
    Message,
    MessageType,
)

__all__ = [
    # Session
    "CollaborativeSession",
    "SessionParticipant",
    "SessionRole",
    "create_session",
    "join_session",
    "leave_session",
    # WebSocket
    "CollaborationServer",
    "CollaborationClient",
    "Message",
    "MessageType",
    # Version Control
    "IntentVersionControl",
    "IntentVersion",
    "IntentDiff",
    "create_version",
    "get_history",
    "restore_version",
    # Editing
    "Operation",
    "OperationType",
    "CollaborativeDocument",
    "apply_operation",
    "transform_operations",
    # Comments
    "Comment",
    "Annotation",
    "CommentThread",
    "add_comment",
    "resolve_thread",
]
