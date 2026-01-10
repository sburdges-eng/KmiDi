#!/usr/bin/env python3
"""
WebSocket Real-time API for UnifiedHub.

Provides a WebSocket server for real-time remote control and state streaming
from the React UI or external tools.

Features:
- Command/response RPC pattern
- Real-time state broadcasting
- Authentication support
- Automatic reconnection handling

Protocol:
    Client -> Server (commands):
        {"type": "command", "id": "uuid", "method": "play", "params": {}}
        {"type": "subscribe", "channels": ["voice_state", "daw_state"]}
        {"type": "unsubscribe", "channels": ["voice_state"]}

    Server -> Client (responses):
        {"type": "response", "id": "uuid", "result": {...}, "error": null}
        {"type": "state", "channel": "voice_state", "key": "vowel", "value": "O"}
        {"type": "broadcast", "event": "daw_connected", "data": true}

Usage:
    from music_brain.agents.websocket_api import HubWebSocketServer

    server = HubWebSocketServer(hub, port=8765)
    await server.start()  # Runs until stopped
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .unified_hub import UnifiedHub

# Optional: websockets library
try:
    import websockets
    from websockets.server import WebSocketServerProtocol

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Types
# =============================================================================


class MessageType(str, Enum):
    """WebSocket message types."""

    COMMAND = "command"
    RESPONSE = "response"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    STATE = "state"
    BROADCAST = "broadcast"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


@dataclass
class WSClient:
    """Represents a connected WebSocket client."""

    ws: WebSocketServerProtocol
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    authenticated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WSMessage:
    """Parsed WebSocket message."""

    type: MessageType
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    channels: Optional[List[str]] = None
    raw: Optional[str] = None

    @classmethod
    def parse(cls, data: str) -> "WSMessage":
        """Parse JSON message into WSMessage."""
        try:
            obj = json.loads(data)
            return cls(
                type=MessageType(obj.get("type", "command")),
                id=obj.get("id"),
                method=obj.get("method"),
                params=obj.get("params", {}),
                channels=obj.get("channels", []),
                raw=data,
            )
        except (json.JSONDecodeError, ValueError) as e:
            return cls(type=MessageType.ERROR, raw=data)


# =============================================================================
# WebSocket Server
# =============================================================================


class HubWebSocketServer:
    """
    WebSocket server for real-time UnifiedHub control.

    Exposes hub methods over WebSocket and broadcasts state changes.
    """

    # Default state channels
    STATE_CHANNELS = ["voice_state", "daw_state", "session", "health"]

    def __init__(
        self,
        hub: "UnifiedHub",
        host: str = "0.0.0.0",
        port: int = 8765,
        auth_token: Optional[str] = None,
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets library required. Install with: pip install websockets"
            )

        self.hub = hub
        self.host = host
        self.port = port
        self.auth_token = auth_token

        self._clients: Dict[str, WSClient] = {}
        self._server: Optional[websockets.WebSocketServer] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Command handlers
        self._handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

        # State subscription callbacks (stored for cleanup)
        self._state_unsubs: List[Callable] = []

    def _register_default_handlers(self):
        """Register default command handlers."""
        # DAW control
        self.register_handler("play", self._cmd_play)
        self.register_handler("stop", self._cmd_stop)
        self.register_handler("record", self._cmd_record)
        self.register_handler("set_tempo", self._cmd_set_tempo)
        self.register_handler("connect_daw", self._cmd_connect_daw)
        self.register_handler("disconnect_daw", self._cmd_disconnect_daw)

        # Voice control
        self.register_handler("speak", self._cmd_speak)
        self.register_handler("note_on", self._cmd_note_on)
        self.register_handler("note_off", self._cmd_note_off)
        self.register_handler("set_vowel", self._cmd_set_vowel)
        self.register_handler("set_breathiness", self._cmd_set_breathiness)
        self.register_handler("set_vibrato", self._cmd_set_vibrato)

        # Agent queries
        self.register_handler("ask_agent", self._cmd_ask_agent)
        self.register_handler("produce", self._cmd_produce)
        self.register_handler("suggest_progression", self._cmd_suggest_progression)

        # Session management
        self.register_handler("new_session", self._cmd_new_session)
        self.register_handler("save_session", self._cmd_save_session)
        self.register_handler("load_session", self._cmd_load_session)
        self.register_handler("list_sessions", self._cmd_list_sessions)

        # State queries
        self.register_handler("get_state", self._cmd_get_state)
        self.register_handler("get_health", self._cmd_get_health)

    def register_handler(self, method: str, handler: Callable):
        """Register a command handler."""
        self._handlers[method] = handler

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start the WebSocket server."""
        self._running = True
        self._loop = asyncio.get_running_loop()

        # Subscribe to hub state changes for broadcasting
        self._setup_state_subscriptions()

        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
        ) as server:
            self._server = server
            await asyncio.Future()  # Run forever

    def start_background(self) -> threading.Thread:
        """Start the server in a background thread."""

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self.start())
            except Exception as e:
                logger.error(f"WebSocket server error: {e}")
            finally:
                loop.close()

        thread = threading.Thread(target=run, daemon=True, name="ws-server")
        thread.start()
        return thread

    async def stop(self):
        """Stop the WebSocket server."""
        self._running = False

        # Cleanup state subscriptions
        for unsub in self._state_unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._state_unsubs.clear()

        # Close all client connections
        for client in list(self._clients.values()):
            try:
                await client.ws.close(1001, "Server shutting down")
            except Exception:
                pass

        self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("WebSocket server stopped")

    def _setup_state_subscriptions(self):
        """Subscribe to hub reactive states for broadcasting."""
        # Voice state
        if hasattr(self.hub, "_voice_state_reactive"):
            unsub = self.hub._voice_state_reactive.subscribe(
                lambda k, o, n: self._schedule_broadcast("voice_state", k, n)
            )
            self._state_unsubs.append(unsub)

        # DAW state
        if hasattr(self.hub, "_daw_state_reactive"):
            unsub = self.hub._daw_state_reactive.subscribe(
                lambda k, o, n: self._schedule_broadcast("daw_state", k, n)
            )
            self._state_unsubs.append(unsub)

        # Session state
        if hasattr(self.hub, "_session_reactive"):
            unsub = self.hub._session_reactive.subscribe(
                lambda k, o, n: self._schedule_broadcast("session", k, n)
            )
            self._state_unsubs.append(unsub)

    def _schedule_broadcast(self, channel: str, key: str, value: Any):
        """Schedule a state broadcast on the event loop."""
        if self._loop and self._running:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._broadcast_state(channel, key, value)
                )
            )

    # =========================================================================
    # Client Handling
    # =========================================================================

    async def _handle_client(self, ws: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket client connection."""
        client = WSClient(ws=ws)
        self._clients[client.id] = client

        logger.info(f"Client connected: {client.id} from {ws.remote_address}")

        # Send welcome message
        await self._send(
            client,
            {
                "type": "broadcast",
                "event": "connected",
                "data": {
                    "client_id": client.id,
                    "channels": self.STATE_CHANNELS,
                },
            },
        )

        try:
            async for message in ws:
                await self._handle_message(client, message)
        except websockets.ConnectionClosed as e:
            logger.info(f"Client {client.id} disconnected: {e.code}")
        except Exception as e:
            logger.error(f"Client {client.id} error: {e}")
        finally:
            self._clients.pop(client.id, None)

    async def _handle_message(self, client: WSClient, raw: str):
        """Handle an incoming message from a client."""
        msg = WSMessage.parse(raw)

        if msg.type == MessageType.ERROR:
            await self._send_error(client, None, "Invalid message format")
            return

        if msg.type == MessageType.PING:
            await self._send(client, {"type": "pong"})
            return

        if msg.type == MessageType.SUBSCRIBE:
            await self._handle_subscribe(client, msg)
            return

        if msg.type == MessageType.UNSUBSCRIBE:
            await self._handle_unsubscribe(client, msg)
            return

        if msg.type == MessageType.COMMAND:
            await self._handle_command(client, msg)
            return

    async def _handle_subscribe(self, client: WSClient, msg: WSMessage):
        """Handle subscription request."""
        channels = msg.channels or []
        valid_channels = [c for c in channels if c in self.STATE_CHANNELS]
        client.subscriptions.update(valid_channels)

        await self._send(
            client,
            {
                "type": "response",
                "id": msg.id,
                "result": {"subscribed": list(client.subscriptions)},
            },
        )

        # Send current state for subscribed channels
        for channel in valid_channels:
            state = self._get_channel_state(channel)
            if state:
                await self._send(
                    client,
                    {
                        "type": "state",
                        "channel": channel,
                        "key": "_initial",
                        "value": state,
                    },
                )

    async def _handle_unsubscribe(self, client: WSClient, msg: WSMessage):
        """Handle unsubscription request."""
        channels = msg.channels or []
        for channel in channels:
            client.subscriptions.discard(channel)

        await self._send(
            client,
            {
                "type": "response",
                "id": msg.id,
                "result": {"subscribed": list(client.subscriptions)},
            },
        )

    async def _handle_command(self, client: WSClient, msg: WSMessage):
        """Handle a command message."""
        method = msg.method
        if not method:
            await self._send_error(client, msg.id, "Missing method")
            return

        handler = self._handlers.get(method)
        if not handler:
            await self._send_error(client, msg.id, f"Unknown method: {method}")
            return

        try:
            # Run handler (may be sync or async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(msg.params or {})
            else:
                # Run sync handlers in executor to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: handler(msg.params or {})
                )

            await self._send(
                client,
                {
                    "type": "response",
                    "id": msg.id,
                    "result": result,
                    "error": None,
                },
            )
        except Exception as e:
            logger.exception(f"Handler error for {method}")
            await self._send_error(client, msg.id, str(e))

    # =========================================================================
    # Broadcasting
    # =========================================================================

    async def _broadcast_state(self, channel: str, key: str, value: Any):
        """Broadcast a state change to subscribed clients."""
        message = {
            "type": "state",
            "channel": channel,
            "key": key,
            "value": self._serialize(value),
        }

        for client in list(self._clients.values()):
            if channel in client.subscriptions:
                await self._send(client, message)

    async def broadcast_event(self, event: str, data: Any = None):
        """Broadcast an event to all connected clients."""
        message = {
            "type": "broadcast",
            "event": event,
            "data": self._serialize(data),
        }

        for client in list(self._clients.values()):
            await self._send(client, message)

    async def _send(self, client: WSClient, message: dict):
        """Send a message to a client."""
        try:
            await client.ws.send(json.dumps(message, default=str))
        except websockets.ConnectionClosed:
            self._clients.pop(client.id, None)
        except Exception as e:
            logger.error(f"Send error to {client.id}: {e}")

    async def _send_error(self, client: WSClient, msg_id: Optional[str], error: str):
        """Send an error response."""
        await self._send(
            client,
            {
                "type": "response",
                "id": msg_id,
                "result": None,
                "error": error,
            },
        )

    def _serialize(self, value: Any) -> Any:
        """Serialize a value for JSON."""
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        return value

    def _get_channel_state(self, channel: str) -> Optional[Dict[str, Any]]:
        """Get current state for a channel."""
        if channel == "voice_state":
            if hasattr(self.hub, "_voice_state_reactive"):
                return self.hub._voice_state_reactive.to_dict()
            return self._serialize(self.hub.voice_state)

        if channel == "daw_state":
            if hasattr(self.hub, "_daw_state_reactive"):
                return self.hub._daw_state_reactive.to_dict()
            return self._serialize(self.hub.daw_state)

        if channel == "session":
            if hasattr(self.hub, "_session_reactive"):
                return self.hub._session_reactive.to_dict()
            return self._serialize(self.hub.session)

        if channel == "health":
            return self.hub.check_llm_health()

        return None

    # =========================================================================
    # Command Handlers
    # =========================================================================

    def _cmd_play(self, params: dict) -> dict:
        self.hub.play()
        return {"success": True}

    def _cmd_stop(self, params: dict) -> dict:
        self.hub.stop_playback()
        return {"success": True}

    def _cmd_record(self, params: dict) -> dict:
        self.hub.record()
        return {"success": True}

    def _cmd_set_tempo(self, params: dict) -> dict:
        bpm = params.get("bpm", 120)
        self.hub.set_tempo(bpm)
        return {"success": True, "tempo": bpm}

    def _cmd_connect_daw(self, params: dict) -> dict:
        success = self.hub.connect_daw()
        return {"success": success, "connected": self.hub.daw_connected}

    def _cmd_disconnect_daw(self, params: dict) -> dict:
        self.hub.disconnect_daw()
        return {"success": True}

    def _cmd_speak(self, params: dict) -> dict:
        text = params.get("text", "")
        vowel = params.get("vowel")
        rate = params.get("rate", 175)
        self.hub.speak(text, vowel, rate)
        return {"success": True}

    def _cmd_note_on(self, params: dict) -> dict:
        pitch = params.get("pitch", 60)
        velocity = params.get("velocity", 100)
        channel = params.get("channel")
        self.hub.note_on(pitch, velocity, channel)
        return {"success": True}

    def _cmd_note_off(self, params: dict) -> dict:
        pitch = params.get("pitch")
        channel = params.get("channel")
        self.hub.note_off(pitch, channel)
        return {"success": True}

    def _cmd_set_vowel(self, params: dict) -> dict:
        vowel = params.get("vowel", "A")
        channel = params.get("channel")
        self.hub.set_vowel(vowel, channel)
        return {"success": True, "vowel": vowel}

    def _cmd_set_breathiness(self, params: dict) -> dict:
        amount = params.get("amount", 0.0)
        channel = params.get("channel")
        self.hub.set_breathiness(amount, channel)
        return {"success": True}

    def _cmd_set_vibrato(self, params: dict) -> dict:
        rate = params.get("rate", 0.0)
        depth = params.get("depth", 0.0)
        channel = params.get("channel")
        self.hub.set_vibrato(rate, depth, channel)
        return {"success": True}

    def _cmd_ask_agent(self, params: dict) -> dict:
        role = params.get("role", "composer")
        task = params.get("task", "")
        response = self.hub.ask_agent(role, task)
        return {"response": response}

    def _cmd_produce(self, params: dict) -> dict:
        brief = params.get("brief", "")
        result = self.hub.produce(brief)
        return {"result": result}

    def _cmd_suggest_progression(self, params: dict) -> dict:
        emotion = params.get("emotion", "neutral")
        key = params.get("key", "C")
        suggestion = self.hub.suggest_progression(emotion, key)
        return {"suggestion": suggestion}

    def _cmd_new_session(self, params: dict) -> dict:
        name = params.get("name", "untitled")
        self.hub.new_session(name)
        return {"success": True, "name": name}

    def _cmd_save_session(self, params: dict) -> dict:
        name = params.get("name")
        path = self.hub.save_session(name)
        return {"success": True, "path": path}

    def _cmd_load_session(self, params: dict) -> dict:
        filepath = params.get("filepath", "")
        success = self.hub.load_session(filepath)
        return {"success": success}

    def _cmd_list_sessions(self, params: dict) -> dict:
        sessions = self.hub.list_sessions()
        return {"sessions": sessions}

    def _cmd_get_state(self, params: dict) -> dict:
        channel = params.get("channel", "all")

        if channel == "all":
            return {
                "voice_state": self._get_channel_state("voice_state"),
                "daw_state": self._get_channel_state("daw_state"),
                "session": self._get_channel_state("session"),
            }

        return {"state": self._get_channel_state(channel)}

    def _cmd_get_health(self, params: dict) -> dict:
        return self.hub.check_llm_health()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Whether the server is running."""
        return self._running


# =============================================================================
# Convenience Functions
# =============================================================================


def create_websocket_server(
    hub: "UnifiedHub",
    port: int = 8765,
    auth_token: Optional[str] = None,
) -> HubWebSocketServer:
    """Create a WebSocket server for the hub."""
    return HubWebSocketServer(hub, port=port, auth_token=auth_token)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "HubWebSocketServer",
    "WSClient",
    "WSMessage",
    "MessageType",
    "create_websocket_server",
    "HAS_WEBSOCKETS",
]

