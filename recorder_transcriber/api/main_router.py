from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from recorder_transcriber.api.response_models import (
    WsCommand,
    WsConnectedEvent,
    WsErrorEvent,
    WsResultEvent,
    WsStateEvent,
)
from recorder_transcriber.core.di import get_listener_service
from recorder_transcriber.services.listening import ListenerEvent

router = APIRouter(prefix="/listen", tags=["listening"])


def _utcnow() -> datetime:
    return datetime.now(UTC)


@router.websocket("/ws")
async def websocket_listen(websocket: WebSocket) -> None:
    """WebSocket endpoint for voice-activated listening.

    Connection lifecycle:
    1. Client connects to /listen/ws
    2. Server sends 'connected' event
    3. Client sends {"action": "start"} to begin listening
    4. Server pushes state_change events (IDLE -> ARMED -> LISTENING)
    5. When utterance is captured and transcribed, server sends result event
    6. Server returns to IDLE and continues listening for next wake-word
    7. Client can send {"action": "stop"} to stop listening
    8. Client disconnects to end session

    Incoming messages:
        {"action": "start"} - Start listening for wake-words
        {"action": "stop"}  - Stop listening

    Outgoing events:
        {"type": "connected", "message": "..."}
        {"type": "state_change", "state": "IDLE|ARMED|LISTENING", "timestamp": "..."}
        {"type": "result", "recording_id": "...", "path": "...", "text": "...", ...}
        {"type": "error", "message": "...", "timestamp": "..."}
    """
    await websocket.accept()
    listener = get_listener_service()

    # Send connected event
    connected = WsConnectedEvent()
    await websocket.send_json(connected.model_dump(mode="json"))

    try:
        while True:
            # Wait for command from client
            data = await websocket.receive_json()

            try:
                command = WsCommand.model_validate(data)
            except ValidationError as e:
                error = WsErrorEvent(
                    message=f"Invalid command: {e.errors()}",
                    timestamp=_utcnow(),
                )
                await websocket.send_json(error.model_dump(mode="json"))
                continue

            if command.action == "start":
                if listener.is_listening:
                    error = WsErrorEvent(
                        message="Listening session already active",
                        timestamp=_utcnow(),
                    )
                    await websocket.send_json(error.model_dump(mode="json"))
                    continue

                try:
                    await listener.start()
                except RuntimeError as e:
                    error = WsErrorEvent(message=str(e), timestamp=_utcnow())
                    await websocket.send_json(error.model_dump(mode="json"))
                    continue

                # Send initial state
                state_event = WsStateEvent(
                    state=listener.state.value,
                    timestamp=_utcnow(),
                )
                await websocket.send_json(state_event.model_dump(mode="json"))

                # Stream events to client
                async for event in listener.events():
                    await _send_event(websocket, event)

            elif command.action == "stop":
                if not listener.is_listening:
                    error = WsErrorEvent(
                        message="No listening session active",
                        timestamp=_utcnow(),
                    )
                    await websocket.send_json(error.model_dump(mode="json"))
                    continue

                await listener.stop()
                state_event = WsStateEvent(
                    state="STOPPED",
                    timestamp=_utcnow(),
                )
                await websocket.send_json(state_event.model_dump(mode="json"))

    except WebSocketDisconnect:
        # Clean up on disconnect
        if listener.is_listening:
            await listener.stop()


async def _send_event(websocket: WebSocket, event: ListenerEvent) -> None:
    """Convert ListenerEvent to WebSocket message and send."""
    if event.type == "state_change" and event.state is not None:
        ws_event = WsStateEvent(
            state=event.state.value,
            timestamp=_utcnow(),
        )
        await websocket.send_json(ws_event.model_dump(mode="json"))

    elif event.type == "result" and event.result is not None:
        ws_event = WsResultEvent.from_result(
            recording=event.result.recording,
            transcript=event.result.transcript,
        )
        await websocket.send_json(ws_event.model_dump(mode="json"))

    elif event.type == "error" and event.error is not None:
        ws_event = WsErrorEvent(
            message=event.error,
            timestamp=_utcnow(),
        )
        await websocket.send_json(ws_event.model_dump(mode="json"))
