from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from recorder_transcriber.core.di import get_listener_service
from recorder_transcriber.api.response_models import (
    ListeningResultResponse,
    ListeningStartResponse,
    ListeningStatusResponse,
)
from recorder_transcriber.services.listening import ListenerService


router = APIRouter(prefix="/listen", tags=["listening"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@router.post("/start", response_model=ListeningStartResponse)
def start_listening(
    listener: ListenerService = Depends(get_listener_service),
) -> ListeningStartResponse:
    """Start voice-activated listening session.

    Begins listening for a wake-word. When detected, transitions to ARMED
    state and waits for speech. When speech ends, the utterance is saved
    and transcribed automatically.

    Returns:
        ListeningStartResponse with current state and start time.

    Raises:
        HTTPException 409: If a listening session is already active.
        HTTPException 409: If the audio stream is not running.
    """
    try:
        listener.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ListeningStartResponse.from_state(
        state=listener.state,
        started_at=_utcnow(),
    )


@router.post("/stop", response_model=ListeningResultResponse | None)
def stop_listening(
    listener: ListenerService = Depends(get_listener_service),
) -> ListeningResultResponse | None:
    """Stop the current listening session.

    Stops listening for wake-words and voice activity. If an utterance
    was captured and transcribed, returns the result.

    Returns:
        ListeningResultResponse if an utterance was captured, None otherwise.

    Raises:
        HTTPException 409: If no listening session is active.
    """
    if not listener.is_listening():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No listening session is active",
        )
    result = listener.stop()
    if result is None:
        return None
    return ListeningResultResponse.from_result(
        recording=result.recording,
        transcript=result.transcript,
    )


@router.get("/status", response_model=ListeningStatusResponse)
def get_listening_status(
    listener: ListenerService = Depends(get_listener_service),
) -> ListeningStatusResponse:
    """Get the current status of the listening service.

    Returns:
        ListeningStatusResponse with is_listening flag and current state.
    """
    return ListeningStatusResponse.from_service(
        is_listening=listener.is_listening(),
        state=listener.state,
    )
