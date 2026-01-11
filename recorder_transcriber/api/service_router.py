from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from recorder_transcriber.api.response_models import (
    EnhancementRequest,
    EnhancementResponse,
    RecordingResponse,
    StartRecordingResponse,
    TranscriptionRequest,
    TranscriptResponse,
)
from recorder_transcriber.core.di import (
    get_enhancement_service,
    get_recorder_service,
    get_transcription_service,
)
from recorder_transcriber.domain.models import Transcript
from recorder_transcriber.services.enhancement import EnhancementService
from recorder_transcriber.services.recording import RecorderService
from recorder_transcriber.services.transcription import TranscriptionService

router = APIRouter(tags=["recording"])


@router.post("/start_recording", response_model=StartRecordingResponse)
def start_recording(
    recorder: RecorderService = Depends(get_recorder_service),
) -> StartRecordingResponse:
    """Start a new audio recording session."""
    try:
        session = recorder.start_recording()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return StartRecordingResponse.from_session(
        started_at=session.started_at,
        max_duration_seconds=session.max_duration_seconds,
    )


@router.post("/stop_recording", response_model=RecordingResponse)
def stop_recording(
    recorder: RecorderService = Depends(get_recorder_service),
) -> RecordingResponse:
    """Stop the current recording session and save the audio."""
    try:
        recording = recorder.stop_recording()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return RecordingResponse.from_recording(recording)


@router.post("/transcribe", response_model=TranscriptResponse)
def transcribe(
    payload: TranscriptionRequest,
    transcriber: TranscriptionService = Depends(get_transcription_service),
    recorder: RecorderService = Depends(get_recorder_service),
) -> TranscriptResponse:
    """Transcribe a previously recorded audio file."""
    try:
        recording = recorder.get_recording(payload.recording_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    transcript = transcriber.transcribe(recording)
    return TranscriptResponse.from_transcript(payload.recording_id, transcript)


@router.post("/enhance", response_model=EnhancementResponse)
def enhance(
    payload: EnhancementRequest,
    enhancer: EnhancementService = Depends(get_enhancement_service),
) -> EnhancementResponse:
    """Enhance a transcript into a structured note."""
    recording_path = Path(payload.recording_id) if payload.recording_id else None
    transcript = Transcript(text=payload.text, recording_path=recording_path)
    try:
        note = enhancer.enhance(transcript)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return EnhancementResponse.from_note(note, recording_id=payload.recording_id)


@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
