from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status

from recorder_transcriber.model import Transcript
from recorder_transcriber.response_models import (
    EnhancementRequest,
    EnhancementResponse,
    RecordingResponse,
    StartRecordingResponse,
    TranscriptResponse,
    TranscriptionRequest,
)

from recorder_transcriber.services.enhancer import EnhancementService
from recorder_transcriber.services.recorder import RecorderService
from recorder_transcriber.services.transcriber import TranscriptionService

from recorder_transcriber.di import (
    get_recorder_service,
    get_transcription_service,
    get_enhancement_service,
)


app = FastAPI(title="recorder-transcriber", version="0.1.0")


@app.post("/start_recording", response_model=StartRecordingResponse)
def start_recording(recorder: RecorderService = Depends(get_recorder_service)) -> StartRecordingResponse:
	try:
		session = recorder.start_recording()
	except RuntimeError as exc:
		raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
	return StartRecordingResponse.from_session(
		started_at=session.started_at,
		max_duration_seconds=session.max_duration_seconds,
	)


@app.post("/stop_recording", response_model=RecordingResponse)
def stop_recording(recorder: RecorderService = Depends(get_recorder_service)) -> RecordingResponse:
	try:
		recording = recorder.stop_recording()
	except RuntimeError as exc:
		raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
	return RecordingResponse.from_recording(recording)


@app.post("/transcribe", response_model=TranscriptResponse)
def transcribe(
	payload: TranscriptionRequest,
	transcriber: TranscriptionService = Depends(get_transcription_service),
	recorder: RecorderService = Depends(get_recorder_service),
) -> TranscriptResponse:
	try:
		recording = recorder.get_recording(payload.recording_id)
	except KeyError as exc:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

	transcript = transcriber.transcribe(recording)
	return TranscriptResponse.from_transcript(payload.recording_id, transcript)


@app.post("/enhance", response_model=EnhancementResponse)
def enhance(
	payload: EnhancementRequest,
	enhancer: EnhancementService = Depends(get_enhancement_service),
) -> EnhancementResponse:
	recording_path = Path(payload.recording_id) if payload.recording_id else None
	transcript = Transcript(text=payload.text, recording_path=recording_path)
	note = enhancer.enhance(transcript)
	return EnhancementResponse.from_note(note, recording_id=payload.recording_id)
