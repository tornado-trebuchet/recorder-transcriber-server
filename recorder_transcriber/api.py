from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status

from recorder_transcriber.adapters.audio.ffmpeg import AudioConverterAdapter
from recorder_transcriber.adapters.audio.sounddevice import AudioRecorderAdapter
from recorder_transcriber.adapters.speech_to_text.whisper import WhisperAdapter
from recorder_transcriber.adapters.text_to_text.localllm import TextEnhancer
from recorder_transcriber.config import config
from recorder_transcriber.model import Transcript
from recorder_transcriber.response_models import (
	EnhancementRequest,
	EnhancementResponse,
	RecordingResponse,
	StartRecordingResponse,
	TranscriptResponse,
	TranscriptionRequest,
)
from recorder_transcriber.services.recorder import RecorderService
from recorder_transcriber.services.transcriber import TranscriptionService
from recorder_transcriber.services.enhancer import EnhancementService


app = FastAPI(title="recorder-transcriber", version="0.1.0")


_recorder_service = RecorderService(
	capture_adapter=AudioRecorderAdapter(),
	storage_adapter=AudioConverterAdapter(),
	max_duration_seconds=config.recording_max_seconds,
)

_transcription_service = TranscriptionService(adapter=WhisperAdapter())
_enhancement_service = EnhancementService(adapter=TextEnhancer())


def get_recorder_service() -> RecorderService:
	return _recorder_service


def get_transcription_service() -> TranscriptionService:
	return _transcription_service


def get_enhancement_service() -> EnhancementService:
	return _enhancement_service


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
