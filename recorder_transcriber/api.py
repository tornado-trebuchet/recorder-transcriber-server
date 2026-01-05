from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from recorder_transcriber.model import Transcript
from recorder_transcriber.response_models import (
    EnhancementRequest,
    EnhancementResponse,
	OrchestratorStatusResponse,
    RecordingResponse,
    StartRecordingResponse,
    TranscriptResponse,
    TranscriptionRequest,
)

from recorder_transcriber.services.enhancer import EnhancementService
from recorder_transcriber.services.orchestrator import OrchestratorService
from recorder_transcriber.services.recorder import RecorderService
from recorder_transcriber.services.transcriber import TranscriptionService

from recorder_transcriber.configuration import load_config
from recorder_transcriber.di import (
	build_container,
    get_recorder_service,
    get_transcription_service,
    get_enhancement_service,
	get_orchestrator_service,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
	cfg = load_config()
	app.state.container = build_container(cfg)
	try:
		yield
	finally:
		container = getattr(app.state, "container", None)
		if container is not None:
			try:
				container.shutdown()
			except Exception:
				pass


app = FastAPI(title="recorder-transcriber", version="0.1.0", lifespan=_lifespan)



def _orchestrator_dep(request: Request) -> OrchestratorService:
	try:
		return get_orchestrator_service(request)
	except RuntimeError as exc:
		raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


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


@app.post("/orchestrator/start", response_model=OrchestratorStatusResponse)
def start_orchestrator(
	orchestrator: OrchestratorService = Depends(_orchestrator_dep),
) -> OrchestratorStatusResponse:
	try:
		status_obj = orchestrator.start()
	except RuntimeError as exc:
		raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
	return OrchestratorStatusResponse.from_status(status_obj)


@app.post("/orchestrator/stop", response_model=OrchestratorStatusResponse)
def stop_orchestrator(
	orchestrator: OrchestratorService = Depends(_orchestrator_dep),
) -> OrchestratorStatusResponse:
	status_obj = orchestrator.stop()
	return OrchestratorStatusResponse.from_status(status_obj)


@app.get("/orchestrator/status", response_model=OrchestratorStatusResponse)
def orchestrator_status(
	orchestrator: OrchestratorService = Depends(_orchestrator_dep),
) -> OrchestratorStatusResponse:
	status_obj = orchestrator.status()
	return OrchestratorStatusResponse.from_status(status_obj)
