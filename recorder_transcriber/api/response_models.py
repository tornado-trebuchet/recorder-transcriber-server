from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from recorder_transcriber.domain.models import Note, Recording, Transcript


class StartRecordingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	status: Literal["recording"] = "recording"
	started_at: datetime
	max_duration_seconds: int

	@classmethod
	def from_session(cls, started_at: datetime, max_duration_seconds: int) -> "StartRecordingResponse":
		return cls(started_at=started_at, max_duration_seconds=max_duration_seconds)


class RecordingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	recording_id: str
	path: str
	captured_at: datetime

	@classmethod
	def from_recording(cls, recording: Recording) -> "RecordingResponse":
		if recording.path is None:
			raise ValueError("Recording path is required for API responses")
		path_str = str(recording.path)
		return cls(recording_id=path_str, path=path_str, captured_at=recording.captured_at)


class TranscriptionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid")

	recording_id: str = Field(..., min_length=1)


class TranscriptResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	recording_id: str
	text: str
	generated_at: datetime

	@classmethod
	def from_transcript(cls, recording_id: str, transcript: Transcript) -> "TranscriptResponse":
		return cls(
			recording_id=recording_id,
			text=transcript.text,
			generated_at=transcript.generated_at,
		)


class EnhancementRequest(BaseModel):
	model_config = ConfigDict(extra="forbid")

	text: str = Field(..., min_length=1)
	recording_id: str | None = Field(default=None, description="Optional link back to previously saved recording")


class EnhancementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	body: str
	title: str
	tags: list[str]
	created_at: datetime
	recording_id: str | None = None

	@classmethod
	def from_note(cls, note: Note, recording_id: str | None = None) -> "EnhancementResponse":
		return cls(body=note.body, title=note.title, tags=note.tags, created_at=note.created_at, recording_id=recording_id)


class ListeningStartResponse(BaseModel):
	"""Response when listening session starts successfully."""
	model_config = ConfigDict(extra="forbid")

	status: Literal["listening"] = "listening"
	state: str
	started_at: datetime

	@classmethod
	def from_state(cls, state: str, started_at: datetime) -> "ListeningStartResponse":
		return cls(state=state, started_at=started_at)


class ListeningStatusResponse(BaseModel):
	"""Response for current listening session status."""
	model_config = ConfigDict(extra="forbid")

	is_listening: bool
	state: str

	@classmethod
	def from_service(cls, is_listening: bool, state: str) -> "ListeningStatusResponse":
		return cls(is_listening=is_listening, state=state)


class ListeningResultResponse(BaseModel):
	"""Response when listening session completes with a transcription."""
	model_config = ConfigDict(extra="forbid")

	status: Literal["completed"] = "completed"
	recording_id: str
	path: str
	text: str
	captured_at: datetime
	transcribed_at: datetime

	@classmethod
	def from_result(
		cls,
		recording: Recording,
		transcript: Transcript,
	) -> "ListeningResultResponse":
		if recording.path is None:
			raise ValueError("Recording path is required for API responses")
		path_str = str(recording.path)
		return cls(
			recording_id=path_str,
			path=path_str,
			text=transcript.text,
			captured_at=recording.captured_at,
			transcribed_at=transcript.generated_at,
		)
