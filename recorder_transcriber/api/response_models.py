from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
		return cls(	
			body=note.body, 
			title=note.title, 
			tags=note.tags, 
			created_at=note.created_at, 
			recording_id=recording_id)


# WebSocket message models for listening service

class WsCommand(BaseModel):
	"""Incoming WebSocket command from client."""
	model_config = ConfigDict(extra="forbid")

	action: Literal["start", "stop"]


class WsStateEvent(BaseModel):
	"""WebSocket event for state changes."""
	model_config = ConfigDict(extra="forbid")

	type: Literal["state_change"] = "state_change"
	state: str
	timestamp: datetime


class WsResultEvent(BaseModel):
	"""WebSocket event when utterance is captured and transcribed."""
	model_config = ConfigDict(extra="forbid")

	type: Literal["result"] = "result"
	recording_id: str
	path: str
	text: str
	captured_at: datetime
	transcribed_at: datetime

	@classmethod
	def from_result(cls, recording: Recording, transcript: Transcript) -> "WsResultEvent":
		if recording.path is None:
			raise ValueError("Recording path is required for WebSocket result events")
		path_str = str(recording.path)
		return cls(
			recording_id=path_str,
			path=path_str,
			text=transcript.text,
			captured_at=recording.captured_at,
			transcribed_at=transcript.generated_at,
		)


class WsErrorEvent(BaseModel):
	"""WebSocket event for errors."""
	model_config = ConfigDict(extra="forbid")

	type: Literal["error"] = "error"
	message: str
	timestamp: datetime


class WsConnectedEvent(BaseModel):
	"""WebSocket event sent on successful connection."""
	model_config = ConfigDict(extra="forbid")

	type: Literal["connected"] = "connected"
	message: str = "WebSocket connected. Send {\"action\": \"start\"} to begin listening."
