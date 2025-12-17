from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Protocol

from recorder_transcriber.model import Recording


def _recording_store() -> dict[str, Recording]:
	return {}


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


class AudioCapturePort(Protocol):
	"""Interface implemented by audio capture adapters."""

	def start(self) -> None:
		...

	def stop(self) -> Recording | None:
		...


class AudioStoragePort(Protocol):
	"""Persists normalized recordings to durable storage."""

	def save_recording(self, recording: Recording) -> Recording:
		...


@dataclass(slots=True)
class RecorderSession:
	"""Tracks metadata for the currently running capture session."""

	started_at: datetime
	max_duration_seconds: int


@dataclass(slots=True)
class RecorderService:
	"""Coordinates audio capture and persistence for a single active session."""

	capture_adapter: AudioCapturePort
	storage_adapter: AudioStoragePort
	max_duration_seconds: int
	_lock: Lock = field(default_factory=Lock, init=False)
	_session: RecorderSession | None = field(default=None, init=False)
	_recordings: dict[str, Recording] = field(default_factory=_recording_store, init=False)

	def start_recording(self) -> RecorderSession:
		with self._lock:
			if self._session is not None:
				raise RuntimeError("Recording session already active")
			self.capture_adapter.start()
			session = RecorderSession(started_at=_utcnow(), max_duration_seconds=self.max_duration_seconds)
			self._session = session
			return session

	def stop_recording(self) -> Recording:
		with self._lock:
			if self._session is None:
				raise RuntimeError("Recording session is not active")
			raw = self.capture_adapter.stop()
			self._session = None

		if raw is None:
			raise RuntimeError("Recorder returned no audio data")

		persisted = self.storage_adapter.save_recording(raw)
		if persisted.path is None:
			raise RuntimeError("Recorder failed to persist audio")

		recording_id = str(persisted.path)
		self._recordings[recording_id] = persisted
		return persisted

	def get_recording(self, recording_id: str) -> Recording:
		try:
			recording = self._recordings[recording_id]
		except KeyError as exc:
			raise KeyError(f"Recording '{recording_id}' not found") from exc
		return replace(recording)

	def is_recording(self) -> bool:
		return self._session is not None
