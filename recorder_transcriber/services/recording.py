from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import Event, Lock, Thread

import numpy as np

from recorder_transcriber.domain.models import Recording
from recorder_transcriber.ports.audiostream import AudioStreamPort, AudioStreamReader
from recorder_transcriber.ports.storage import AudioStoragePort


def _recording_store() -> dict[str, Recording]:
	return {}


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)

@dataclass(slots=True)
class RecorderSession:
	"""Tracks metadata for the currently running capture session."""

	started_at: datetime
	max_duration_seconds: int


@dataclass(slots=True)
class RecorderService:
	"""Coordinates audio capture and persistence for a single active session."""

	stream: AudioStreamPort
	storage_adapter: AudioStoragePort
	max_duration_seconds: int
	_lock: Lock = field(default_factory=Lock, init=False)
	_session: RecorderSession | None = field(default=None, init=False)
	_recordings: dict[str, Recording] = field(default_factory=_recording_store, init=False)
	_reader: AudioStreamReader | None = field(default=None, init=False)
	_thread: Thread | None = field(default=None, init=False)
	_stop: Event = field(default_factory=Event, init=False)
	_chunks: list[np.ndarray] = field(default_factory=list, init=False)

	def start_recording(self) -> RecorderSession:
		with self._lock:
			if self._session is not None:
				raise RuntimeError("Recording session already active")
			if not self.stream.is_running():
				raise RuntimeError("Audio stream is not running")
			self._stop.clear()
			self._chunks = []
			reader = self.stream.subscribe(name="recorder", max_frames=4096)
			self._reader = reader
			self._thread = Thread(target=self._run_capture, name="recorder-capture", daemon=True)
			self._thread.start()
			session = RecorderSession(started_at=_utcnow(), max_duration_seconds=self.max_duration_seconds)
			self._session = session
			return session

	def stop_recording(self) -> Recording:
		with self._lock:
			if self._session is None:
				raise RuntimeError("Recording session is not active")
			thread = self._thread
			reader = self._reader
			self._stop.set()

		if thread:
			thread.join(timeout=5.0)

		with self._lock:
			self._session = None
			self._thread = None
			self._reader = None
			chunks = self._chunks
			self._chunks = []

		if reader is not None:
			try:
				reader.close()
			except Exception:
				pass

		if not chunks:
			raise RuntimeError("Recorder returned no audio data")

		payload = np.concatenate(chunks, axis=0)
		fmt = self.stream.audio_format()
		raw = Recording(
			data=payload,
			path=None,
			sample_rate=fmt.sample_rate,
			channels=fmt.channels,
			dtype=fmt.dtype,
			blocksize=fmt.blocksize,
			device_name=None,
		)

		persisted = self.storage_adapter.save_recording(raw)
		self.store_recording(persisted)
		return persisted

	def _run_capture(self) -> None:
		reader = self._reader
		if reader is None:
			return
		while not self._stop.is_set():
			frame = reader.read(timeout_seconds=0.25)
			if frame is None:
				continue
			with self._lock:
				self._chunks.append(frame.data)

	def store_recording(self, recording: Recording) -> str:
		"""Register an already-persisted recording in the in-memory registry."""
		if recording.path is None:
			raise RuntimeError("Recording must have a path to be stored")
		recording_id = str(recording.path)
		self._recordings[recording_id] = recording
		return recording_id

	def get_recording(self, recording_id: str) -> Recording:
		try:
			recording = self._recordings[recording_id]
		except KeyError as exc:
			raise KeyError(f"Recording '{recording_id}' not found") from exc
		return replace(recording)

	def is_recording(self) -> bool:
		return self._session is not None
