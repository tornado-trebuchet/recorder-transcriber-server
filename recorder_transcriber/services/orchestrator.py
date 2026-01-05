from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Literal

import numpy as np

from recorder_transcriber.adapters.audio.openww import OpenWakeWordAdapter
from recorder_transcriber.adapters.audio.silerovad import SileroVadAdapter
from recorder_transcriber.adapters.audio.sddevice import AudioRecorderAdapter
from recorder_transcriber.model import Recording
from recorder_transcriber.services.recorder import AudioStoragePort, RecorderService


OrchestratorPhase = Literal["idle", "armed", "recording", "stopping", "error"]


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


@dataclass(slots=True)
class OrchestratorConfig:
	# Stream format for wake/VAD (keep fixed for correctness)
	sample_rate: int = 16000
	channels: int = 1
	blocksize: int = 512
	dtype: str = "float32"

	# Wake detection
	wake_window_seconds: float = 2.0
	wake_frame_ms: int = 80
	wake_threshold: float = 0.5
	wake_models: list[str] | None = None

	# VAD
	vad_threshold: float = 0.5
	vad_min_silence_ms: int = 250
	vad_speech_pad_ms: int = 30

	# Orchestration
	armed_timeout_seconds: float = 5.0
	max_utterance_seconds: float = 20.0
	end_hangover_ms: int = 300


@dataclass(slots=True)
class OrchestratorStatus:
	running: bool
	phase: OrchestratorPhase
	started_at: datetime | None
	last_wake_at: datetime | None
	last_recording_id: str | None
	last_error: str | None


@dataclass(slots=True)
class OrchestratorService:
	"""Server-wide wake+VAD loop.

	- Keeps only a small wake window in memory.
	- After wake word: records only while VAD is active and stops on endpoint.
	- Persists audio via existing ffmpeg adapter and registers it with RecorderService.
	"""

	recorder_service: RecorderService
	storage_adapter: AudioStoragePort
	stream: AudioRecorderAdapter
	wake: OpenWakeWordAdapter
	vad: SileroVadAdapter
	config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

	_lock: Lock = field(default_factory=Lock, init=False)
	_thread: Thread | None = field(default=None, init=False)
	_stop: Event = field(default_factory=Event, init=False)

	_phase: OrchestratorPhase = field(default="idle", init=False)
	_started_at: datetime | None = field(default=None, init=False)
	_last_wake_at: datetime | None = field(default=None, init=False)
	_last_recording_id: str | None = field(default=None, init=False)
	_last_error: str | None = field(default=None, init=False)

	def start(self) -> OrchestratorStatus:
		with self._lock:
			if self._thread and self._thread.is_alive():
				return self.status()
			if self.recorder_service.is_recording():
				raise RuntimeError("Cannot start orchestrator while manual recording is active")

			self._stop.clear()
			self._phase = "idle"
			self._started_at = _utcnow()
			self._last_error = None
			self._thread = Thread(target=self._run, name="orchestrator", daemon=True)
			self._thread.start()
			return self.status()

	def stop(self) -> OrchestratorStatus:
		with self._lock:
			self._stop.set()
			thread = self._thread

		if thread:
			thread.join(timeout=5.0)

		with self._lock:
			self._thread = None
			self._started_at = None
			self._phase = "idle"
			return self.status()

	def status(self) -> OrchestratorStatus:
		with self._lock:
			running = self._thread is not None and self._thread.is_alive()
			return OrchestratorStatus(
				running=running,
				phase=self._phase,
				started_at=self._started_at,
				last_wake_at=self._last_wake_at,
				last_recording_id=self._last_recording_id,
				last_error=self._last_error,
			)

	def _run(self) -> None:
		cfg = self.config
		wake_frame_samples = int(round(cfg.sample_rate * (cfg.wake_frame_ms / 1000.0)))
		wake_window_samples = int(round(cfg.sample_rate * cfg.wake_window_seconds))
		hangover_samples = int(round(cfg.sample_rate * (cfg.end_hangover_ms / 1000.0)))
		max_utterance_samples = int(round(cfg.sample_rate * cfg.max_utterance_seconds))

		wake_pcm_accum = np.zeros((0,), dtype=np.int16)

		wake_ring: deque[np.ndarray] = deque()
		ring_samples = 0

		armed_at: datetime | None = None
		armed_buf: deque[np.ndarray] = deque()
		armed_samples = 0
		max_armed_samples = int(round(cfg.sample_rate * min(cfg.armed_timeout_seconds, 1.0)))

		recording_blocks: list[np.ndarray] = []
		recording_samples = 0
		pending_stop_samples: int | None = None

		try:
			self.stream.start()
			self.stream.clear_buffer()
			self.vad.reset()

			while not self._stop.is_set():
				chunk = self.stream.read_chunk(timeout_seconds=0.25)
				if chunk is None:
					continue
				mono = _to_mono_float32(chunk)

				# Maintain wake window ring buffer
				wake_ring.append(mono)
				ring_samples += mono.shape[0]
				while ring_samples > wake_window_samples and wake_ring:
					dropped = wake_ring.popleft()
					ring_samples -= dropped.shape[0]

				# Wake detection runs while idle
				phase = self._phase
				if phase == "idle":
					wake_pcm_accum = _append_pcm16(wake_pcm_accum, mono)
					while wake_pcm_accum.shape[0] >= wake_frame_samples:
						frame = wake_pcm_accum[:wake_frame_samples]
						wake_pcm_accum = wake_pcm_accum[wake_frame_samples:]
						res = self.wake.predict(frame)
						if res.detected:
							armed_at = _utcnow()
							self._set_phase("armed")
							self._last_wake_at = armed_at
							self.vad.reset()
							armed_buf.clear()
							armed_samples = 0
							break
					continue

				# If we are armed, keep only a small buffer and wait for VAD start
				if phase == "armed":
					if armed_at and (_utcnow() - armed_at).total_seconds() > cfg.armed_timeout_seconds:
						self._set_phase("idle")
						armed_at = None
						armed_buf.clear()
						armed_samples = 0
						pending_stop_samples = None
						recording_blocks = []
						recording_samples = 0
						continue

					armed_buf.append(mono)
					armed_samples += mono.shape[0]
					while armed_samples > max_armed_samples and armed_buf:
						dropped = armed_buf.popleft()
						armed_samples -= dropped.shape[0]

					vad_event = self.vad.process(mono)
					if vad_event and vad_event.kind == "speech_start":
						recording_blocks = list(armed_buf)
						recording_samples = sum(b.shape[0] for b in recording_blocks)
						armed_buf.clear()
						armed_samples = 0
						pending_stop_samples = None
						self._set_phase("recording")
					continue

				# Recording: keep appending until endpoint
				if phase in ("recording", "stopping"):
					recording_blocks.append(mono)
					recording_samples += mono.shape[0]
					if recording_samples >= max_utterance_samples:
						pending_stop_samples = 0
						self._set_phase("stopping")

					if phase == "recording":
						vad_event = self.vad.process(mono)
						if vad_event and vad_event.kind == "speech_end":
							pending_stop_samples = hangover_samples
							self._set_phase("stopping")

					if pending_stop_samples is not None:
						remaining = pending_stop_samples - mono.shape[0]
						if remaining <= 0:
							persisted_id = self._finalize_recording(cfg, recording_blocks)
							self._last_recording_id = persisted_id
							recording_blocks = []
							recording_samples = 0
							pending_stop_samples = None
							self._set_phase("idle")
						else:
							pending_stop_samples = remaining

		except Exception as exc:
			self._last_error = str(exc)
			self._set_phase("error")
		finally:
			try:
				self.stream.stop()
			except Exception:
				pass

	def _finalize_recording(self, cfg: OrchestratorConfig, blocks: list[np.ndarray]) -> str:
		if not blocks:
			raise RuntimeError("No audio captured for segment")

		payload = np.concatenate(blocks, axis=0)
		recording = Recording(
			data=payload,
			path=None,
			sample_rate=cfg.sample_rate,
			channels=cfg.channels,
			dtype=cfg.dtype,
			blocksize=cfg.blocksize,
			device_name=None,
		)
		persisted = self.storage_adapter.save_recording(recording)
		return self.recorder_service.store_recording(persisted)

	def _set_phase(self, phase: OrchestratorPhase) -> None:
		with self._lock:
			self._phase = phase


def _to_mono_float32(chunk: np.ndarray) -> np.ndarray:
	if chunk.ndim == 1:
		mono = chunk
	elif chunk.ndim == 2:
		mono = chunk.mean(axis=1)
	else:
		raise ValueError("Audio chunks must be 1D or 2D")
	if mono.dtype != np.float32:
		mono = mono.astype(np.float32, copy=False)
	return mono


def _append_pcm16(acc: np.ndarray, mono_float32: np.ndarray) -> np.ndarray:
	pcm = np.clip(mono_float32, -1.0, 1.0)
	pcm16 = (pcm * 32767.0).astype(np.int16, copy=False)
	if acc.size == 0:
		return np.ascontiguousarray(pcm16)
	return np.concatenate([acc, pcm16], axis=0)
