from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from recorder_transcriber.config import config
from recorder_transcriber.model import Recording, Transcript


class WhisperAdapter:
	def __init__(self) -> None:
		wcfg = config.whisper
		self.model_name: str = wcfg["model"]
		self.device: str = wcfg.get("device", "gpu")
		self.download_root: str = wcfg.get("download_root", "")
		self._model: Any = None
		self.target_sample_rate: int = int(config.audio.get("samplerate", 16000))

	@staticmethod
	def _extract_text(segments: Any, info: Any) -> str:
		"""Extract text from faster-whisper's iterator result."""
		text_parts = []
		for segment in segments:
			text_parts.append(segment.text.strip())
		return " ".join(text_parts).strip()

	def _lazy_model(self) -> WhisperModel:
		if self._model is None:
			compute_type = "int8" 
			self._model = WhisperModel(  
				self.model_name,
				device=self.device,
				download_root=self.download_root,
				compute_type=compute_type,
			)
		return self._model

	def _prepare_audio(self, audio: np.ndarray, sample_rate: int, channels: int | None) -> np.ndarray:
		arr = np.asarray(audio, dtype=np.float32, copy=False)
		if arr.ndim > 1:
			arr = arr.reshape(-1)
		return arr

	def _transcribe_file(self, audio_path: str | Path) -> str:
		model = self._lazy_model()
		segments, info = model.transcribe(str(audio_path))
		return self._extract_text(segments, info)

	def _transcribe_array(self, audio: np.ndarray, sample_rate: int, channels: int | None = None) -> str:
		prepared = self._prepare_audio(audio, sample_rate, channels)
		model = self._lazy_model()
		segments, info = model.transcribe(prepared)
		return self._extract_text(segments, info)

	def transcribe_recording(self, recording: Recording) -> Transcript:
		if recording.data is not None:
			transcript = self._transcribe_array(recording.data, recording.sample_rate, recording.channels)
			return Transcript(transcript)
		if recording.path is not None:
			transcript = self._transcribe_file(recording.path)
			return Transcript(transcript)
		raise ValueError("Recording must include data or path")