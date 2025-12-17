from pathlib import Path
from typing import Any

import numpy as np
import whisper # type: ignore

from recorder_transcriber.config import config
from recorder_transcriber.model import Recording, Transcript


class WhisperAdapter:
	def __init__(self) -> None:
		wcfg = config.whisper
		self.model_name: str = str(wcfg["model"])
		self.device: str = str(wcfg.get("device", "cpu"))
		self.download_root: str = str(wcfg.get("download_root"))
		self._model: Any = None
		self.target_sample_rate: int = int(config.audio.get("samplerate", 16000))

	@staticmethod
	def _extract_text(result: dict[str, Any]) -> str:
		segments = result.get("segments")
		if segments:
			stitched = " ".join(str(seg.get("text", "")).strip() for seg in segments)
			return stitched.strip()

		text = result.get("text", "")
		return str(text).strip()

	def _lazy_model(self) -> Any:
		if self._model is None:
			self._model = whisper.load_model(  
				name=self.model_name,
				device=self.device,
				download_root=self.download_root,
			)
		return self._model

	def _prepare_audio(self, audio: np.ndarray, sample_rate: int, channels: int | None) -> np.ndarray:
		arr = np.asarray(audio, dtype=np.float32, copy=False)
		if arr.ndim > 1:
			arr = arr.reshape(-1)
		return arr

	def _transcribe_file(self, audio_path: str | Path) -> str:
		model = self._lazy_model()
		result = model.transcribe(str(audio_path), fp16=False)
		return self._extract_text(result)

	def _transcribe_array(self, audio: np.ndarray, sample_rate: int, channels: int | None = None) -> str:
		prepared = self._prepare_audio(audio, sample_rate, channels)
		model = self._lazy_model()
		result = model.transcribe(prepared, fp16=False)
		return self._extract_text(result)

	def transcribe_recording(self, recording: Recording) -> Transcript:
		if recording.data is not None:
			transcript = self._transcribe_array(recording.data, recording.sample_rate, recording.channels)
			return Transcript(transcript)
		if recording.path is not None:
			transcript = self._transcribe_file(recording.path)
			return Transcript(transcript)
		raise ValueError("Recording must include data or path")