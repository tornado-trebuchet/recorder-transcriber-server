from pathlib import Path
from typing import Any

import numpy as np
import whisper

from recorder_transcriber.config import config


class WhisperAdapter:
	def __init__(self) -> None:
		wcfg = config.whisper
		self.model_name: str = str(wcfg["model"])
		self.device: str = str(wcfg.get("device", "cpu"))
		self.download_root: str = str(wcfg.get("download_root"))
		self._model: Any = None
		self.target_sample_rate: int = int(config.audio.get("samplerate", 16000))

	def transcribe_file(self, audio_path: str | Path) -> str:
		model = self._lazy_model()
		result = model.transcribe(str(audio_path), fp16=False)
		return self._extract_text(result)

	def transcribe_array(self, audio: np.ndarray, sample_rate: int, channels: int | None = None) -> str:
		prepared = self._prepare_audio(audio, sample_rate, channels)
		model = self._lazy_model()
		result = model.transcribe(prepared, fp16=False)
		return self._extract_text(result)

	def _lazy_model(self) -> Any:
		if self._model is None:
			self._model = whisper.load_model(  
				name=self.model_name,
				device=self.device,
				download_root=self.download_root,
			)
		return self._model

	def _prepare_audio(self, audio: np.ndarray, sample_rate: int, channels: int | None) -> np.ndarray:
		arr = np.asarray(audio, dtype=np.float32)

		if arr.ndim == 2:
			arr = arr.mean(axis=1)
		elif arr.ndim > 2:
			arr = arr.mean(axis=tuple(range(1, arr.ndim)))

		if channels and channels > 1 and arr.ndim == 2:
			arr = arr.mean(axis=1)

		if sample_rate != self.target_sample_rate:
			arr = WhisperAdapter._resample_linear(arr, sample_rate, self.target_sample_rate)

		return arr.astype(np.float32, copy=False)

	@staticmethod
	def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
		if audio.size == 0 or orig_sr == target_sr:
			return audio

		duration = audio.shape[0] / float(orig_sr)
		target_len = int(duration * target_sr)

		if target_len <= 1:
			return np.zeros(1, dtype=np.float32)

		x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
		x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)

		return np.interp(x_new, x_old, audio).astype(np.float32)

	@staticmethod
	def _extract_text(result: dict[str, Any]) -> str:
		segments = result.get("segments")
		if segments:
			stitched = " ".join(str(seg.get("text", "")).strip() for seg in segments)
			return stitched.strip()

		text = result.get("text", "")
		return str(text).strip()
