import numpy as np
from silero_vad import load_silero_vad # type: ignore
from silero_vad.utils_vad import VADIterator # type: ignore

from recorder_transcriber.model import VadEvent


class SileroVadAdapter:
	def __init__(
		self,
		*,
		sampling_rate: int = 16000,
		threshold: float = 0.5,
		min_silence_duration_ms: int = 200,
		speech_pad_ms: int = 30,
	) -> None:
		if sampling_rate != 16000:
			raise ValueError("SileroVadAdapter currently expects 16kHz audio")

		model = load_silero_vad()
		self._iterator = VADIterator(
			model,
			threshold=float(threshold),
			sampling_rate=int(sampling_rate),
			min_silence_duration_ms=int(min_silence_duration_ms),
			speech_pad_ms=int(speech_pad_ms),
		)

	def reset(self) -> None:
		self._iterator.reset_states()

	def process(self, frame: np.ndarray) -> VadEvent | None:
		"""Process one audio frame"""
		mono = _to_mono_float32(frame)
		if mono.shape[0] != 512:
			raise ValueError(f"Expected 512 samples per frame, got {mono.shape[0]}")

		event = self._iterator(mono, return_seconds=False)
		if event is None:
			return None
		if "start" in event:
			return VadEvent(kind="speech_start")
		if "end" in event:
			return VadEvent(kind="speech_end")
		return None


def _to_mono_float32(frame: np.ndarray) -> np.ndarray:
	if frame.ndim == 1:
		mono = frame
	elif frame.ndim == 2:
		# (frames, channels)
		mono = frame.mean(axis=1)
	else:
		raise ValueError("Audio frame must be 1D or 2D")

	if mono.dtype != np.float32:
		mono = mono.astype(np.float32, copy=False)
	return mono