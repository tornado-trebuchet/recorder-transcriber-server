import numpy as np
from openwakeword.model import Model # type: ignore

from recorder_transcriber.model import WakeResult

class OpenWakeWordAdapter:
	def __init__(
		self,
		*,
		wakeword_models: list[str] | None = None,
		threshold: float = 0.5,
	) -> None:
		models = [str(m) for m in wakeword_models] if wakeword_models else None
		self._model = Model(wakeword_models=models)
		self._threshold = float(threshold)

	def predict(self, frame_pcm16: np.ndarray) -> WakeResult:
		pcm = _as_pcm16_mono(frame_pcm16)
		pred = self._model.predict(pcm)
		scores = {str(k): float(v) for k, v in dict(pred).items()}
		detected = any(score >= self._threshold for score in scores.values())
		return WakeResult(detected=detected, scores=scores)


def _as_pcm16_mono(frame: np.ndarray) -> np.ndarray:
	if frame.ndim == 2:
		frame = frame.mean(axis=1)
	elif frame.ndim != 1:
		raise ValueError("Wake frames must be 1D (mono) or 2D (frames, channels)")

	if frame.dtype == np.int16:
		pcm = frame
	elif np.issubdtype(frame.dtype, np.floating):
		# float audio is expected in [-1, 1]
		pcm = np.clip(frame, -1.0, 1.0)
		pcm = (pcm * 32767.0).astype(np.int16, copy=False)
	else:
		pcm = frame.astype(np.int16, copy=False)

	return np.ascontiguousarray(pcm)