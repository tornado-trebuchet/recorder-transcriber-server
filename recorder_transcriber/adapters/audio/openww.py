import numpy as np
from openwakeword.model import Model  # type: ignore

from recorder_transcriber.domain.models import AudioFrame, WakeEvent
from recorder_transcriber.ports.wakeword import WakeWordPort


class OpenWakeWordAdapter(WakeWordPort):
    """Wake-word detection adapter using OpenWakeWord library."""

    def __init__(
        self,
        *,
        wakeword_models: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        models = [str(m) for m in wakeword_models] if wakeword_models else None
        self._model = Model(wakeword_models=models)
        self._threshold = float(threshold)
        self._active_models = models or []

    @property
    def active_models(self) -> list[str]:
        """Return list of active wake-word model names."""
        return self._active_models.copy()

    def reset(self) -> None:
        """Reset internal state of the wake-word model."""
        self._model.reset()

    def detect(self, frame: AudioFrame) -> WakeEvent:
        """Check for wake-word in the given frame.

        Uses frame.to_mono_int16() for format conversion to PCM16.

        Returns:
            WakeEvent with detection result and confidence scores.
        """
        pcm = frame.to_mono_int16()
        pred = self._model.predict(np.ascontiguousarray(pcm))
        scores = {str(k): float(v) for k, v in dict(pred).items()}
        detected = any(score >= self._threshold for score in scores.values())
        return WakeEvent(detected=detected, scores=scores)
