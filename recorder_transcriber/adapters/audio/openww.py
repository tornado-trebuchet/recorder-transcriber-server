import numpy as np
from openwakeword.model import Model  # type: ignore

from recorder_transcriber.core.logger import get_logger
from recorder_transcriber.domain.models import AudioFrame, WakeEvent
from recorder_transcriber.ports.wakeword import WakeWordPort

logger = get_logger("adapters.audio.openww")


class OpenWakeWordAdapter(WakeWordPort):

    def __init__(
        self,
        *,
        wakeword_models: list[str] | None = None,
        threshold: float = 0.5,
        melspec_model_path: str | None = None,
        embedding_model_path: str | None = None,
    ) -> None:
        models = [str(m) for m in wakeword_models] if wakeword_models else None
        kwargs: dict[str, str] = {}
        if melspec_model_path:
            kwargs["melspec_model_path"] = melspec_model_path
        if embedding_model_path:
            kwargs["embedding_model_path"] = embedding_model_path

        try:
            self._model = Model(wakeword_models=models, **kwargs)
        except Exception:
            logger.exception("Failed to load OpenWakeWord model with models=%s", models)
            raise

        self._threshold = float(threshold)
        self._active_models = models or []
        
        logger.info(
            "OpenWakeWord adapter initialized: models=%s, threshold=%.2f",
            self._active_models,
            self._threshold,
        )

    @property
    def active_models(self) -> list[str]:
        return self._active_models.copy()

    def reset(self) -> None:
        self._model.reset()

    def detect(self, frame: AudioFrame) -> WakeEvent:
        pcm = frame.to_mono_int16()

        logger.debug("Processing frame seq=%d for wake-word detection", frame.sequence)

        pred = self._model.predict(np.ascontiguousarray(pcm))
        scores = {str(k): float(v) for k, v in dict(pred).items()}
        detected = any(score >= self._threshold for score in scores.values())

        if detected:
            logger.info("Wake-word detected! scores=%s", scores)

        return WakeEvent(detected=detected, scores=scores)
