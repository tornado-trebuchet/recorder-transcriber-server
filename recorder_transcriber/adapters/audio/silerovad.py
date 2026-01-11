from collections import deque
from collections.abc import Iterator

import numpy as np
from silero_vad import load_silero_vad  # type: ignore
from silero_vad.utils_vad import VADIterator  # type: ignore

from recorder_transcriber.core.logger import get_logger
from recorder_transcriber.domain.models import AudioFrame, VadEvent
from recorder_transcriber.ports.vad import VadPort

logger = get_logger("adapters.audio.silerovad")


class SileroVadAdapter(VadPort):

    def __init__(
        self,
        *,
        threshold: float,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        sampling_rate: int,
    ) -> None:
        
        model = load_silero_vad()

        self.frame_size = 512 # FIXME: magic number, silero frame size
        self._sampling_rate = int(sampling_rate)
        self._iterator = VADIterator(
            model,
            threshold=float(threshold),
            sampling_rate=self._sampling_rate,
            min_silence_duration_ms=int(min_silence_duration_ms),
            speech_pad_ms=int(speech_pad_ms),
        )
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0

        logger.info(
            "SileroVAD adapter initialized: threshold=%.2f, min_silence=%dms, sample_rate=%d",
            threshold,
            min_silence_duration_ms,
            sampling_rate,
        )

    def reset(self) -> None:
        self._iterator.reset_states()
        self._buffer.clear()
        self._buffer_samples = 0

    def process(self, frame: AudioFrame) -> VadEvent | None:
        mono = frame.to_mono_float32()
        self._buffer.append(mono)
        self._buffer_samples += mono.shape[0]

        logger.debug("VAD processing frame seq=%d, buffer_samples=%d", frame.sequence, self._buffer_samples)

        last_event: VadEvent | None = None
        for chunk in self._emit_chunks():
            event = self._process_chunk(chunk)
            if event is not None:
                if last_event is not None and not last_event.detected:
                    continue
                last_event = event

        return last_event

    def _emit_chunks(self) -> Iterator[np.ndarray]:
        while self._buffer_samples >= self.frame_size:
            combined = np.concatenate(list(self._buffer))
            self._buffer.clear()

            chunk = combined[:self.frame_size]
            remainder = combined[self.frame_size:]

            if remainder.shape[0] > 0:
                self._buffer.append(remainder)
                self._buffer_samples = remainder.shape[0]
            else:
                self._buffer_samples = 0

            yield chunk

    def _process_chunk(self, chunk: np.ndarray) -> VadEvent | None:
        event = self._iterator(chunk, return_seconds=False)
        if event is None:
            return None
        if "start" in event:
            logger.info("Speech start detected")
            return VadEvent(detected=True)
        if "end" in event:
            logger.info("Speech end detected")
            return VadEvent(detected=False)
        return None
