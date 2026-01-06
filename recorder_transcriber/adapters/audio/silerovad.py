from collections import deque
from typing import Iterator

import numpy as np
from silero_vad import load_silero_vad  # type: ignore
from silero_vad.utils_vad import VADIterator  # type: ignore

from recorder_transcriber.core.logger import get_logger
from recorder_transcriber.domain.models import AudioFrame, VadEvent
from recorder_transcriber.ports.vad import VadPort

logger = get_logger("adapters.audio.silerovad")


_SILERO_FRAME_SIZE = 512


class SileroVadAdapter(VadPort):
    """Voice Activity Detection adapter using Silero VAD model.

    Implements VadPort and handles internal buffering to meet Silero's
    512-sample frame requirement regardless of input frame size.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 250,
        speech_pad_ms: int = 30,
        sampling_rate: int = 16000,
    ) -> None:
        try:
            model = load_silero_vad()
        except Exception:
            logger.exception("Failed to load Silero VAD model")
            raise

        self._sampling_rate = int(sampling_rate)
        self._iterator = VADIterator(
            model,
            threshold=float(threshold),
            sampling_rate=self._sampling_rate,
            min_silence_duration_ms=int(min_silence_duration_ms),
            speech_pad_ms=int(speech_pad_ms),
        )
        # Internal buffer for accumulating samples to reach 512
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0

        logger.info(
            "SileroVAD adapter initialized: threshold=%.2f, min_silence=%dms, sample_rate=%d",
            threshold,
            min_silence_duration_ms,
            sampling_rate,
        )

    @property
    def required_frame_size(self) -> int | None:
        """Silero requires 512 samples, but we handle buffering internally."""
        return None  # We buffer internally, so callers don't need to worry

    def reset(self) -> None:
        """Reset VAD state and internal buffer."""
        self._iterator.reset_states()
        self._buffer.clear()
        self._buffer_samples = 0

    def process(self, frame: AudioFrame) -> VadEvent | None:
        """Process an audio frame through VAD.

        Uses frame.to_mono_float32() for conversion, then buffers internally
        to produce 512-sample chunks for Silero.

        Returns:
            VadEvent on speech start/end transitions, None otherwise.
        """
        mono = frame.to_mono_float32()
        self._buffer.append(mono)
        self._buffer_samples += mono.shape[0]
        logger.debug("VAD processing frame seq=%d, buffer_samples=%d", frame.sequence, self._buffer_samples)

        # Process all complete 512-sample chunks
        # Prioritize "end" events to ensure speech end transitions are never missed
        last_event: VadEvent | None = None
        for chunk in self._emit_chunks():
            event = self._process_chunk(chunk)
            if event is not None:
                # If we already have an "end" event, don't overwrite it
                # This ensures speech end detection takes priority
                if last_event is not None and not last_event.detected:
                    continue
                last_event = event

        return last_event

    def _emit_chunks(self) -> Iterator[np.ndarray]:
        """Yield complete 512-sample chunks from the buffer."""
        while self._buffer_samples >= _SILERO_FRAME_SIZE:
            # Concatenate buffered arrays
            combined = np.concatenate(list(self._buffer))
            self._buffer.clear()

            # Extract one chunk
            chunk = combined[:_SILERO_FRAME_SIZE]
            remainder = combined[_SILERO_FRAME_SIZE:]

            # Put remainder back in buffer
            if remainder.shape[0] > 0:
                self._buffer.append(remainder)
                self._buffer_samples = remainder.shape[0]
            else:
                self._buffer_samples = 0

            yield chunk

    def _process_chunk(self, chunk: np.ndarray) -> VadEvent | None:
        """Process a single 512-sample chunk through Silero."""
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
