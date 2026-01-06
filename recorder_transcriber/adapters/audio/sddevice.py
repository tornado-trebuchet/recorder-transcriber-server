from dataclasses import dataclass
from threading import Lock
from typing import Any
import queue

import numpy as np
import sounddevice  # type: ignore

from recorder_transcriber.core.logger import get_logger
from recorder_transcriber.domain.models import AudioFormat, AudioFrame
from recorder_transcriber.ports.audiostream import AudioStreamPort, AudioStreamReader

logger = get_logger("adapters.audio.sddevice")


@dataclass(slots=True)
class _Subscriber:
    name: str
    queue: queue.Queue[AudioFrame]
    closed: bool = False


class _QueueReader(AudioStreamReader):
    """Concrete AudioStreamReader backed by a thread-safe queue."""

    def __init__(self, owner: "SoundDeviceAudioStreamAdapter", subscriber_id: int) -> None:
        self._owner = owner
        self._subscriber_id = subscriber_id

    def read(self, timeout_seconds: float | None = None) -> AudioFrame | None:
        subscriber = self._owner._get_subscriber(self._subscriber_id)
        if subscriber is None or subscriber.closed:
            return None
        try:
            if timeout_seconds is None:
                return subscriber.queue.get()  # Block indefinitely
            elif timeout_seconds == 0:
                return subscriber.queue.get_nowait()
            else:
                return subscriber.queue.get(timeout=max(float(timeout_seconds), 0.0))
        except queue.Empty:
            return None

    def close(self) -> None:
        self._owner._close_subscriber(self._subscriber_id)


class SoundDeviceAudioStreamAdapter(AudioStreamPort):
    """Audio capture adapter using sounddevice library.

    Implements AudioStreamPort by capturing audio via InputStream and
    distributing AudioFrame objects to subscribers via pub-sub pattern.
    """

    def __init__(self, *, audio_format: AudioFormat) -> None:
        self._format = audio_format

        self._lock = Lock()
        self._stream: sounddevice.InputStream | None = None
        self._selected_device_name: str | None = None
        self._next_subscriber_id = 1
        self._subscribers: dict[int, _Subscriber] = {}
        self._frame_sequence = 0
        logger.info(
            "SoundDevice adapter initialized: sample_rate=%d, channels=%d, blocksize=%d",
            audio_format.sample_rate,
            audio_format.channels,
            audio_format.blocksize,
        )

    def audio_format(self) -> AudioFormat:
        return self._format

    def is_running(self) -> bool:
        with self._lock:
            return self._stream is not None

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._frame_sequence = 0
            self._get_device()
            self._stream = sounddevice.InputStream(
                samplerate=self._format.sample_rate,
                channels=self._format.channels,
                blocksize=self._format.blocksize,
                dtype=self._format.dtype,
                callback=self._callback,
            )
            self._stream.start()
            logger.info("Audio stream started on device: %s", self._selected_device_name)

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            subscriber_count = len(self._subscribers)
            for sub in self._subscribers.values():
                sub.closed = True
                while True:
                    try:
                        sub.queue.get_nowait()
                    except queue.Empty:
                        break
            self._subscribers.clear()

        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            logger.exception("Error stopping audio stream")
        try:
            stream.close()
        except Exception:
            logger.exception("Error closing audio stream")

        logger.info("Audio stream stopped, cleared %d subscribers", subscriber_count)

    def subscribe(self, *, name: str, max_frames: int = 1024) -> AudioStreamReader:
        """Create a new subscriber that receives audio frames.

        Args:
            name: Identifier for this subscriber (for debugging/logging).
            max_frames: Maximum frames to buffer before dropping.

        Returns:
            AudioStreamReader that will receive frames.
        """
        maxsize = max(1, int(max_frames))
        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = _Subscriber(name=str(name), queue=queue.Queue(maxsize=maxsize))
        logger.info("New audio subscriber: name=%s, id=%d, max_frames=%d", name, subscriber_id, maxsize)
        return _QueueReader(self, subscriber_id)

    def _get_subscriber(self, subscriber_id: int) -> _Subscriber | None:
        with self._lock:
            return self._subscribers.get(subscriber_id)

    def _close_subscriber(self, subscriber_id: int) -> None:
        with self._lock:
            sub = self._subscribers.pop(subscriber_id, None)
            if sub is None:
                return
            sub.closed = True
            while True:
                try:
                    sub.queue.get_nowait()
                except queue.Empty:
                    break
        logger.info("Subscriber closed: name=%s, id=%d", sub.name, subscriber_id)

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Sounddevice callback - wraps raw audio in AudioFrame and distributes."""
        try:
            arr = indata.copy()
        except Exception:
            arr = np.array(indata, copy=True)

        with self._lock:
            seq = self._frame_sequence
            self._frame_sequence += 1
            subs = list(self._subscribers.values())

        frame = AudioFrame(data=arr, format=self._format, sequence=seq)

        for sub in subs:
            if sub.closed:
                continue
            try:
                sub.queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest frame if queue is full (backpressure)
                logger.debug("Frame dropped for subscriber %s due to backpressure", sub.name)

    def _get_device(self) -> None:
        """Select audio input device."""
        try:
            self._selected_device_name = str(sounddevice.query_devices(0).get("name"))
            index = sounddevice.query_devices('pulse').get("index")
            sounddevice.default.device = index, None
            logger.debug("Selected audio device: %s (index=%s)", self._selected_device_name, index)
        except Exception:
            logger.exception("Failed to select audio device, using system default")

