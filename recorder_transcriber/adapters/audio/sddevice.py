from typing import Any

import queue
import numpy as np
import sounddevice

from recorder_transcriber.config import config
from recorder_transcriber.model import Recording

class AudioRecorderAdapter:

    def __init__(
        self,
        samplerate: int = config.audio["samplerate"],
        channels: int = config.audio["channels"],
        blocksize: int = config.audio["blocksize"],
        dtype: str = config.audio["dtype"],
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sounddevice.InputStream | None = None
        self._selected_device_name: str | None = None

    def start(self) -> None:
        if self._stream:
            return

        self._get_device()

        self._stream = sounddevice.InputStream(
            callback=self._callback,
        )
        self._stream.start()
        print("starting")

    def stop(self) -> Recording | None:
        """Stop recording, drain queued frames and return a Recording"""
        input_samplerate: float | None = None
        stream = self._stream
        if stream:
            input_samplerate = getattr(stream, "samplerate", None)
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self._stream = None

        payload = self._drain_queue()
        if payload is None:
            return None
        payload = self._resample(payload, input_samplerate)
        return self._build_recording(payload)

    def _build_recording(self, payload: np.ndarray) -> Recording:
        return Recording(
            data=payload,
            path=None,
            sample_rate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            device_name=self._selected_device_name,
        )

    def _drain_queue(self) -> np.ndarray | None:
        """Collect all queued chunks and return a single ndarray"""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not chunks:
            return None

        try:
            return np.vstack(chunks)
        except ValueError:
            return np.concatenate([np.atleast_2d(c) for c in chunks], axis=0)

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        try:
            array = indata.copy()
        except Exception:
            array = np.array(indata, copy=True)
        try:
            self._queue.put_nowait(array)
        except queue.Full:
            pass

    def _resample(self, payload: np.ndarray, source_samplerate: float | None = None) -> np.ndarray:
        """Match ndarray to configured dtype, channels, and samplerate."""
        if payload.size == 0:
            return payload.astype(self.dtype, copy=False)

        if payload.ndim == 1:
            working = payload.reshape(-1, 1)
        elif payload.ndim == 2:
            working = payload
        else:
            raise ValueError("Audio payload must be 1D or 2D")

        target_channels = int(self.channels)
        if target_channels < 1:
            raise ValueError("Configured channel count must be >= 1")

        if working.shape[1] == target_channels:
            channel_matched = working
        elif target_channels == 1:
            channel_matched = working.mean(axis=1, keepdims=True)
        elif working.shape[1] == 1:
            channel_matched = np.repeat(working, target_channels, axis=1)
        else:
            mono = working.mean(axis=1, keepdims=True)
            channel_matched = np.repeat(mono, target_channels, axis=1)

        target_samplerate = float(self.samplerate)
        source_rate = float(source_samplerate) if source_samplerate else target_samplerate

        if source_rate <= 0:
            source_rate = target_samplerate

        if abs(source_rate - target_samplerate) <= 1e-6:
            resampled = channel_matched
        else:
            duration = channel_matched.shape[0] / source_rate
            if duration == 0:
                resampled = channel_matched
            else:
                target_length = max(int(round(duration * target_samplerate)), 1)
                old_times = np.linspace(0.0, duration, num=channel_matched.shape[0], endpoint=False, dtype=np.float64)
                new_times = np.linspace(0.0, duration, num=target_length, endpoint=False, dtype=np.float64)
                resampled = np.empty((target_length, channel_matched.shape[1]), dtype=np.float64)
                for idx in range(channel_matched.shape[1]):
                    resampled[:, idx] = np.interp(new_times, old_times, channel_matched[:, idx])

        target_dtype = np.dtype(self.dtype)
        cast_ready = resampled.astype(np.float64, copy=False)
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            cast_ready = np.clip(cast_ready, info.min, info.max)

        output = cast_ready.astype(target_dtype, copy=False)
        if target_channels == 1:
            return output.reshape(-1)
        return output

    def _get_device(self) -> None :
        """ Look here for a bug"""
        self._selected_device_name = str(sounddevice.query_devices(0).get("name"))
        index = sounddevice.query_devices('pulse').get("index")
        sounddevice.default.device = index, None