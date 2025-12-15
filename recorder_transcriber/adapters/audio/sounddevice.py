import logging
import queue
from typing import Any

import numpy as np
import sounddevice as sd # type: ignore

from recorder_transcriber.config import config


class AudioRecorderAdapter:
    """
    Device selection: prefers the first Pulse monitor, otherwise default input.
    """

    def __init__(
        self,
        samplerate: int = config.audio["samplerate"],
        channels: int = config.audio["channels"],
        blocksize: int = config.audio["blocksize"],
        dtype: str = config.audio["dtype"],
        device: int | str | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self._requested_device = device

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.RawInputStream | None = None
        self._selected_device: int | str | None = None

    def start(self) -> None:
        if self._stream:
            return

        device = self._requested_device or self._auto_select_device()
        self._selected_device = device

        self._stream = sd.RawInputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype=self.dtype,
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if not self._stream:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._queue.mutex:
            self._queue.queue.clear()

    def close(self) -> None:
        self.stop()

    @property
    def info(self) -> dict[str, int | str | None]:
        return {
            "samplerate": self.samplerate,
            "channels": self.channels,
            "blocksize": self.blocksize,
            "dtype": self.dtype,
            "device": self._selected_device,
        }

    def _callback(self, indata: bytes, frames: int, time: Any, status: Any) -> None:
        if status:
            logging.warning("sounddevice status: %s", status)
        np_dtype = np.dtype(self.dtype)
        array = np.frombuffer(indata, dtype=np_dtype).reshape(-1, self.channels)
        try:
            self._queue.put_nowait(array)
        except queue.Full:
            logging.warning("Audio queue full; dropping %s frames", frames)

    @staticmethod
    def _auto_select_device() -> int | str | None:
        try:
            devices: list[dict[str, Any]] = sd.query_devices() # type: ignore
        except Exception:
            return None

        monitors = [
            idx
            for idx, dev in enumerate(devices)
            if dev.get("max_input_channels", 0) > 0
            and "monitor" in str(dev.get("name", "")).lower()
        ]
        if monitors:
            return monitors[0]

        default_device = sd.default.device
        candidate = default_device[0]
        if candidate is not None:
            return candidate
        if isinstance(default_device, (int, str)):
            return default_device

        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                return idx
        return None
