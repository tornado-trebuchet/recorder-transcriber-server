from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import time

import numpy as np


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


AudioDtype = Literal["float32", "int16", "float64"]


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Describes audio stream parameters. Single source of truth for frame config."""

    sample_rate: int
    channels: int
    blocksize: int
    dtype: AudioDtype

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {self.blocksize}")


@dataclass(slots=True)
class AudioFrame:
    """A single chunk of audio with metadata. Canonical frame unit for all processing."""

    data: np.ndarray
    format: AudioFormat
    timestamp_ns: int = field(default_factory=lambda: time.monotonic_ns())
    sequence: int = 0

    def to_mono_float32(self) -> np.ndarray:
        """Convert to normalized mono float32 for ML models (e.g., SileroVAD).

        This is the single conversion point - adapters should call this method
        rather than implementing their own conversion logic.
        """
        arr = self.data
        # Convert stereo to mono by averaging channels
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        # Normalize int16 to [-1.0, 1.0] range
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr

    def to_mono_int16(self) -> np.ndarray:
        """Convert to mono int16 for wake-word models (e.g., OpenWakeWord).

        This is the single conversion point - adapters should call this method
        rather than implementing their own conversion logic.
        """
        mono = self.to_mono_float32()
        return (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)

    @property
    def num_samples(self) -> int:
        """Number of samples in this frame."""
        if self.data.ndim == 2:
            return self.data.shape[0]
        return self.data.shape[0]

    @property
    def duration_seconds(self) -> float:
        """Duration of this frame in seconds."""
        return self.num_samples / self.format.sample_rate


@dataclass(slots=True)
class Recording:
    """Normalized audio capture or conversion output."""
    data: np.ndarray | None
    path: Path | None
    sample_rate: int
    channels: int
    dtype: str
    blocksize: int | None
    device_name: str | None
    captured_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if self.data is None and self.path is None:
            raise ValueError("Recording requires either in-memory data or a filesystem path")

    def clear_data(self) -> None:
        """Set bytes free when not needed."""
        if self.path is None:
            raise ValueError("Cannot drop payload without a backing path")
        self.data = None


@dataclass(slots=True)
class Transcript:
    """Speech-to-text result consumed by downstream services."""

    text: str
    recording_path: Path | None = None
    generated_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class Note:
    """Enhanced transcript packaged for text-to-text consumers."""

    body: str
    title: str
    tags: list[str]
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class WakeEvent:
	detected: bool
	scores: dict[str, float]

@dataclass(frozen=True, slots=True)
class VadEvent:
	detected: bool


@dataclass(frozen=True, slots=True)
class ListeningResult:
    """Result from a completed listening session."""

    recording: Recording
    transcript: Transcript