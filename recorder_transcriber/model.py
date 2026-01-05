from typing import Literal
from dataclasses import dataclass, field

from datetime import datetime, timezone
from pathlib import Path
import numpy as np

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

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
    # language: str | None = None
    generated_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class Note:
    """Enhanced transcript packaged for text-to-text consumers."""

    body: str
    title: str
    tags: list[str]
    created_at: datetime = field(default_factory=_utcnow)

@dataclass(frozen=True, slots=True)
class WakeResult:
	detected: bool
	scores: dict[str, float]

VadEventType = Literal["speech_start", "speech_end"]

@dataclass(frozen=True, slots=True)
class VadEvent:
	kind: VadEventType