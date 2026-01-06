from typing import Protocol

from recorder_transcriber.domain.models import AudioFrame, WakeEvent


class WakeWordPort(Protocol):
    """Wake-word detection interface"""

    def detect(self, frame: AudioFrame) -> WakeEvent:
        """Check for wake-word in the given frame.

        Args:
            frame: AudioFrame to analyze. The implementation should use
                frame.to_mono_int16() for format conversion.

        Returns:
            WakeEvent with detection result and confidence scores
            for each configured wake-word model.
        """
        ...

    def reset(self) -> None:
        """Reset internal state.

        Call this after wake-word detection to prepare for the next
        detection cycle.
        """
        ...

    @property
    def active_models(self) -> list[str]:
        """Return list of active wake-word model names."""
        ...
