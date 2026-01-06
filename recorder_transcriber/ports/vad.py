from typing import Protocol

from recorder_transcriber.domain.models import AudioFrame, VadEvent


class VadPort(Protocol):
    """Voice Activity Detection interface.

    Implementations analyze audio frames to detect speech activity.
    State transitions (speech start/end) are signaled via VadEvent.
    """

    def process(self, frame: AudioFrame) -> VadEvent | None:
        """Process a frame and detect voice activity.

        Args:
            frame: AudioFrame to analyze. The implementation should use
                frame.to_mono_float32() for format conversion.

        Returns:
            VadEvent on state transitions (speech started/ended),
            None if no state change.
        """
        ...

    def reset(self) -> None:
        """Reset internal state.

        Call this when starting a new detection session or after
        processing a complete utterance.
        """
        ...

    @property
    def required_frame_size(self) -> int | None:
        """Return required frame size in samples, or None if flexible.

        Some VAD models (e.g., SileroVAD) require fixed frame sizes.
        If not None, callers should buffer frames to this size.
        """
        ...
