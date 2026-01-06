from typing import Protocol

from recorder_transcriber.domain.models import AudioFormat, AudioFrame


class AudioStreamReader(Protocol):
    """Read handle for consuming frames from an audio stream.

    Each reader maintains its own queue of frames, allowing multiple
    consumers (recorder, listener) to read from the same stream independently.
    """

    def read(self, timeout_seconds: float | None = None) -> AudioFrame | None:
        """Read the next frame from the stream.

        Args:
            timeout_seconds: Maximum time to wait for a frame.
                If None, blocks indefinitely.
                If 0, returns immediately (non-blocking).

        Returns:
            AudioFrame if available, None if timeout or stream closed.
        """
        ...

    def close(self) -> None:
        """Close this reader and release resources.

        After closing, read() will return None.
        """
        ...


class AudioStreamPort(Protocol):
    """Abstract audio capture source.

    Implementations capture audio from a device (microphone, file, etc.)
    and distribute frames to multiple subscribers via the pub-sub pattern.
    """

    def start(self) -> None:
        """Begin capturing audio from the device."""
        ...

    def stop(self) -> None:
        """Stop capturing and release device resources."""
        ...

    def is_running(self) -> bool:
        """Check if stream is actively capturing."""
        ...

    def audio_format(self) -> AudioFormat:
        """Return the stream's audio format configuration."""
        ...

    def subscribe(self, *, name: str, max_frames: int = 1024) -> AudioStreamReader:
        """Create a new consumer that receives frames.

        Args:
            name: Identifier for this subscriber (for debugging/logging).
            max_frames: Maximum frames to buffer before dropping oldest.

        Returns:
            AudioStreamReader that will receive all frames captured
            after subscription.
        """
        ...
