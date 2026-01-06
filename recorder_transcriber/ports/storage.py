"""Port for audio storage operations."""

from typing import Protocol

from recorder_transcriber.domain.models import Recording


class AudioStoragePort(Protocol):
    """Port for persisting audio recordings."""

    def save_recording(self, recording: Recording) -> Recording:
        """Save recording to storage and return updated Recording with path."""
        ...
