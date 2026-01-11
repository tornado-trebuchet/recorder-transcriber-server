from typing import Protocol

from recorder_transcriber.domain.models import Recording


class AudioStoragePort(Protocol):

    def save_recording(self, recording: Recording) -> Recording:
        ...
