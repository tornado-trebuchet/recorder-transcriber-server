from typing import Protocol

from recorder_transcriber.domain.models import Recording, Transcript


class SpeechToTextPort(Protocol):

	def transcribe_recording(self, recording: Recording) -> Transcript:
		...

	def start(self) -> None:
		...

	def stop(self) -> None:
		...
