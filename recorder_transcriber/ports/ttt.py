from typing import Protocol

from recorder_transcriber.domain.models import Note, Transcript


class TextToTextPort(Protocol):

	def enhance(self, text: Transcript) -> Note:
		...


