from dataclasses import dataclass

from recorder_transcriber.domain.models import Note, Transcript
from recorder_transcriber.ports.ttt import TextToTextPort


@dataclass(slots=True)
class EnhancementService:

	adapter: TextToTextPort

	def enhance(self, transcript: Transcript) -> Note:
		if not transcript.text.strip():
			raise ValueError("Transcript text is empty")

		return self.adapter.enhance(transcript)