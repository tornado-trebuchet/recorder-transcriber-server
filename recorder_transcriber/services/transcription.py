from dataclasses import dataclass

from recorder_transcriber.domain.models import Recording, Transcript
from recorder_transcriber.ports.stt import SpeechToTextPort


@dataclass(slots=True)
class TranscriptionService:

	adapter: SpeechToTextPort

	def transcribe(self, recording: Recording) -> Transcript:
		if recording.path is None and recording.data is None:
			raise ValueError("Recording must include audio data or saved path")
		transcript = self.adapter.transcribe_recording(recording)
		if transcript.recording_path is None and recording.path is not None:
			transcript.recording_path = recording.path
		return transcript