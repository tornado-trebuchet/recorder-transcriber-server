from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from recorder_transcriber.model import Recording, Transcript


class SpeechToTextPort(Protocol):
	"""Interface for speech-to-text models."""

	def transcribe_recording(self, recording: Recording) -> Transcript:
		...


@dataclass(slots=True)
class TranscriptionService:
	"""Wraps the speech-to-text adapter behind a simple service boundary."""

	adapter: SpeechToTextPort

	def transcribe(self, recording: Recording) -> Transcript:
		if recording.path is None and recording.data is None:
			raise ValueError("Recording must include audio data or saved path")
		transcript = self.adapter.transcribe_recording(recording)
		if transcript.recording_path is None and recording.path is not None:
			transcript.recording_path = recording.path
		return transcript
