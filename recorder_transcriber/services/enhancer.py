from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from recorder_transcriber.model import Note, Transcript


class TextEnhancementPort(Protocol):
	"""Interface for downstream text enhancement."""

	def enhance(self, text: str) -> Note:
		...


@dataclass(slots=True)
class EnhancementService:
	"""Cleans transcripts and enriches them for downstream consumers."""

	adapter: TextEnhancementPort

	def enhance(self, transcript: Transcript) -> Note:
		if not transcript.text.strip():
			raise ValueError("Transcript text is empty")

		return self.adapter.enhance(transcript.text)