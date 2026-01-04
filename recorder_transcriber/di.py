from recorder_transcriber.config import config
from recorder_transcriber.adapters.audio.ffmpeg import AudioConverterAdapter
from recorder_transcriber.adapters.audio.sounddevice import AudioRecorderAdapter
from recorder_transcriber.adapters.speech_to_text.whisper import WhisperAdapter
from recorder_transcriber.adapters.text_to_text.localllm import TextEnhancer
from recorder_transcriber.services.recorder import RecorderService
from recorder_transcriber.services.transcriber import TranscriptionService
from recorder_transcriber.services.enhancer import EnhancementService


_recorder_service = RecorderService(
    capture_adapter=AudioRecorderAdapter(),
    storage_adapter=AudioConverterAdapter(),
    max_duration_seconds=config.recording_max_seconds,
)

_transcription_service = TranscriptionService(adapter=WhisperAdapter())
_enhancement_service = EnhancementService(adapter=TextEnhancer())


def get_recorder_service() -> RecorderService:
    return _recorder_service


def get_transcription_service() -> TranscriptionService:
    return _transcription_service


def get_enhancement_service() -> EnhancementService:
    return _enhancement_service