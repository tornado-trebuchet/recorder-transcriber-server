from functools import lru_cache

from recorder_transcriber.adapters.audio.ffmpeg import AudioConverterAdapter
from recorder_transcriber.adapters.audio.openww import OpenWakeWordAdapter
from recorder_transcriber.adapters.audio.sddevice import SoundDeviceAudioStreamAdapter
from recorder_transcriber.adapters.audio.silerovad import SileroVadAdapter
from recorder_transcriber.adapters.speech_to_text.whisper import WhisperAdapter
from recorder_transcriber.adapters.text_to_text.localllm import LangchainAdapter
from recorder_transcriber.core.settings import AppConfig, load_config
from recorder_transcriber.domain.models import AudioFormat
from recorder_transcriber.services.enhancement import EnhancementService
from recorder_transcriber.services.listening import ListenerService
from recorder_transcriber.services.recording import RecorderService
from recorder_transcriber.services.transcription import TranscriptionService


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get singleton AppConfig instance."""
    return load_config()


@lru_cache(maxsize=1)
def get_audio_format() -> AudioFormat:
    """Get AudioFormat from config."""
    cfg = get_config()
    return AudioFormat(
        sample_rate=cfg.audio.samplerate,
        channels=cfg.audio.channels,
        blocksize=cfg.audio.blocksize,
        dtype=cfg.audio.dtype,
    )


@lru_cache(maxsize=1)
def get_audio_stream() -> SoundDeviceAudioStreamAdapter:
    """Get singleton audio stream adapter."""
    return SoundDeviceAudioStreamAdapter(audio_format=get_audio_format())


@lru_cache(maxsize=1)
def get_storage_adapter() -> AudioConverterAdapter:
    """Get singleton audio storage/converter adapter."""
    cfg = get_config()
    return AudioConverterAdapter(
        ffmpeg_bin=cfg.ffmpeg.binary,
        input_format=cfg.ffmpeg.input_format,
        output_codec=cfg.ffmpeg.output_codec,
        audio_format=cfg.ffmpeg.audio_format,
        dtype=cfg.ffmpeg.dtype,
        tmp_dir=cfg.paths.tmp_dir,
    )


@lru_cache(maxsize=1)
def get_whisper_adapter() -> WhisperAdapter:
    """Get singleton Whisper STT adapter."""
    cfg = get_config()
    return WhisperAdapter(
        model_name=cfg.whisper.model,
        device=cfg.whisper.device,
        download_root=cfg.whisper.download_root,
        resource_management= cfg.whisper.resource_management,
        target_sample_rate=cfg.audio.samplerate,
    )


@lru_cache(maxsize=1)
def get_llm_adapter() -> LangchainAdapter:
    """Get singleton LLM adapter for enhancement."""
    cfg = get_config()
    return LangchainAdapter(
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        timeout=cfg.llm.timeout,
    )


@lru_cache(maxsize=1)
def get_recorder_service() -> RecorderService:
    """Get singleton RecorderService."""
    cfg = get_config()
    return RecorderService(
        stream=get_audio_stream(),
        storage_adapter=get_storage_adapter(),
        max_duration_seconds=cfg.recording.max_duration_seconds,
    )


@lru_cache(maxsize=1)
def get_transcription_service() -> TranscriptionService:
    """Get singleton TranscriptionService."""
    return TranscriptionService(adapter=get_whisper_adapter())


@lru_cache(maxsize=1)
def get_enhancement_service() -> EnhancementService:
    """Get singleton EnhancementService."""
    return EnhancementService(adapter=get_llm_adapter())


@lru_cache(maxsize=1)
def get_wakeword_adapter() -> OpenWakeWordAdapter:
    """Get singleton OpenWakeWord adapter for wake-word detection."""
    cfg = get_config()
    model_dir = cfg.listener.wake_model_dir.resolve()
    model_paths = [
        str(model_dir / f"{model}_v0.1.tflite") # this will backfire TODO: put as configurable parameter
        for model in cfg.listener.wake_models
    ]
    return OpenWakeWordAdapter(
        wakeword_models=model_paths,
        threshold=cfg.listener.wake_threshold,
        melspec_model_path=str(model_dir / cfg.listener.wake_melspec_model),
        embedding_model_path=str(model_dir / cfg.listener.wake_embedding_model),
    )


@lru_cache(maxsize=1)
def get_vad_adapter() -> SileroVadAdapter:
    """Get singleton Silero VAD adapter for voice activity detection."""
    cfg = get_config()
    return SileroVadAdapter(
        threshold=cfg.listener.vad_threshold,
        min_silence_duration_ms=cfg.listener.vad_min_silence_ms,
        speech_pad_ms=cfg.listener.vad_speech_pad_ms,
        sampling_rate=cfg.audio.samplerate,
    )


@lru_cache(maxsize=1)
def get_listener_service() -> ListenerService:
    """Get singleton ListenerService for voice-activated recording."""
    cfg = get_config()
    return ListenerService(
        stream=get_audio_stream(),
        wake=get_wakeword_adapter(),
        vad=get_vad_adapter(),
        config=cfg.listener,
        storage=get_storage_adapter(),
        stt=get_whisper_adapter(),
    )

