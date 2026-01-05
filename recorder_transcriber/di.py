from dataclasses import dataclass
from typing import cast

from fastapi import Request

from recorder_transcriber.configuration import Config

from recorder_transcriber.adapters.audio.ffmpeg import AudioConverterAdapter
from recorder_transcriber.adapters.audio.sddevice import AudioRecorderAdapter
from recorder_transcriber.adapters.speech_to_text.whisper import WhisperAdapter
from recorder_transcriber.adapters.text_to_text.localllm import TextEnhancer
from recorder_transcriber.adapters.audio.openww import OpenWakeWordAdapter
from recorder_transcriber.adapters.audio.silerovad import SileroVadAdapter

from recorder_transcriber.services.recorder import RecorderService
from recorder_transcriber.services.orchestrator import OrchestratorConfig, OrchestratorService
from recorder_transcriber.services.transcriber import TranscriptionService
from recorder_transcriber.services.enhancer import EnhancementService


@dataclass(slots=True)
class AppContainer:
    config: Config
    recorder_service: RecorderService
    transcription_service: TranscriptionService
    enhancement_service: EnhancementService
    _orchestrator_service: OrchestratorService | None = None

    def get_orchestrator_service(self) -> OrchestratorService:
        if self._orchestrator_service is not None:
            return self._orchestrator_service

        listener_cfg = self.config.listener
        orchestrator_stream = AudioRecorderAdapter(
            samplerate=16000,
            channels=1,
            blocksize=512,
            dtype="float32",
            queue_max_chunks=max(8, int((float(listener_cfg["wake_window_seconds"]) * 16000) // 512) * 8),
        )

        self._orchestrator_service = OrchestratorService(
            recorder_service=self.recorder_service,
            storage_adapter=self.recorder_service.storage_adapter,
            stream=orchestrator_stream,
            wake=OpenWakeWordAdapter(
                wakeword_models=(listener_cfg["wake_models"] or None),
                threshold=float(listener_cfg["wake_threshold"]),
            ),
            vad=SileroVadAdapter(
                sampling_rate=16000,
                threshold=float(listener_cfg["vad_threshold"]),
                min_silence_duration_ms=int(listener_cfg["vad_min_silence_ms"]),
                speech_pad_ms=int(listener_cfg["vad_speech_pad_ms"]),
            ),
            config=OrchestratorConfig(
                wake_window_seconds=float(listener_cfg["wake_window_seconds"]),
                wake_frame_ms=int(listener_cfg["wake_frame_ms"]),
                wake_threshold=float(listener_cfg["wake_threshold"]),
                wake_models=(listener_cfg["wake_models"] or None),
                vad_threshold=float(listener_cfg["vad_threshold"]),
                vad_min_silence_ms=int(listener_cfg["vad_min_silence_ms"]),
                vad_speech_pad_ms=int(listener_cfg["vad_speech_pad_ms"]),
                armed_timeout_seconds=float(listener_cfg["armed_timeout_seconds"]),
                max_utterance_seconds=float(listener_cfg["max_utterance_seconds"]),
                end_hangover_ms=int(listener_cfg["end_hangover_ms"]),
            ),
        )
        return self._orchestrator_service

    def shutdown(self) -> None:
        svc = self._orchestrator_service
        if svc is None:
            return
        try:
            svc.stop()
        except Exception:
            pass


def build_container(cfg: Config) -> AppContainer:
    audio_cfg = cfg.audio
    ffmpeg_cfg = cfg.ffmpeg
    whisper_cfg = cfg.whisper
    llm_cfg = cfg.llm

    capture = AudioRecorderAdapter(
        samplerate=int(audio_cfg["samplerate"]),
        channels=int(audio_cfg["channels"]),
        blocksize=int(audio_cfg["blocksize"]),
        dtype=str(audio_cfg["dtype"]),
    )
    storage = AudioConverterAdapter(
        ffmpeg_bin=str(ffmpeg_cfg["binary"]),
        input_format=str(ffmpeg_cfg["input_format"]),
        output_codec=str(ffmpeg_cfg["output_codec"]),
        audio_format=str(ffmpeg_cfg["audio_format"]),
        dtype=str(ffmpeg_cfg["dtype"]),
        tmp_dir=str(cfg.tmp_dir),
    )
    recorder_service = RecorderService(
        capture_adapter=capture,
        storage_adapter=storage,
        max_duration_seconds=int(cfg.recording_max_seconds),
    )

    transcription_service = TranscriptionService(
        adapter=WhisperAdapter(
            model_name=str(whisper_cfg["model"]),
            device=str(whisper_cfg.get("device", "gpu")),
            download_root=cast(str | None, whisper_cfg.get("download_root")),
            target_sample_rate=int(audio_cfg.get("samplerate", 16000)),
        )
    )
    enhancement_service = EnhancementService(
        adapter=TextEnhancer(
            base_url=str(llm_cfg["base_url"]),
            model=str(llm_cfg["model"]),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            timeout=int(llm_cfg.get("timeout", 160)),
        )
    )

    return AppContainer(
        config=cfg,
        recorder_service=recorder_service,
        transcription_service=transcription_service,
        enhancement_service=enhancement_service,
    )


def _container_from_request(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("App container not initialized")
    return cast(AppContainer, container)


def get_recorder_service(request: Request) -> RecorderService:
    return _container_from_request(request).recorder_service


def get_transcription_service(request: Request) -> TranscriptionService:
    return _container_from_request(request).transcription_service


def get_enhancement_service(request: Request) -> EnhancementService:
    return _container_from_request(request).enhancement_service


def get_orchestrator_service(request: Request) -> OrchestratorService:
    return _container_from_request(request).get_orchestrator_service()