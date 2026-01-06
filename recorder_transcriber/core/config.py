from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from recorder_transcriber.domain.models import AudioDtype


class PathsConfig(BaseModel):

    tmp_dir: Path
    fs_dir: Path

    @field_validator("tmp_dir", "fs_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser()


class AudioConfig(BaseModel):
    """Audio stream configuration."""

    samplerate: int
    channels: int
    blocksize: int
    dtype: AudioDtype


class RecordingConfig(BaseModel):
    """Recording configuration."""

    max_duration_seconds: int = 300


class FFmpegConfig(BaseModel):
    """FFmpeg audio conversion configuration."""

    binary: str = "ffmpeg"
    input_format: str = "f32le"
    sample_rate: int = 16000
    channels: int = 1
    output_codec: str = "pcm_s16le"
    audio_format: str = "wav"
    dtype: AudioDtype = "float32"


class WhisperConfig(BaseModel):
    """Whisper STT configuration."""

    model: str
    device: str = "cpu"
    download_root: Path | None = None

    @field_validator("download_root", mode="before")
    @classmethod
    def expand_download_root(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v).expanduser()


class LLMConfig(BaseModel):
    """LLM configuration for text enhancement."""

    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int


class ListenerConfig(BaseModel):
    """Voice listener configuration for wake-word and VAD."""

    wake_window_seconds: float
    wake_frame_ms: int
    wake_threshold: float
    wake_models: list[str]
    wake_model_dir: Path
    wake_melspec_model: str
    wake_embedding_model: str
    vad_threshold: float
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    armed_timeout_seconds: float
    max_utterance_seconds: float
    end_hangover_ms: int

    @field_validator("wake_model_dir", mode="before")
    @classmethod
    def expand_wake_model_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser()


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: LogLevel = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    json_output: bool = False
    rotate_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    rotate_backup_count: int = 5


class AppConfig(BaseModel):
    """Application configuration loaded from YAML."""

    paths: PathsConfig
    audio: AudioConfig
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    whisper: WhisperConfig
    llm: LLMConfig
    listener: ListenerConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class Settings(BaseSettings):
    """Environment settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_path: Path

    @field_validator("config_path", mode="before")
    @classmethod
    def expand_config_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser()


def load_config() -> AppConfig:
    """The config file path is read from CONFIG_PATH environment variable."""
    settings = Settings() # type: ignore 

    if not settings.config_path.exists():
        raise FileNotFoundError(f"Config file not found: {settings.config_path}")

    with settings.config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = AppConfig.model_validate(raw)
    config.paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    config.paths.fs_dir.mkdir(parents=True, exist_ok=True)

    return config


