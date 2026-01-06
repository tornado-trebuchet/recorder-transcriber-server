import os
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import dotenv
import yaml


class Config:
    def __init__(self) -> None:
        self.llm_server: str = ""
        self.audio: dict[str, Any] = {}
        self.recording: dict[str, Any] = {}
        self.recording_max_seconds: int = 0
        self.ffmpeg: dict[str, Any] = {}
        self.whisper: dict[str, Any] = {}
        self.llm: dict[str, Any] = {}
        self.listener: dict[str, Any] = {}

        self.config_path: Path | None = None
        self.config: dict[str, Any] = {}
        self.config_dir: str = ""
        self.fsdir: str = ""
        self.tmp_dir: str = ""

    def load_core(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dotenv.load_dotenv(repo_root / ".env")

        self.llm_server = self._require_env("LLM_SERVER")

        self.config_path = Path(self._require_env("CONFIG_PATH")).expanduser()
        self.config = self._load_yaml(self.config_path)
        self.config_dir = str(self.config_path.parent)

        self.fsdir = self._require_env("FS_DIR")

        tmp_path = Path(self._require_env("TMP_DIR")).expanduser()
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(tmp_path)

        self.load_audio()
        self.load_stt()
        self.load_ttt()

    def load_audio(self) -> None:
        raw = self.config
        self._load_audio_params(raw)
        self._load_recording(raw)
        self._load_ffmpeg(raw)
        self._load_listener(raw)

    def _load_audio_params(self, raw: Mapping[str, Any]) -> None:
        audio = self._require_mapping(raw, "audio")
        self.audio = {
            "samplerate": int(self._require_value(audio, "samplerate")),
            "channels": int(self._require_value(audio, "channels")),
            "blocksize": int(self._require_value(audio, "blocksize")),
            "dtype": str(self._require_value(audio, "dtype")),
        }

    def _load_recording(self, raw: Mapping[str, Any]) -> None:
        recording = self._require_mapping(raw, "recording")
        self.recording = {
            "max_duration_seconds": int(self._require_value(recording, "max_duration_seconds"))
        }
        self.recording_max_seconds = int(self.recording["max_duration_seconds"])

    def _load_ffmpeg(self, raw: Mapping[str, Any]) -> None:
        ffmpeg = self._require_mapping(raw, "ffmpeg")
        self.ffmpeg = {
            "binary": str(self._require_value(ffmpeg, "binary")),
            "input_format": str(self._require_value(ffmpeg, "input_format")),
            "sample_rate": int(self._require_value(ffmpeg, "sample_rate")),
            "channels": int(self._require_value(ffmpeg, "channels")),
            "output_codec": str(self._require_value(ffmpeg, "output_codec")),
            "audio_format": str(self._require_value(ffmpeg, "audio_format")),
            "dtype": str(self._require_value(ffmpeg, "dtype")),
        }

    def _load_listener(self, raw: Mapping[str, Any]) -> None:
        listener_raw = self._get_optional_mapping(raw, "listener")
        listener = dict(cast(Mapping[str, Any], listener_raw))
        wake_models = self._require_optional_sequence_of_str(listener, "wake_models")
        wake_download_models = self._require_optional_sequence_of_str(listener, "wake_download_models")
        wake_model_dir = self._expand_path_optional(listener, "wake_model_dir")
        self.listener = {
            "wake_window_seconds": float(listener.get("wake_window_seconds", 2.0)),
            "wake_frame_ms": int(listener.get("wake_frame_ms", 80)),
            "wake_threshold": float(listener.get("wake_threshold", 0.5)),
            "wake_models": wake_models,
            "wake_model_dir": wake_model_dir,
            "wake_download_models": wake_download_models,
            "vad_threshold": float(listener.get("vad_threshold", 0.5)),
            "vad_min_silence_ms": int(listener.get("vad_min_silence_ms", 250)),
            "vad_speech_pad_ms": int(listener.get("vad_speech_pad_ms", 30)),
            "armed_timeout_seconds": float(listener.get("armed_timeout_seconds", 5.0)),
            "max_utterance_seconds": float(listener.get("max_utterance_seconds", 20.0)),
            "end_hangover_ms": int(listener.get("end_hangover_ms", 300)),
        }

    def load_stt(self) -> None:
        raw = self.config
        whisper = self._require_mapping(raw, "whisper")
        download_root_raw = whisper.get("download_root")
        download_root_expanded = str(Path(download_root_raw).expanduser()) if download_root_raw else None
        self.whisper = {
            "model": str(self._require_value(whisper, "model")),
            "device": str(whisper.get("device", "gpu")),
            "download_root": download_root_expanded,
        }

    def load_ttt(self) -> None:
        raw = self.config
        llm = self._require_mapping(raw, "llm")
        self.llm = {
            "base_url": str(self._require_value(llm, "base_url")),
            "model": str(self._require_value(llm, "model")),
            "temperature": float(llm.get("temperature", 0.2)),
            "max_tokens": int(llm.get("max_tokens", 800)),
            "timeout": int(llm.get("timeout", 30)),
        }

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        return cast(dict[str, Any], data)

    @staticmethod
    def _require_mapping(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
        if key not in raw:
            raise ValueError(f"Missing required section '{key}' in config")
        value = raw[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"Section '{key}' must be a mapping")
        return dict(cast(Mapping[str, Any], value))

    @staticmethod
    def _require_value(raw: Mapping[str, Any], key: str) -> Any:
        if key not in raw:
            raise ValueError(f"Missing required value '{key}' in config")
        return raw[key]

    @staticmethod
    def _require_sequence(raw: Mapping[str, Any], key: str) -> list[Any]:
        if key not in raw:
            raise ValueError(f"Missing required sequence '{key}' in config")
        value = raw[key]
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError(f"'{key}' must be a sequence")
        seq = cast(Sequence[Any], value)
        return list(seq)

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if value is None or value == "":
            raise ValueError(f"Missing required env var: {name}")
        return value

    @staticmethod
    def _get_optional_mapping(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
        v = raw.get(key, {})
        if v is None:
            return {}
        if not isinstance(v, Mapping):
            raise ValueError(f"Section '{key}' must be a mapping")
        return dict(cast(Mapping[str, Any], v))

    @staticmethod
    def _require_optional_sequence_of_str(raw: Mapping[str, Any], key: str) -> list[str]:
        raw_val = raw.get(key, [])
        if raw_val is None:
            return []
        if isinstance(raw_val, str) or not isinstance(raw_val, Sequence):
            raise ValueError(f"{key} must be a sequence of strings")
        return [str(x) for x in cast(Sequence[Any], raw_val)]

    @staticmethod
    def _expand_path_optional(raw: Mapping[str, Any], key: str) -> str:
        v = raw.get(key)
        return str(Path(str(v)).expanduser()) if v else ""


def load_config() -> Config:
    return Config()


