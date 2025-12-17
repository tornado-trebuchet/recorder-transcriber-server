import os
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import dotenv
import yaml


class Config:
    """Loads env values and structured settings from an external config file."""

    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        dotenv.load_dotenv(repo_root / ".env")

        config_path_env = os.getenv("CONFIG_PATH")

        if config_path_env:
            config_path = Path(config_path_env).expanduser()
        else:
            raise ValueError("CONFIG_PATH not set")

        raw = self._load_yaml(config_path)

        self.config_dir: str = self._require_env("CONFIG_PATH")
        self.fsdir: str = self._require_env("FS_DIR")
        self.llm_server: str = self._require_env("LLM_SERVER")
        self.server_addr: str = self._require_env("SERVER_ADDR")
        self.server_port: int = int(self._require_env("SERVER_PORT"))

        tmp_dir_env = self._require_env("TMP_DIR")
        tmp_path = Path(tmp_dir_env).expanduser()
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(tmp_path)

        audio = self._require_mapping(raw, "audio")
        self.audio: dict[str, Any] = {
            "samplerate": int(self._require_value(audio, "samplerate")),
            "channels": int(self._require_value(audio, "channels")),
            "blocksize": int(self._require_value(audio, "blocksize")),
            "dtype": str(self._require_value(audio, "dtype")),
        }

        recording = self._require_mapping(raw, "recording")
        self.recording: dict[str, Any] = {
            "max_duration_seconds": int(self._require_value(recording, "max_duration_seconds"))
        }
        self.recording_max_seconds: int = int(self.recording["max_duration_seconds"])

        ffmpeg = self._require_mapping(raw, "ffmpeg")
        self.ffmpeg: dict[str, Any] = {
            "binary": str(self._require_value(ffmpeg, "binary")),
            "input_format": str(self._require_value(ffmpeg, "input_format")),
            "sample_rate": int(self._require_value(ffmpeg, "sample_rate")),
            "channels": int(self._require_value(ffmpeg, "channels")),
            "output_codec": str(self._require_value(ffmpeg, "output_codec")),
            "audio_format": str(self._require_value(ffmpeg, "audio_format")),
            "dtype": str(self._require_value(ffmpeg, "dtype")),
        }

        whisper = self._require_mapping(raw, "whisper")
        self.whisper: dict[str, Any] = {
            "model": str(self._require_value(whisper, "model")),
            "device": str(whisper.get("device", "cpu")),
            "download_root": str(whisper.get("download_root")),
        }

        llm = self._require_mapping(raw, "llm")
        self.llm: dict[str, Any] = {
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


config = Config()


