import subprocess
from pathlib import Path
from typing import Iterable

from recorder_transcriber.config import config


class AudioConverterAdapter:
    """Wraps ffmpeg CLI. Default output: WAV (pcm_s16le), 16 kHz, mono."""

    def __init__(self) -> None:
        ff = config.ffmpeg
        self.ffmpeg_bin: str = str(ff["binary"])
        self.input_format: str = str(ff["input_format"])
        self.sample_rate: int = int(ff["sample_rate"])
        self.channels: int = int(ff["channels"])
        self.extra_args: list[str] = [str(a) for a in ff.get("extra_args", [])]
        self.tmp_dir = Path(config.tmp_dir).expanduser()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def convert_file(self, src: str | Path, dst: str | Path | None = None) -> Path:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(src_path)

        dst_path = Path(dst) if dst is not None else self.tmp_dir / f"{src_path.stem}.wav"

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(src_path),
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-acodec",
            "pcm_s16le",
            *self.extra_args,
            str(dst_path),
        ]
        self._run(cmd)
        return dst_path

    def convert_bytes(self, pcm_bytes: bytes) -> bytes:
        """Convert raw PCM input to WAV bytes."""

        cmd = [
            self.ffmpeg_bin,
            "-f",
            self.input_format,
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            "-i",
            "-",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            *self.extra_args,
            "-",
        ]
        return self._run_capture(cmd, pcm_bytes)

    def convert_iter(self, frames: Iterable[bytes]) -> bytes:
        """Convert streamed PCM chunks to WAV bytes."""
        pcm_bytes = b"".join(frames)
        return self.convert_bytes(pcm_bytes)

    @staticmethod
    def _run(cmd: list[str]) -> None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")

    @staticmethod
    def _run_capture(cmd: list[str], stdin_bytes: bytes) -> bytes:
        proc = subprocess.run(
            cmd,
            input=stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")
        return proc.stdout