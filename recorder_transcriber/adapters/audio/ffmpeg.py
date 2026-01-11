import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np

from recorder_transcriber.core.logger import get_logger
from recorder_transcriber.domain.models import Recording

logger = get_logger("adapters.audio.ffmpeg")


class AudioConverterAdapter:

    def __init__(
        self,
        *,
        ffmpeg_bin: str,
        input_format: str,
        output_codec: str,
        audio_format: str,
        dtype: str,
        tmp_dir: str | Path,
    ) -> None:
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.input_format = str(input_format)
        self.output_codec = str(output_codec)
        self.audio_format = str(audio_format)
        self.dtype = str(dtype)
        self.tmp_dir = Path(tmp_dir)

    def save_recording(self, recording: Recording) -> Recording:
        data = recording.data
        if data is None:
            logger.error("Cannot save recording: audio data is None")
            raise ValueError("Recording data is None, cannot save")

        logger.info(
            "Converting recording: sample_rate=%d, channels=%d, frames=%d",
            recording.sample_rate,
            recording.channels,
            data.shape[0] if data.ndim >= 1 else 0,
        )

        if data.ndim == 1:
            if recording.channels != 1:
                raise ValueError("Mono ndarray but recording.channels != 1")
        elif data.ndim == 2:
            if data.shape[1] == recording.channels:
                pass
            elif data.shape[0] == recording.channels:
                data = data.T  # (channels, frames) -> (frames, channels)
            else:
                raise ValueError("ndarray channels do not match recording.channels")
        else:
            raise ValueError("Recording.data must be 1D or 2D ndarray")

        np_dtype = np.dtype(self.dtype)
        data = data.astype(np_dtype, copy=False)

        ext = self.audio_format.lstrip(".")
        out_path = self.tmp_dir / f"rec-{uuid4().hex}.{ext}"

        ffmpeg_cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f", self.input_format,
            "-ar", str(recording.sample_rate),
            "-ac", str(recording.channels),
            "-i", "pipe:0",
            "-c:a", self.output_codec,
            str(out_path),
        ]

        logger.debug("Executing FFmpeg command: %s", " ".join(ffmpeg_cmd))

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdin_bytes = data.tobytes()
        _, stderr = proc.communicate(input=stdin_bytes)

        if proc.returncode != 0:
            logger.error("FFmpeg process failed with code %d: %s", proc.returncode, stderr.decode(errors="replace"))
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")

        logger.info("Recording saved successfully: %s", out_path)

        recording.path = out_path
        recording.clear_data()
        
        return recording