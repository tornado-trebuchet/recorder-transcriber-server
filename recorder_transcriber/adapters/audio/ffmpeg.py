import subprocess
import numpy as np
from uuid import uuid4
from pathlib import Path
from recorder_transcriber.domain.models import Recording


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
            raise
        
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

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdin_bytes = data.tobytes()
        _, stderr = proc.communicate(input=stdin_bytes)

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")

        recording.path = out_path
        recording.clear_data()
        return recording