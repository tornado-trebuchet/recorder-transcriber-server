from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Callable
import time

import numpy as np

from recorder_transcriber.core.settings import ListenerConfig
from recorder_transcriber.domain.models import AudioFrame, ListeningResult, Recording
from recorder_transcriber.ports.audiostream import AudioStreamPort, AudioStreamReader
from recorder_transcriber.ports.vad import VadPort
from recorder_transcriber.ports.wakeword import WakeWordPort
from recorder_transcriber.ports.stt import SpeechToTextPort
from recorder_transcriber.ports.storage import AudioStoragePort


@dataclass(slots=True)
class ListenerService:
    """Orchestrates wake-word detection and VAD for voice-activated recording.

    State machine:
    - IDLE: Waiting for wake-word
    - ARMED: Wake-word detected, waiting for speech
    - LISTENING: Speech detected, capturing utterance
    """

    stream: AudioStreamPort
    wake: WakeWordPort
    vad: VadPort
    config: ListenerConfig # FIXME: should not be here 
    storage: AudioStoragePort | None = None
    stt: SpeechToTextPort | None = None

    _lock: Lock = field(default_factory=Lock, init=False)
    _reader: AudioStreamReader | None = field(default=None, init=False)
    _thread: Thread | None = field(default=None, init=False)
    _stop: Event = field(default_factory=Event, init=False)
    _state: str = field(default="IDLE", init=False)
    _on_utterance: Callable[[list[AudioFrame]], None] | None = field(default=None, init=False)
    _result: ListeningResult | None = field(default=None, init=False)
    _result_ready: Event = field(default_factory=Event, init=False)

    def start(self, on_utterance: Callable[[list[AudioFrame]], None] | None = None) -> None:
        """Start listening for wake-words.

        Args:
            on_utterance: Callback invoked with captured frames when an
                utterance is complete.
        """
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Listening session already active")
            if not self.stream.is_running():
                raise RuntimeError("Audio stream is not running")
            self._stop.clear()
            self._result_ready.clear()
            self._result = None
            self._on_utterance = on_utterance
            self._reader = self.stream.subscribe(name="listener", max_frames=4096)
            self._thread = Thread(target=self._run, name="listener-loop", daemon=True)
            self._thread.start()

    def stop(self) -> ListeningResult | None:
        """Stop listening and release resources.

        Returns:
            ListeningResult if an utterance was captured and processed,
            None otherwise.
        """
        with self._lock:
            thread = self._thread
            reader = self._reader
            self._stop.set()

        if thread:
            thread.join(timeout=5.0)

        with self._lock:
            self._thread = None
            self._reader = None
            self._state = "IDLE"
            self._on_utterance = None
            result = self._result
            self._result = None

        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

        self.wake.reset()
        self.vad.reset()
        return result

    def wait_for_result(self, timeout_seconds: float | None = None) -> ListeningResult | None:
        """Block until an utterance is captured and processed, or timeout.

        Args:
            timeout_seconds: Maximum time to wait. None waits indefinitely.

        Returns:
            ListeningResult if available, None if timed out.
        """
        if self._result_ready.wait(timeout=timeout_seconds):
            with self._lock:
                return self._result
        return None

    def is_listening(self) -> bool:
        """Check if the listener is active."""
        with self._lock:
            return self._thread is not None

    @property
    def state(self) -> str:
        """Current state: IDLE, ARMED, or LISTENING."""
        with self._lock:
            return self._state

    def _run(self) -> None:
        """Main listening loop - processes frames through wake-word and VAD."""
        reader = self._reader
        if reader is None:
            return

        fmt = self.stream.audio_format()
        utterance_frames: list[AudioFrame] = []
        pre_roll_frames: list[AudioFrame] = []
        hangover_frames: list[AudioFrame] = []

        # Calculate buffer sizes based on config
        frames_per_second = fmt.sample_rate / fmt.blocksize
        pre_roll_max = max(1, int(self.config.vad_speech_pad_ms / 1000 * frames_per_second) + 5)
        hangover_max = max(1, int(self.config.end_hangover_ms / 1000 * frames_per_second))
        max_utterance_frames = int(self.config.max_utterance_seconds * frames_per_second)

        armed_start_time: float | None = None
        speech_ended = False

        while not self._stop.is_set():
            frame = reader.read(timeout_seconds=0.1)
            if frame is None:
                continue

            with self._lock:
                current_state = self._state

            if current_state == "IDLE":
                # Check for wake-word
                wake_result = self.wake.detect(frame)
                if wake_result.detected:
                    with self._lock:
                        self._state = "ARMED"
                    # Reset VAD state for fresh speech detection
                    self.vad.reset()
                    pre_roll_frames.clear()
                    armed_start_time = time.monotonic()

            elif current_state == "ARMED":
                # Check for armed timeout
                if armed_start_time is not None:
                    elapsed = time.monotonic() - armed_start_time
                    if elapsed > self.config.armed_timeout_seconds:
                        # Timeout - return to IDLE
                        with self._lock:
                            self._state = "IDLE"
                        pre_roll_frames.clear()
                        armed_start_time = None
                        self.wake.reset()
                        self.vad.reset()
                        continue

                # Buffer frames for pre-roll
                pre_roll_frames.append(frame)
                if len(pre_roll_frames) > pre_roll_max:
                    pre_roll_frames.pop(0)

                # Check for speech start
                vad_event = self.vad.process(frame)
                if vad_event is not None and vad_event.detected:
                    with self._lock:
                        self._state = "LISTENING"
                    # Include pre-roll buffer in utterance
                    utterance_frames = pre_roll_frames.copy()
                    pre_roll_frames.clear()
                    armed_start_time = None
                    speech_ended = False
                    hangover_frames.clear()

            elif current_state == "LISTENING":
                # Accumulate frames
                utterance_frames.append(frame)

                # Check for max utterance duration
                if len(utterance_frames) >= max_utterance_frames:
                    # Force end recording
                    self._process_utterance(utterance_frames)
                    utterance_frames = []
                    hangover_frames.clear()
                    speech_ended = False
                    with self._lock:
                        self._state = "IDLE"
                    self.wake.reset()
                    self.vad.reset()
                    continue

                # Check for speech end
                vad_event = self.vad.process(frame)
                if vad_event is not None and not vad_event.detected:
                    speech_ended = True
                    hangover_frames = [frame]

                # If speech ended, collect hangover frames
                if speech_ended:
                    # Use sequence comparison to avoid numpy array truth value error
                    if not any(f.sequence == frame.sequence for f in hangover_frames):
                        hangover_frames.append(frame)
                    if len(hangover_frames) >= hangover_max:
                        # Hangover complete - include hangover frames and process utterance
                        utterance_frames.extend(hangover_frames)
                        self._process_utterance(utterance_frames)
                        utterance_frames = []
                        hangover_frames.clear()
                        speech_ended = False
                        with self._lock:
                            self._state = "IDLE"
                        self.wake.reset()
                        self.vad.reset()

    def _process_utterance(self, frames: list[AudioFrame]) -> None:
        """Process captured utterance: save recording, transcribe, invoke callback."""
        # Invoke the callback if set
        callback = self._on_utterance
        if callback is not None:
            try:
                callback(frames)
            except Exception:
                pass

        # If storage and STT are configured, process the utterance
        if self.storage is None or self.stt is None:
            return

        if not frames:
            return

        # Concatenate all frames into a single audio payload
        audio_data = np.concatenate([f.data for f in frames], axis=0)
        fmt = self.stream.audio_format()

        # Create raw recording
        raw_recording = Recording(
            data=audio_data,
            path=None,
            sample_rate=fmt.sample_rate,
            channels=fmt.channels,
            dtype=fmt.dtype,
            blocksize=fmt.blocksize,
            device_name=None,
        )

        # Persist the recording
        try:
            persisted = self.storage.save_recording(raw_recording)
        except Exception:
            return

        # Transcribe the recording
        try:
            transcript = self.stt.transcribe_recording(persisted)
        except Exception:
            return

        # Store the result and signal completion
        result = ListeningResult(recording=persisted, transcript=transcript)
        with self._lock:
            self._result = result
        self._result_ready.set()
