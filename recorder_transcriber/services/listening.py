import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator
import time

import numpy as np

from recorder_transcriber.core.settings import ListenerConfig
from recorder_transcriber.domain.models import AudioFrame, ListeningResult, Recording
from recorder_transcriber.ports.audiostream import AudioStreamPort, AudioStreamReader
from recorder_transcriber.ports.vad import VadPort
from recorder_transcriber.ports.wakeword import WakeWordPort
from recorder_transcriber.ports.stt import SpeechToTextPort
from recorder_transcriber.ports.storage import AudioStoragePort


class ListenerState(str, Enum):
    """Listener state machine states."""

    IDLE = "IDLE"
    ARMED = "ARMED"
    LISTENING = "LISTENING"


@dataclass
class ListenerEvent:
    """Event emitted by the listener service."""

    type: str  # "state_change", "result", "error"
    state: ListenerState | None = None
    result: ListeningResult | None = None
    error: str | None = None


@dataclass(slots=True)
class ListenerService:
    """Async orchestrator for wake-word detection and VAD-based voice recording.

    State machine:
    - IDLE: Waiting for wake-word
    - ARMED: Wake-word detected, waiting for speech
    - LISTENING: Speech detected, capturing utterance

    Produces events via async iterator for WebSocket consumers.
    """

    stream: AudioStreamPort
    wake: WakeWordPort
    vad: VadPort
    config: ListenerConfig
    storage: AudioStoragePort | None = None
    stt: SpeechToTextPort | None = None

    _state: ListenerState = field(default=ListenerState.IDLE, init=False)
    _running: bool = field(default=False, init=False)
    _reader: AudioStreamReader | None = field(default=None, init=False)
    _event_queue: asyncio.Queue[ListenerEvent] = field(default_factory=asyncio.Queue, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)

    @property
    def state(self) -> ListenerState:
        """Current listener state."""
        return self._state

    @property
    def is_listening(self) -> bool:
        """Check if the listener is active."""
        return self._running

    async def start(self) -> None:
        """Start the listening service.

        Raises:
            RuntimeError: If already running or audio stream not active.
        """
        if self._running:
            raise RuntimeError("Listening session already active")
        if not self.stream.is_running():
            raise RuntimeError("Audio stream is not running")

        self._running = True
        self._state = ListenerState.IDLE
        self._reader = self.stream.subscribe(name="listener", max_frames=4096)
        self._event_queue = asyncio.Queue()

        # Start the main loop as a background task
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the listening service and release resources."""
        self._running = False

        # Cancel and await the task
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

        self._state = ListenerState.IDLE
        self.wake.reset()
        self.vad.reset()

    async def events(self) -> AsyncIterator[ListenerEvent]:
        """Async iterator that yields events as they occur.

        Yields:
            ListenerEvent for state changes, results, and errors.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.5,
                )
                yield event
            except asyncio.TimeoutError:
                # No event, but still running - continue waiting
                continue

    async def _emit(self, event: ListenerEvent) -> None:
        """Put an event on the queue for consumers."""
        await self._event_queue.put(event)

    async def _set_state(self, new_state: ListenerState) -> None:
        """Update state and emit state_change event."""
        if self._state != new_state:
            self._state = new_state
            await self._emit(ListenerEvent(type="state_change", state=new_state))

    async def _run(self) -> None:
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

        while self._running:
            # Read frame in thread pool to avoid blocking event loop
            frame = await asyncio.to_thread(reader.read, 0.1)
            if frame is None:
                continue

            current_state = self._state

            if current_state == ListenerState.IDLE:
                armed_start_time = await self._handle_idle(frame, pre_roll_frames)

            elif current_state == ListenerState.ARMED:
                armed_start_time, speech_ended = await self._handle_armed(
                    frame,
                    pre_roll_frames,
                    pre_roll_max,
                    armed_start_time,
                    utterance_frames,
                )

            elif current_state == ListenerState.LISTENING:
                speech_ended, armed_start_time = await self._handle_listening(
                    frame,
                    utterance_frames,
                    hangover_frames,
                    max_utterance_frames,
                    hangover_max,
                    speech_ended,
                    pre_roll_frames,
                )
                # Reset if we processed an utterance
                if self._state == ListenerState.IDLE:
                    armed_start_time = None
                    speech_ended = False

    async def _handle_idle(
        self,
        frame: AudioFrame,
        pre_roll_frames: list[AudioFrame],
    ) -> float | None:
        """Handle frame in IDLE state - check for wake-word."""
        wake_result = await asyncio.to_thread(self.wake.detect, frame)
        if wake_result.detected:
            await self._set_state(ListenerState.ARMED)
            self.vad.reset()
            pre_roll_frames.clear()
            return time.monotonic()
        return None

    async def _handle_armed(
        self,
        frame: AudioFrame,
        pre_roll_frames: list[AudioFrame],
        pre_roll_max: int,
        armed_start_time: float | None,
        utterance_frames: list[AudioFrame],
    ) -> tuple[float | None, bool]:
        """Handle frame in ARMED state - wait for speech start."""
        if armed_start_time is None:
            armed_start_time = time.monotonic()

        # Check for armed timeout
        elapsed = time.monotonic() - armed_start_time
        if elapsed > self.config.armed_timeout_seconds:
            await self._set_state(ListenerState.IDLE)
            pre_roll_frames.clear()
            self.wake.reset()
            self.vad.reset()
            return None, False

        # Buffer frames for pre-roll
        pre_roll_frames.append(frame)
        if len(pre_roll_frames) > pre_roll_max:
            pre_roll_frames.pop(0)

        # Check for speech start
        vad_event = await asyncio.to_thread(self.vad.process, frame)
        if vad_event is not None and vad_event.detected:
            await self._set_state(ListenerState.LISTENING)
            utterance_frames.clear()
            utterance_frames.extend(pre_roll_frames)
            pre_roll_frames.clear()
            return None, False

        return armed_start_time, False

    async def _handle_listening(
        self,
        frame: AudioFrame,
        utterance_frames: list[AudioFrame],
        hangover_frames: list[AudioFrame],
        max_utterance_frames: int,
        hangover_max: int,
        speech_ended: bool,
        pre_roll_frames: list[AudioFrame],
    ) -> tuple[bool, float | None]:
        """Handle frame in LISTENING state - capture utterance."""
        utterance_frames.append(frame)

        # Check for max utterance duration
        if len(utterance_frames) >= max_utterance_frames:
            await self._process_utterance(utterance_frames.copy())
            utterance_frames.clear()
            hangover_frames.clear()
            pre_roll_frames.clear()
            await self._set_state(ListenerState.IDLE)
            self.wake.reset()
            self.vad.reset()
            return False, None

        # Check for speech end
        vad_event = await asyncio.to_thread(self.vad.process, frame)
        if vad_event is not None and not vad_event.detected:
            speech_ended = True
            hangover_frames.clear()
            hangover_frames.append(frame)

        # Collect hangover frames after speech ends
        if speech_ended:
            if not any(f.sequence == frame.sequence for f in hangover_frames):
                hangover_frames.append(frame)

            if len(hangover_frames) >= hangover_max:
                # Include hangover frames and process utterance
                utterance_frames.extend(hangover_frames)
                await self._process_utterance(utterance_frames.copy())
                utterance_frames.clear()
                hangover_frames.clear()
                pre_roll_frames.clear()
                await self._set_state(ListenerState.IDLE)
                self.wake.reset()
                self.vad.reset()
                return False, None

        return speech_ended, None

    async def _process_utterance(self, frames: list[AudioFrame]) -> None:
        """Process captured utterance: save recording, transcribe, emit result."""
        if not frames:
            return

        if self.storage is None or self.stt is None:
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
            persisted = await asyncio.to_thread(self.storage.save_recording, raw_recording)
        except Exception as e:
            await self._emit(ListenerEvent(type="error", error=f"Failed to save recording: {e}"))
            return

        # Transcribe the recording
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe_recording, persisted)
        except Exception as e:
            await self._emit(ListenerEvent(type="error", error=f"Failed to transcribe: {e}"))
            return

        # Emit result event
        result = ListeningResult(recording=persisted, transcript=transcript)
        await self._emit(ListenerEvent(type="result", result=result))
