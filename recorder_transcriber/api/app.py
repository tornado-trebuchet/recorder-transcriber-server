from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from recorder_transcriber.api import main_router as listening_router
from recorder_transcriber.api import service_router as core_router
from recorder_transcriber.core.di import get_audio_stream, get_whisper_adapter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - start/stop audio stream and STT adapter."""
    stream = get_audio_stream()
    stt_adapter = get_whisper_adapter()
    stream.start()
    stt_adapter.start()
    yield
    stt_adapter.stop()
    stream.stop()


app = FastAPI(
    title="recorder-transcriber",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(core_router.router)
app.include_router(listening_router.router)
