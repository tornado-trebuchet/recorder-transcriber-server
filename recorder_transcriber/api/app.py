from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from recorder_transcriber.core.di import get_audio_stream
from recorder_transcriber.api import service_router as core_router
from recorder_transcriber.api import main_router as listening_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - start/stop audio stream."""
    stream = get_audio_stream()
    stream.start()
    yield
    stream.stop()


app = FastAPI(
    title="recorder-transcriber",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(core_router.router)
app.include_router(listening_router.router)
