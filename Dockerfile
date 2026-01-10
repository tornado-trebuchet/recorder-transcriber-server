FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    portaudio19-dev \
    pulseaudio-utils \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

RUN mkdir -p /tmp/recorder-transcriber-server \
    && mkdir -p /root/.local/share/recorder-transcriber-server \
    && mkdir -p /root/.cache/huggingface

ENV CONFIG_PATH=/app/config.yml
ENV PYTHONUNBUFFERED=1

EXPOSE 1643

CMD ["uv", "run", "main.py"]