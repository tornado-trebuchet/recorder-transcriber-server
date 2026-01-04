# for 1070 

FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    pulseaudio \
    alsa-utils \
    curl \
    ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv python install 3.10 && uv sync --frozen

COPY . .

RUN mkdir -p /tmp/recorder-transcriber-server /root/.config/recorder-transcriber-server
COPY ./config.yml /root/.config/recorder-transcriber-server/

CMD ["uv", "run", "main.py"]