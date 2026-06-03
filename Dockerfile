# Scriber speech-to-text server using OpenAI Whisper.
# Base: NVIDIA CUDA runtime + cuDNN on Ubuntu 24.04. Python 3.12 from system repos.
# Image size depends on the selected Whisper model.
#
# Run through the root repo Compose stack with `stt start`.

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SCRIBER_WHISPER_MODEL=base \
    SCRIBER_WHISPER_DEVICE=auto \
    SCRIBER_WHISPER_LANGUAGE=en \
    SCRIBER_SILENCE_RMS_THRESHOLD=0.0005 \
    SCRIBER_WHISPER_CACHE_DIR=/cache/whisper

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ca-certificates curl libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --home-dir /tmp --shell /usr/sbin/nologin --no-create-home app && \
    mkdir -p /cache/whisper && \
    chown -R app:app /cache

# uv for fast deps install
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

COPY pyproject.toml ./
COPY scriber_server/ ./scriber_server/

RUN uv venv --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# Pre-cache the model into the image. Skipped at build time if BUILD_CACHE_MODEL=0.
ARG BUILD_CACHE_MODEL=1
RUN if [ "$BUILD_CACHE_MODEL" = "1" ]; then \
        /opt/venv/bin/python -c "import os, whisper; whisper.load_model(os.environ.get('SCRIBER_WHISPER_MODEL', 'base'), download_root=os.environ.get('SCRIBER_WHISPER_CACHE_DIR'))" ; \
    fi && \
    chown -R app:app /cache

USER app:app

EXPOSE 8765
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "scriber_server.app:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1", "--limit-concurrency", "4"]
