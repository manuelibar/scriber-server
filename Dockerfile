# Scriber speech-to-text server using OpenAI Whisper.
# Base: NVIDIA CUDA runtime + cuDNN on Ubuntu 24.04. Python 3.12 from system repos.
# Image size depends on the selected Whisper model.
#
# Run with: docker run --gpus all -p 127.0.0.1:8765:8765 scriber-server
# Or use docker-compose.yml in this directory.

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SCRIBER_WHISPER_MODEL=base.en \
    SCRIBER_WHISPER_DEVICE=auto \
    SCRIBER_WHISPER_LANGUAGE=en \
    SCRIBER_SILENCE_RMS_THRESHOLD=0.0005 \
    SCRIBER_WHISPER_CACHE_DIR=/cache/whisper

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ca-certificates curl libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# uv for fast deps install
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

COPY pyproject.toml ./
COPY scriber_server/ ./scriber_server/

# BuildKit cache mount keeps uv's wheel cache across rebuilds so source edits
# don't re-download torch/Whisper dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# Pre-cache the model into the image. Skipped at build time if BUILD_CACHE_MODEL=0.
ARG BUILD_CACHE_MODEL=1
RUN if [ "$BUILD_CACHE_MODEL" = "1" ]; then \
        /opt/venv/bin/python -c "import os, whisper; whisper.load_model(os.environ.get('SCRIBER_WHISPER_MODEL', 'base.en'), download_root=os.environ.get('SCRIBER_WHISPER_CACHE_DIR'))" ; \
    fi

EXPOSE 8765
CMD ["/opt/venv/bin/python", "-m", "scriber_server.app"]
