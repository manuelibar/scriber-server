# Scriber speech-to-text server.
# Base: NVIDIA CUDA runtime + cuDNN on Ubuntu 24.04. Python 3.12 from system repos.
# Image size ~6 GB after model download is materialized into the image cache.
#
# Run with: docker run --gpus all -p 127.0.0.1:8765:8765 scriber-server
# Or use docker-compose.yml in this directory.

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/huggingface \
    NEMO_CACHE_DIR=/cache/nemo

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ca-certificates curl libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# uv for fast deps install
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

# Install deps first (cache layer) before copying source.
COPY pyproject.toml ./
RUN uv venv --python 3.12 /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --python /opt/venv/bin/python .

# Copy source AFTER deps so code edits don't bust the heavy install layer.
COPY scriber_server/ ./scriber_server/

# Pre-cache the model into the image. Skipped at build time if BUILD_CACHE_MODEL=0.
ARG BUILD_CACHE_MODEL=1
RUN if [ "$BUILD_CACHE_MODEL" = "1" ]; then \
        /opt/venv/bin/python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')" ; \
    fi

EXPOSE 8765
CMD ["/opt/venv/bin/python", "-m", "scriber_server.app"]
