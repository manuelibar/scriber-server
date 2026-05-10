# scriber-server

FastAPI + NeMo speech-to-text backend for [scriber](../). Loads `nvidia/parakeet-tdt-0.6b-v2` once and serves a single `/transcribe` endpoint over localhost.

## Two install paths

### Docker (recommended)

```bash
docker compose up -d --build
# first build downloads the model (~600 MB) and bakes it into the image
docker compose logs -f scriber-server  # follow warm-up
```

Requires NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Image exposes port 8765 on `127.0.0.1` only.

To skip baking the model into the image (smaller image, model downloaded into the named volume on first request):

```bash
docker compose build --build-arg BUILD_CACHE_MODEL=0
docker compose up -d
```

### Native

Requirements: NVIDIA driver, Python 3.12 (NeMo wheels are unreliable on 3.13/3.14), [`uv`](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not installed
uv venv --python 3.12
uv pip install -e .
.venv/bin/scriber-server
```

Or as a systemd user service:

```bash
cp systemd/scriber-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now scriber-server
journalctl --user -u scriber-server -f
```

## API

- `GET /healthz` — `200 {"ok": true}` once the model is warm. `503` during cold start.
- `POST /transcribe` — body: raw int16 little-endian PCM, header `X-Sample-Rate: 16000` (only 16 kHz supported in v1). Returns `{"text": "...", "raw": "...", "ms": 187, "audio_ms": 1000}`.

## Notes

- Single worker, single GPU lock. Concurrent requests serialize.
- After each request: `torch.cuda.empty_cache()`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation alongside browsers.
- Audio shorter than 1.0 s is padded with trailing silence (Parakeet's punctuation/casing model wants context).
- Post-processing: capitalize first letter, append `.` if no terminal punctuation and ≥ 2 words.
