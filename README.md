# scriber-server

FastAPI + OpenAI Whisper speech-to-text backend for [scriber](../). Loads a local Whisper model once and serves a single `/transcribe` endpoint over localhost.

The client sends raw 16 kHz mono PCM to this server. Users normally do not call the API directly; the Go daemon does that after a hotkey capture.

## Docker

```bash
stt start --no-daemon
```

The root Compose stack is the only supported setup path. It starts the server through Docker and exposes port 8765 on `127.0.0.1` only.

Requires NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The first build downloads the selected Whisper model and can take a while.

The root setup script installs Docker's official Engine packages, the Compose plugin, buildx, and NVIDIA Container Toolkit. The image creates an `app` user with UID/GID 10001 and runs the server as that user. Compose mounts `/cache` as the writable Whisper model volume, mounts `/tmp` as tmpfs, drops Linux capabilities, prevents privilege escalation, and keeps the runtime root filesystem read-only.

To skip baking the model into the image, set `BUILD_CACHE_MODEL=0` in `.private/.env` from the repo root. That keeps the image smaller and downloads the model into the named volume on first request:

```bash
stt start --no-daemon
```

## API

- `GET /healthz` — `200 {"ok": true}` once the model is warm. `503` during cold start or model-load failure.
- `POST /transcribe` — body: raw int16 little-endian PCM, header `X-Sample-Rate: 16000` (only 16 kHz supported in v1), optional `X-Language: es` or `X-Language: auto`. Returns `{"text": "...", "raw": "...", "ms": 187, "audio_ms": 1000, "language": "es"}`.

## Whisper tuning

Environment variables:

```bash
SCRIBER_WHISPER_MODEL=base        # tiny, base, small, medium, turbo, etc.; .en models are English-only
SCRIBER_WHISPER_DEVICE=auto       # auto, cuda, cpu
SCRIBER_WHISPER_LANGUAGE=en       # en, auto, or another Whisper language code
SCRIBER_WHISPER_CACHE_DIR=...     # optional model cache location
SCRIBER_SILENCE_RMS_THRESHOLD=0.0005
```

The root `.private/.env` uses `STT_WHISPER_MODEL`, `STT_WHISPER_DEVICE`, `STT_WHISPER_LANGUAGE`, `STT_SILENCE_RMS_THRESHOLD`, `STT_SERVER_HOST`, `STT_SERVER_PORT`, `STT_GPU_COUNT`, and `BUILD_CACHE_MODEL`; the Compose file maps the server-specific values into the container as `SCRIBER_*` variables.

Use `base` for multilingual streams. Use `small`, `medium`, or `turbo` when you want better quality and can spend more VRAM/download time. English-only `.en` models should not be used when Spanish streams are needed.

After changing environment variables in `.private/.env`, run `stt start --no-daemon` and then `curl http://127.0.0.1:8765/healthz`.

Expected healthy response:

```json
{"ok":true,"backend":"openai-whisper","model":"base","device":"cuda"}
```

## Notes

- Single worker, single model lock. Concurrent requests serialize.
- After each CUDA request: `torch.cuda.empty_cache()`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation alongside browsers.
- Audio shorter than 1.0 s is padded with trailing silence.
- Post-processing: capitalize first letter, append `.` if no terminal punctuation and ≥ 2 words.
