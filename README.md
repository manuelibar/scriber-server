# scriber-server

FastAPI + OpenAI Whisper speech-to-text backend for [scriber](../). Loads a local Whisper model once and serves a single `/transcribe` endpoint over localhost.

The client sends raw 16 kHz mono PCM to this server. Users normally do not call the API directly; the Go daemon does that after a hotkey capture.

## Docker

```bash
cd ..
make services-start
make logs
```

The root Compose stack is the only supported setup path. It starts the server through Docker and exposes port 8765 on `127.0.0.1` only.

Requires NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The first build downloads the selected Whisper model and can take a while.

To skip baking the model into the image (smaller image, model downloaded into the named volume on first request):

```bash
docker compose build --build-arg BUILD_CACHE_MODEL=0
docker compose up -d
```

## API

- `GET /healthz` — `200 {"ok": true}` once the model is warm. `503` during cold start or model-load failure.
- `POST /transcribe` — body: raw int16 little-endian PCM, header `X-Sample-Rate: 16000` (only 16 kHz supported in v1). Returns `{"text": "...", "raw": "...", "ms": 187, "audio_ms": 1000}`.

## Whisper tuning

Environment variables:

```bash
SCRIBER_WHISPER_MODEL=base.en     # tiny.en, base.en, small.en, medium.en, turbo, etc.
SCRIBER_WHISPER_DEVICE=auto       # auto, cuda, cpu
SCRIBER_WHISPER_LANGUAGE=en       # en, auto, or another Whisper language code
SCRIBER_WHISPER_CACHE_DIR=...     # optional model cache location
SCRIBER_SILENCE_RMS_THRESHOLD=0.0005
```

Use `base.en` for the quickest usable setup. Use `small.en` or `turbo` when you want better quality and can spend more VRAM/download time.

After changing environment variables in `.private/.env`, run `make services-start` from the repo root and then `curl http://127.0.0.1:8765/healthz`.

Expected healthy response:

```json
{"ok":true,"backend":"openai-whisper","model":"base.en","device":"cuda"}
```

## Notes

- Single worker, single model lock. Concurrent requests serialize.
- After each CUDA request: `torch.cuda.empty_cache()`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation alongside browsers.
- Audio shorter than 1.0 s is padded with trailing silence.
- Post-processing: capitalize first letter, append `.` if no terminal punctuation and ≥ 2 words.
