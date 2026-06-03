import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .asr import ASR
from .postprocess import pad_audio, postprocess_text

log = logging.getLogger("scriber-server")

asr = ASR()
silence_rms_threshold = float(os.getenv("SCRIBER_SILENCE_RMS_THRESHOLD", "0.0005"))


def normalize_language(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return ""
    value = value.strip().lower().replace("_", "-").replace(":", "-")
    if value == "auto":
        return None
    primary = value.split("-", 1)[0]
    if not re.fullmatch(r"[a-z]{2,3}", primary):
        raise HTTPException(400, "language must be auto, a Whisper language code, or a locale like es-ES")
    return primary


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(asr.load())
    yield


app = FastAPI(lifespan=lifespan, title="scriber-server")


@app.get("/healthz")
async def healthz():
    if asr.load_error:
        return JSONResponse({"ok": False, "reason": "load_failed", "error": asr.load_error}, status_code=503)
    if not asr.ready:
        return JSONResponse({"ok": False, "reason": "warming up"}, status_code=503)
    return {"ok": True, "backend": "openai-whisper", "model": asr.model_name, "device": asr.device}


@app.post("/transcribe")
async def transcribe(request: Request):
    if asr.load_error:
        raise HTTPException(503, "model load failed: " + asr.load_error)
    if not asr.ready:
        raise HTTPException(503, "warming up")

    sample_rate = int(request.headers.get("x-sample-rate", "16000"))
    if sample_rate != 16000:
        raise HTTPException(400, f"only 16000 Hz supported, got {sample_rate}")
    language = normalize_language(request.headers.get("x-language"))
    response_language = language
    if response_language == "":
        response_language = asr.language
    response_language = response_language or "auto"

    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    if len(body) % 2 != 0:
        raise HTTPException(400, "body must be int16 PCM (even byte count)")

    samples = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    audio_ms = int(samples.shape[0] / sample_rate * 1000)
    if float(np.sqrt(np.mean(samples * samples))) < silence_rms_threshold:
        return {"text": "", "raw": "", "ms": 0, "audio_ms": audio_ms, "language": response_language}
    samples = pad_audio(samples, sample_rate, min_seconds=1.0)

    t0 = time.monotonic()
    raw = await asr.transcribe(samples, language)
    text = postprocess_text(raw)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "transcribed %.2fs audio language=%s -> %d chars in %dms",
        samples.shape[0] / sample_rate,
        response_language,
        len(text),
        elapsed_ms,
    )
    return {
        "text": text,
        "raw": raw,
        "ms": elapsed_ms,
        "audio_ms": int(samples.shape[0] / sample_rate * 1000),
        "language": response_language,
    }


def main():
    import uvicorn

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "scriber_server.app:app",
        host="127.0.0.1",
        port=8765,
        workers=1,
        limit_concurrency=4,
    )


if __name__ == "__main__":
    main()
