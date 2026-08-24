import asyncio
import logging
import os
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def _extract_text(out: Any, no_speech_threshold: float = 0.6) -> str:
    """Return transcript text from Whisper's public result shape."""
    if isinstance(out, dict):
        segments = out.get("segments") or []
        if segments and all(s.get("no_speech_prob", 0) >= no_speech_threshold for s in segments):
            return ""
        return str(out.get("text", ""))
    item = out[0] if isinstance(out, list) and out else out
    if hasattr(item, "text"):
        return item.text
    if isinstance(item, list) and item:
        first = item[0]
        return first.text if hasattr(first, "text") else str(first)
    if isinstance(item, tuple) and item:
        first = item[0]
        if isinstance(first, list) and first:
            return first[0] if isinstance(first[0], str) else getattr(first[0], "text", str(first[0]))
    return str(item)


class ASR:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("SCRIBER_WHISPER_MODEL", "medium")
        self.language = os.getenv("SCRIBER_WHISPER_LANGUAGE", "en")
        if self.language.lower() == "auto":
            self.language = None
        self.no_speech_threshold = float(os.getenv("SCRIBER_WHISPER_NO_SPEECH_THRESHOLD", "0.6"))
        self.device = self._resolve_device(os.getenv("SCRIBER_WHISPER_DEVICE", "auto"))
        self.download_root = os.getenv("SCRIBER_WHISPER_CACHE_DIR") or None
        self.model = None
        self._lock = asyncio.Lock()
        self._ready = False
        self._load_error = ""

    @staticmethod
    def _resolve_device(value: str) -> str:
        if value != "auto":
            return value
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> str:
        return self._load_error

    async def load(self):
        try:
            log.info("loading OpenAI Whisper model %s on %s", self.model_name, self.device)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_sync)
            log.info("model loaded; warming up")
            await loop.run_in_executor(None, self._warmup)
            self._ready = True
            self._load_error = ""
            log.info("server ready")
        except Exception as exc:
            self._ready = False
            self._load_error = str(exc)
            log.exception("model load failed")

    def _load_sync(self):
        import whisper

        self.model = whisper.load_model(
            self.model_name,
            device=self.device,
            download_root=self.download_root,
        )

    def _warmup(self):
        silence = np.zeros(16000, dtype=np.float32)
        with torch.no_grad():
            self.model.transcribe(
                silence,
                language=self.language,
                task="transcribe",
                fp16=self.device == "cuda",
                verbose=False,
                condition_on_previous_text=False,
            )
        if self.device == "cuda":
            torch.cuda.empty_cache()

    async def transcribe(self, samples: np.ndarray, language: str | None = "") -> str:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._transcribe_sync, samples, language)

    def _transcribe_sync(self, samples: np.ndarray, language: str | None = "") -> str:
        if language == "":
            language = self.language
        with torch.no_grad():
            out = self.model.transcribe(
                samples,
                language=language,
                task="transcribe",
                fp16=self.device == "cuda",
                verbose=False,
                condition_on_previous_text=False,
            )
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return _extract_text(out, self.no_speech_threshold)
