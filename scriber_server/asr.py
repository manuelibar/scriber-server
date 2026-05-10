import asyncio
import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def _extract_text(out: Any) -> str:
    """NeMo returns list-of-Hypothesis or tuple-of-lists depending on version."""
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
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        self.model_name = model_name
        self.model = None
        self._lock = asyncio.Lock()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def load(self):
        log.info("loading model %s", self.model_name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)
        log.info("model loaded; warming up")
        await loop.run_in_executor(None, self._warmup)
        self._ready = True
        log.info("server ready")

    def _load_sync(self):
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        m = m.half().cuda()
        m.eval()
        self.model = m

    def _warmup(self):
        silence = np.zeros(16000, dtype=np.float32)
        with torch.no_grad():
            self.model.transcribe([silence], batch_size=1, verbose=False)
        torch.cuda.empty_cache()

    async def transcribe(self, samples: np.ndarray) -> str:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._transcribe_sync, samples)

    def _transcribe_sync(self, samples: np.ndarray) -> str:
        with torch.no_grad():
            out = self.model.transcribe([samples], batch_size=1, verbose=False)
        torch.cuda.empty_cache()
        return _extract_text(out)
