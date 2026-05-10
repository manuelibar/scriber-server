import numpy as np


def pad_audio(samples: np.ndarray, sample_rate: int, min_seconds: float = 1.0) -> np.ndarray:
    min_samples = int(sample_rate * min_seconds)
    if samples.shape[0] >= min_samples:
        return samples
    pad = min_samples - samples.shape[0]
    return np.concatenate([samples, np.zeros(pad, dtype=samples.dtype)])


_TERMINAL_PUNCT = ".?!"


def postprocess_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    if s[0].isalpha():
        s = s[0].upper() + s[1:]
    if len(s.split()) >= 2 and s[-1] not in _TERMINAL_PUNCT:
        s = s + "."
    return s
