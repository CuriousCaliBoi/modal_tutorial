from __future__ import annotations

import io
from typing import Any, Dict

import aiohttp
import modal
import numpy as np
import soundfile as sf


async def _download_audio_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=300) as resp:
            resp.raise_for_status()
            return await resp.read()


def _decode_audio_to_array(audio_bytes: bytes) -> Dict[str, Any]:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    # Ensure mono by averaging channels if needed
    if data.ndim == 2 and data.shape[1] > 1:
        data = np.mean(data, axis=1)
    return {"array": data, "sampling_rate": int(sr)}


async def transcribe_from_url(audio_url: str, language: str | None = None) -> Dict[str, Any]:
    audio_bytes = await _download_audio_bytes(audio_url)
    sample = _decode_audio_to_array(audio_bytes)

    # Lookup deployed Whisper model class
    whisper_cls = modal.Cls.lookup("example-batched-whisper", "Model")
    model = whisper_cls()

    # Dynamic batching interface expects a list of samples
    results = await model.transcribe.remote.aio([sample])
    text = results[0]["text"] if results else ""
    return {"text": text, "language": language}
