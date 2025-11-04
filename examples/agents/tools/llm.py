from __future__ import annotations

import os
from typing import Any, Dict

import aiohttp
import modal


def _resolve_vllm_base_url() -> str:
    # Prefer explicit override
    override = os.environ.get("VLLM_SERVER_URL")
    if override:
        return override.rstrip("/")

    # Fallback to looking up deployed vLLM web server
    try:
        fn = modal.Function.lookup("example-vllm-inference", "serve")
        # Try both access patterns for compatibility across modal versions
        if hasattr(fn, "web_url") and fn.web_url:
            return str(fn.web_url).rstrip("/")
        if hasattr(fn, "get_web_url"):
            return str(fn.get_web_url()).rstrip("/")
    except Exception as e:  # pragma: no cover - best-effort discovery
        raise RuntimeError(
            "Could not resolve vLLM server URL. Deploy 'examples/ml/vllm_inference.py' "
            "or set VLLM_SERVER_URL."
        ) from e

    raise RuntimeError(
        "vLLM server URL not available. Deploy 'examples/ml/vllm_inference.py' "
        "or set VLLM_SERVER_URL."
    )


async def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = _resolve_vllm_base_url()
    url = f"{base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=300) as resp:
            resp.raise_for_status()
            return await resp.json()
