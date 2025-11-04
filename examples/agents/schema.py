from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Loose JSON schema-like param metadata for clients.",
    )


class ToolManifest(BaseModel):
    tools: List[ToolDefinition]


class ExecRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None
    idempotency_key: Optional[str] = None


class ExecResponse(BaseModel):
    job_id: str
    status: Literal[JobStatus.queued]


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    logs: Optional[str] = None


DEFAULT_MANIFEST = ToolManifest(
    tools=[
        ToolDefinition(
            name="llm.generate",
            description="OpenAI-compatible chat completions via vLLM.",
            params={
                "messages": {"type": "array", "items": {"type": "object"}},
                "model": {"type": "string", "default": "llm"},
                "stream": {"type": "boolean", "default": False},
                "temperature": {"type": "number", "optional": True},
                "max_tokens": {"type": "integer", "optional": True},
            },
        ),
        ToolDefinition(
            name="asr.transcribe",
            description="Transcribe audio using Whisper with dynamic batching.",
            params={
                "audio_url": {"type": "string"},
                "format": {"type": "string", "optional": True},
                "language": {"type": "string", "optional": True},
            },
        ),
        ToolDefinition(
            name="repo.test",
            description="Clone a repo, install deps, run tests.",
            params={
                "repo": {"type": "string"},
                "ref": {"type": "string", "default": "main"},
                "cmd": {"type": "string", "default": "pytest -q"},
                "python": {"type": "string", "optional": True},
            },
        ),
    ]
)
