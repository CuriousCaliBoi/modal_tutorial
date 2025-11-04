from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

import modal

from .schema import DEFAULT_MANIFEST, ExecRequest, ExecResponse, JobResult, JobStatus, ToolManifest
from .tools import llm, asr
from .tools import repo_ci


# Application and shared resources
app = modal.App("cursor-agent-tools")

server_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pydantic>=2.7.0",
        "aiohttp>=3.9.0",
        "numpy>=1.26.0",
        "soundfile>=0.12.1",
    )
)

job_store = modal.Dict.from_name("agent-jobs", create_if_missing=True)


def _require_auth(headers: Dict[str, str]) -> None:
    header = headers.get("X-Agent-Token") or headers.get("x-agent-token")
    secret = os.environ.get("CURSOR_AGENT_TOKEN")
    if not secret or not header or header != secret:
        raise PermissionError("Unauthorized: missing or invalid X-Agent-Token")


def _idempotency_lookup(idempotency_key: Optional[str]) -> Optional[str]:
    if not idempotency_key:
        return None
    idem_key = f"idem:{idempotency_key}"
    return job_store.get(idem_key)  # type: ignore[no-any-return]


def _idempotency_set(idempotency_key: str, job_id: str) -> None:
    job_store[f"idem:{idempotency_key}"] = job_id


@app.function(image=server_image, secrets=[modal.Secret.from_name("CURSOR_AGENT_TOKEN")])
@modal.web_endpoint(method="POST")
async def tools_manifest(headers: Dict[str, str]) -> Dict[str, Any]:
    _require_auth(headers)
    manifest: ToolManifest = DEFAULT_MANIFEST
    return manifest.model_dump()


@app.function(image=server_image, secrets=[modal.Secret.from_name("CURSOR_AGENT_TOKEN")])
@modal.web_endpoint(method="POST")
async def tools_exec(headers: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
    _require_auth(headers)

    req = ExecRequest(**(data or {}))

    # Idempotency: return existing job if present
    existing = _idempotency_lookup(req.idempotency_key)
    if existing:
        return ExecResponse(job_id=existing, status=JobStatus.queued).model_dump()

    job_id = str(uuid.uuid4())

    # Persist queued state
    job_store[job_id] = JobResult(job_id=job_id, status=JobStatus.queued).model_dump()

    # Map idempotency key to job id
    if req.idempotency_key:
        _idempotency_set(req.idempotency_key, job_id)

    # Spawn async execution
    _ = execute_tool.spawn(job_id, req.tool, req.params)

    return ExecResponse(job_id=job_id, status=JobStatus.queued).model_dump()


@app.function(image=server_image, secrets=[modal.Secret.from_name("CURSOR_AGENT_TOKEN")])
@modal.web_endpoint(method="GET")
async def jobs(headers: Dict[str, str], job_id: str) -> Dict[str, Any]:
    _require_auth(headers)
    data = job_store.get(job_id)
    if not data:
        return {"job_id": job_id, "status": "not_found"}
    return data


@app.function(image=server_image, concurrency_limit=64)
async def execute_tool(job_id: str, tool: str, params: Dict[str, Any]) -> None:
    def _update(status: JobStatus, result: Any | None = None, error: str | None = None, logs: str | None = None) -> None:
        job_store[job_id] = JobResult(
            job_id=job_id, status=status, result=result, error=error, logs=logs
        ).model_dump()

    try:
        print(f"[{job_id}] starting tool={tool}")
        _update(JobStatus.running)

        if tool == "llm.generate":
            result = await llm.generate(params)
        elif tool == "asr.transcribe":
            audio_url = params.get("audio_url")
            language = params.get("language")
            if not audio_url:
                raise ValueError("'audio_url' is required for asr.transcribe")
            result = await asr.transcribe_from_url(audio_url, language)
        elif tool == "repo.test":
            repo = params.get("repo")
            ref = params.get("ref", "main")
            cmd = params.get("cmd", "pytest -q")
            python = params.get("python")
            if not repo:
                raise ValueError("'repo' is required for repo.test")
            result = await repo_ci.run.remote.aio(repo, ref, cmd, python)
        else:
            raise ValueError(f"Unknown tool: {tool}")

        _update(JobStatus.succeeded, result=result)
        print(f"[{job_id}] succeeded tool={tool}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[{job_id}] failed tool={tool} error={err}")
        _update(JobStatus.failed, error=err)


# Optional: expose an ASGI app with the exact routes specified in the plan
api_image = server_image.pip_install("fastapi[standard]>=0.115.0")


@app.function(image=api_image, secrets=[modal.Secret.from_name("CURSOR_AGENT_TOKEN")])
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse

    web_app = FastAPI()

    @web_app.post("/tools/manifest")
    async def manifest(x_agent_token: str = Header(None)):
        try:
            _require_auth({"X-Agent-Token": x_agent_token} if x_agent_token else {})
        except PermissionError as e:
            raise HTTPException(status_code=401, detail=str(e))
        return DEFAULT_MANIFEST.model_dump()

    @web_app.post("/tools/exec")
    async def exec_(data: Dict[str, Any], x_agent_token: str = Header(None)):
        try:
            _require_auth({"X-Agent-Token": x_agent_token} if x_agent_token else {})
        except PermissionError as e:
            raise HTTPException(status_code=401, detail=str(e))

        req = ExecRequest(**(data or {}))
        existing = _idempotency_lookup(req.idempotency_key)
        if existing:
            return ExecResponse(job_id=existing, status=JobStatus.queued).model_dump()

        job_id = str(uuid.uuid4())
        job_store[job_id] = JobResult(job_id=job_id, status=JobStatus.queued).model_dump()
        if req.idempotency_key:
            _idempotency_set(req.idempotency_key, job_id)
        _ = execute_tool.spawn(job_id, req.tool, req.params)
        return ExecResponse(job_id=job_id, status=JobStatus.queued).model_dump()

    @web_app.get("/jobs/{job_id}")
    async def get_job(job_id: str, x_agent_token: str = Header(None)):
        try:
            _require_auth({"X-Agent-Token": x_agent_token} if x_agent_token else {})
        except PermissionError as e:
            raise HTTPException(status_code=401, detail=str(e))
        data = job_store.get(job_id)
        if not data:
            return JSONResponse({"job_id": job_id, "status": "not_found"}, status_code=404)
        return data

    return web_app
