import uuid
import time
from typing import Any, Dict, Optional

import modal


# Image with FastAPI and httpx for optional callbacks
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
    .pip_install("httpx")
)

app = modal.App("cursor-modal-bridge", image=image)


# Persistent task status store accessible from web and worker functions
task_store = modal.Dict.from_name("cursor-modal-task-store", create_if_missing=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


@app.function()
def _update_status(run_id: str, update: Dict[str, Any]) -> None:
    # Small helper so both ASGI and worker can atomically update status
    current = task_store.get(run_id, {})
    current.update(update)
    task_store[run_id] = current


@app.function(timeout=900)
def run_task(run_id: str, payload: Dict[str, Any]) -> None:
    """
    Background worker that runs the requested operation and updates status.
    Supports a couple of simple demo operations. Extend as needed for real tasks.
    """
    import httpx  # available via image

    callback_url: Optional[str] = payload.get("callback_url")
    operation: str = payload.get("operation", "echo")
    params: Dict[str, Any] = payload.get("params", {})

    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)
        _update_status.local(run_id, {"logs": logs})

    try:
        _update_status.local(
            run_id,
            {
                "status": "running",
                "progress": 0,
                "started_at": _now_ms(),
                "logs": [],
            },
        )

        # Simulated progress
        for pct in (5, 20, 45, 70, 90):
            time.sleep(0.5)
            log(f"progress {pct}%")
            _update_status.local(run_id, {"progress": pct})

        # Execute the requested operation
        result: Any
        if operation == "sleep":
            duration = float(params.get("seconds", 2))
            log(f"sleeping for {duration} seconds")
            time.sleep(duration)
            result = {"slept_seconds": duration}
        elif operation == "square":
            value = float(params.get("value", 2))
            log(f"squaring value {value}")
            result = {"value": value, "result": value * value}
        elif operation == "echo":
            log("echoing params")
            result = {"echo": params}
        else:
            raise ValueError(f"unsupported operation: {operation}")

        _update_status.local(
            run_id,
            {
                "status": "completed",
                "progress": 100,
                "result": result,
                "completed_at": _now_ms(),
            },
        )

        # Optional callback to Cursor (or any webhook consumer)
        if callback_url:
            try:
                with httpx.Client(timeout=10) as client:
                    client.post(
                        callback_url,
                        json={
                            "run_id": run_id,
                            "status": "completed",
                            "result": result,
                        },
                    )
                log("callback delivered")
            except Exception as callback_err:
                log(f"callback failed: {callback_err}")

    except Exception as exc:  # meaningful handling: record error
        _update_status.local(
            run_id,
            {
                "status": "failed",
                "error": str(exc),
                "completed_at": _now_ms(),
            },
        )


@app.function()
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    web = FastAPI()

    @web.get("/healthz")
    def healthz():
        return {"ok": True}

    @web.post("/cursor/tasks")
    def start_task(payload: Dict[str, Any]):
        # Expected payload: { operation: str, params?: dict, callback_url?: str }
        run_id = str(uuid.uuid4())

        # Seed initial status
        task_store[run_id] = {
            "status": "queued",
            "progress": 0,
            "created_at": _now_ms(),
            "operation": payload.get("operation", "echo"),
        }

        # Fire-and-forget background execution
        run_task.spawn(run_id, payload)

        return {"run_id": run_id, "status": "queued"}

    @web.get("/cursor/tasks/{run_id}")
    def get_status(run_id: str):
        state = task_store.get(run_id)
        if not state:
            raise HTTPException(status_code=404, detail="run_id not found")
        return JSONResponse(state)

    return web


# Deploy:
#   modal deploy cursor_modal_bridge.py
# This exposes routes from the FastAPI app and runs workers in Modal.


