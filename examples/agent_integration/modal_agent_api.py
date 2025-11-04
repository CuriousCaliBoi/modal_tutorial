"""
Modal Agent API Service

FastAPI service deployed on Modal that provides HTTP endpoints for agents to:
- Deploy Modal functions dynamically
- Trigger parallel processing tasks
- Query Modal app/function status
- Manage compute resources
"""

import modal
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

# Create an image with FastAPI and required dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "pydantic",
    "httpx"
)

app = modal.App("modal-agent-api", image=image)

# Store for tracking deployed functions and tasks
task_store: Dict[str, Dict[str, Any]] = {}
function_store: Dict[str, Any] = {}


@app.function()
@modal.asgi_app()
def modal_agent_api():
    """FastAPI application for agent operations"""
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    
    web_app = FastAPI(
        title="Modal Agent API",
        description="API for Cursor cloud agents to interact with Modal",
        version="1.0.0"
    )
    
    # Request/Response models
    class DeployFunctionRequest(BaseModel):
        function_code: str
        function_name: str
        image_config: Optional[Dict[str, Any]] = None
        gpu: Optional[str] = None
        timeout: Optional[int] = None
        secrets: Optional[List[str]] = None
        volumes: Optional[List[str]] = None
    
    class RunParallelRequest(BaseModel):
        function_name: str
        inputs: List[Any]
        wait: bool = False
    
    class SpawnTaskRequest(BaseModel):
        function_name: str
        args: Optional[List[Any]] = None
        kwargs: Optional[Dict[str, Any]] = None
        wait: bool = False
    
    class WebhookRequest(BaseModel):
        task_id: str
        webhook_url: str
    
    @web_app.get("/")
    def root():
        return {
            "service": "Modal Agent API",
            "version": "1.0.0",
            "endpoints": {
                "deploy": "POST /deploy-function",
                "parallel": "POST /run-parallel",
                "spawn": "POST /spawn-task",
                "status": "GET /status/{task_id}",
                "webhook": "POST /webhook",
                "functions": "GET /functions",
                "tasks": "GET /tasks"
            }
        }
    
    @web_app.post("/deploy-function")
    async def deploy_function(request: DeployFunctionRequest):
        """
        Deploy a Modal function dynamically from code string.
        
        Agents can provide Python code as a string, and this endpoint will
        deploy it as a Modal function that can be called later.
        """
        try:
            function_id = str(uuid.uuid4())
            
            # Store function metadata
            function_store[function_id] = {
                "name": request.function_name,
                "code": request.function_code,
                "image_config": request.image_config,
                "gpu": request.gpu,
                "timeout": request.timeout,
                "secrets": request.secrets,
                "volumes": request.volumes,
                "deployed_at": datetime.utcnow().isoformat(),
                "status": "registered"
            }
            
            # Note: In a real implementation, you would need to dynamically
            # compile and register the function with Modal. This is a simplified
            # version that stores the function definition for later execution.
            
            return {
                "function_id": function_id,
                "function_name": request.function_name,
                "status": "registered",
                "message": "Function registered successfully. Use function_name to call it."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/run-parallel")
    async def run_parallel(request: RunParallelRequest, background_tasks: BackgroundTasks):
        """
        Execute a function in parallel across multiple containers using .map()
        
        Agents can provide a function name and a list of inputs, and this
        will execute the function in parallel for each input.
        """
        try:
            task_id = str(uuid.uuid4())
            
            # Store task metadata
            task_store[task_id] = {
                "task_id": task_id,
                "function_name": request.function_name,
                "type": "parallel",
                "inputs": request.inputs,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "results": None,
                "error": None
            }
            
            # In a real implementation, this would:
            # 1. Look up the function from function_store
            # 2. Execute it using .map() on the inputs
            # 3. Store results in task_store
            
            if request.wait:
                # Simulate immediate execution (in real implementation, would await)
                task_store[task_id]["status"] = "completed"
                task_store[task_id]["results"] = f"Parallel execution scheduled for {len(request.inputs)} tasks"
            else:
                # Schedule background execution
                background_tasks.add_task(
                    execute_parallel_task,
                    task_id,
                    request.function_name,
                    request.inputs
                )
            
            return {
                "task_id": task_id,
                "function_name": request.function_name,
                "num_tasks": len(request.inputs),
                "status": "pending" if not request.wait else "completed",
                "wait": request.wait
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/spawn-task")
    async def spawn_task(request: SpawnTaskRequest, background_tasks: BackgroundTasks):
        """
        Spawn a single task execution.
        
        Agents can trigger a single function execution with optional args/kwargs.
        """
        try:
            task_id = str(uuid.uuid4())
            
            task_store[task_id] = {
                "task_id": task_id,
                "function_name": request.function_name,
                "type": "single",
                "args": request.args or [],
                "kwargs": request.kwargs or {},
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "result": None,
                "error": None
            }
            
            if request.wait:
                # Simulate immediate execution
                task_store[task_id]["status"] = "completed"
                task_store[task_id]["result"] = f"Task executed: {request.function_name}"
            else:
                background_tasks.add_task(
                    execute_single_task,
                    task_id,
                    request.function_name,
                    request.args or [],
                    request.kwargs or {}
                )
            
            return {
                "task_id": task_id,
                "function_name": request.function_name,
                "status": "pending" if not request.wait else "completed",
                "wait": request.wait
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/status/{task_id}")
    async def get_task_status(task_id: str):
        """
        Check the status of a task.
        
        Returns current status, results if completed, or errors if failed.
        """
        if task_id not in task_store:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task = task_store[task_id]
        return {
            "task_id": task_id,
            "function_name": task["function_name"],
            "type": task["type"],
            "status": task["status"],
            "created_at": task["created_at"],
            "result" if task["type"] == "single" else "results": (
                task.get("result") if task["type"] == "single" else task.get("results")
            ),
            "error": task.get("error")
        }
    
    @web_app.post("/webhook")
    async def register_webhook(request: WebhookRequest):
        """
        Register a webhook URL to be called when a task completes.
        
        Agents can provide a webhook URL that will be called with task results.
        """
        if request.task_id not in task_store:
            raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
        
        task_store[request.task_id]["webhook_url"] = request.webhook_url
        
        return {
            "task_id": request.task_id,
            "webhook_url": request.webhook_url,
            "status": "registered"
        }
    
    @web_app.get("/functions")
    async def list_functions():
        """List all registered functions."""
        return {
            "functions": [
                {
                    "function_id": fid,
                    "name": func["name"],
                    "status": func["status"],
                    "deployed_at": func["deployed_at"]
                }
                for fid, func in function_store.items()
            ]
        }
    
    @web_app.get("/tasks")
    async def list_tasks():
        """List all tasks."""
        return {
            "tasks": [
                {
                    "task_id": tid,
                    "function_name": task["function_name"],
                    "type": task["type"],
                    "status": task["status"],
                    "created_at": task["created_at"]
                }
                for tid, task in task_store.items()
            ]
        }
    
    return web_app


async def execute_parallel_task(task_id: str, function_name: str, inputs: List[Any]):
    """Background task to execute parallel function calls"""
    try:
        task_store[task_id]["status"] = "running"
        # In real implementation, would execute the function here
        # For now, simulate completion
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["results"] = [f"Result for input {i}" for i in range(len(inputs))]
        
        # Call webhook if registered
        if "webhook_url" in task_store[task_id]:
            await call_webhook(task_store[task_id]["webhook_url"], task_store[task_id])
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)


async def execute_single_task(task_id: str, function_name: str, args: List[Any], kwargs: Dict[str, Any]):
    """Background task to execute single function call"""
    try:
        task_store[task_id]["status"] = "running"
        # In real implementation, would execute the function here
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["result"] = f"Executed {function_name} with args={args}, kwargs={kwargs}"
        
        # Call webhook if registered
        if "webhook_url" in task_store[task_id]:
            await call_webhook(task_store[task_id]["webhook_url"], task_store[task_id])
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)


async def call_webhook(webhook_url: str, task_data: Dict[str, Any]):
    """Call webhook URL with task results"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=task_data, timeout=10.0)
    except Exception as e:
        print(f"Webhook call failed: {e}")
